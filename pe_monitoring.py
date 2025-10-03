import os
import re
import json
import time
import math
import queue
import string
import threading
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import requests
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler

# =========================
# 전역 상태 (런타임 교차 중복 제거)
# =========================
RUN_SEEN_URLS = set()
RUN_SEEN_TITLES = set()
last_run_info = {"ts": None, "sent": 0, "picked": 0}

# =========================
# 유틸
# =========================
KST = timezone(timedelta(hours=9))

def now_kst():
    return datetime.now(tz=KST)

def normalize_title(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = t.translate(str.maketrans("", "", string.punctuation + "“”‘’"))
    return t

def within_working_hours():
    # 08~20 KST
    h = now_kst().hour
    return 8 <= h < 20

def get_config_path():
    candidates = [
        os.environ.get("CONFIG_PATH"),
        "/opt/render/project/src/config.json",  # Render 기본 경로
        os.path.abspath(os.path.join(os.path.dirname(__file__), "config.json"))
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

@st.cache_data(show_spinner=False)
def load_config():
    p = get_config_path()
    if not p:
        return None, {"path": None, "exists": False}
    try:
        with open(p, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg, {"path": p, "exists": True}
    except Exception as e:
        return None, {"path": p, "exists": False, "err": str(e)}

def env_ok(env_name):
    return bool(os.environ.get(env_name, "").strip())

def to_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ["1","true","yes","y","on"]
    return default

def get_domain(url):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def title_has_any(title, words):
    t = title.lower()
    return any(w.lower() in t for w in words)

def score_item(item, cfg):
    """도메인 가중치 + 워치리스트 + 최신성 스코어"""
    score = 0.0
    domain = get_domain(item["link"])
    weights = cfg.get("DOMAIN_WEIGHTS", {})
    score += float(weights.get(domain, 1.0))

    # 워치리스트 부스트
    if title_has_any(item["title"], cfg.get("FIRM_WATCHLIST", [])):
        score += 1.5

    # 최신성(가까울수록 가점)
    if item.get("pub_dt"):
        hrs = (now_kst() - item["pub_dt"]).total_seconds() / 3600.0
        score += max(0.0, 2.0 - (hrs / 24.0))  # 24h 이내면 최대 +2 → 점차 감소
    return score

# =========================
# 뉴스 수집기
# =========================
def search_naver_news(query, display=30):
    """
    Naver Search API (뉴스)
    ENV: NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
    """
    cid = os.environ.get("NAVER_CLIENT_ID", "")
    csc = os.environ.get("NAVER_CLIENT_SECRET", "")
    if not cid or not csc:
        return []

    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": cid,
        "X-Naver-Client-Secret": csc,
    }
    params = {
        "query": query,
        "display": min(display, 100),
        "start": 1,
        "sort": "date",
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        items = []
        for it in data.get("items", []):
            # 일부 결과는 originallink가 없기도 함
            link = it.get("link") or it.get("originallink") or ""
            title = re.sub("<.*?>", "", it.get("title", ""))
            desc = re.sub("<.*?>", "", it.get("description", ""))
            # Naver는 pubDate 예: 'Fri, 03 Oct 2025 08:20:00 +0900'
            pub_dt = None
            try:
                pub_dt = datetime.strptime(it.get("pubDate",""), "%a, %d %b %Y %H:%M:%S %z").astimezone(KST)
            except Exception:
                pub_dt = now_kst()
            items.append({
                "title": title,
                "desc": desc,
                "link": link,
                "source": "Naver",
                "pub_dt": pub_dt
            })
        return items
    except Exception:
        return []

def search_newsapi(query, page_size=30):
    """
    NewsAPI (선택)
    ENV: NEWSAPI_KEY
    """
    key = os.environ.get("NEWSAPI_KEY", "")
    if not key:
        return []
    url = "https://newsapi.org/v2/everything"
    from_dt = (now_kst() - timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%S")
    params = {
        "q": query,
        "pageSize": min(page_size, 100),
        "from": from_dt,
        "language": "ko",
        "sortBy": "publishedAt",
        "apiKey": key
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        items = []
        for a in data.get("articles", []):
            pub = a.get("publishedAt")
            try:
                pub_dt = datetime.fromisoformat(pub.replace("Z","+00:00")).astimezone(KST)
            except Exception:
                pub_dt = now_kst()
            items.append({
                "title": a.get("title",""),
                "desc": a.get("description",""),
                "link": a.get("url",""),
                "source": a.get("source",{}).get("name","NewsAPI"),
                "pub_dt": pub_dt
            })
        return items
    except Exception:
        return []

# =========================
# 필터링/집계
# =========================
def filter_and_rank(items, cfg):
    # 1) 제목 제외 키워드
    ex_words = cfg.get("EXCLUDE_TITLE_KEYWORDS", [])
    items = [x for x in items if not title_has_any(x["title"], ex_words)]

    # 2) 포함 키워드(있으면 +, 없으면 통과 — 너무 강하게 제한하지 않음)
    inc_words = cfg.get("INCLUDE_TITLE_KEYWORDS", [])
    if inc_words:
        kept = []
        for x in items:
            if title_has_any(x["title"], inc_words):
                kept.append(x)
            else:
                # 포함어가 없더라도 워치리스트/도메인 점수로 올라올 수 있게 low priority로 남김
                kept.append(x)
        items = kept

    # 3) 도메인 허용/차단
    allow = set([d.lower() for d in cfg.get("ALLOW_DOMAINS", [])])
    block = set([d.lower() for d in cfg.get("BLOCK_DOMAINS", [])])
    if allow:
        items = [x for x in items if get_domain(x["link"]) in allow or "naver.com" in get_domain(x["link"])]
    items = [x for x in items if get_domain(x["link"]) not in block]

    # 4) 최신성
    recency_h = int(cfg.get("RECENCY_HOURS", 48))
    since_ts = now_kst() - timedelta(hours=recency_h)
    items = [x for x in items if not x.get("pub_dt") or x["pub_dt"] >= since_ts]

    # 5) 실행 내 중복제거 (URL/제목)
    seen_url = set()
    seen_title = set()
    uniq = []
    for x in items:
        u = x["link"].strip()
        t = normalize_title(x["title"])
        if u in seen_url: 
            continue
        if t in seen_title:
            continue
        seen_url.add(u)
        seen_title.add(t)
        uniq.append(x)
    items = uniq

    # 6) 실행 간(글로벌) 중복제거
    global RUN_SEEN_URLS, RUN_SEEN_TITLES
    tmp = []
    for x in items:
        u = x["link"].strip()
        t = normalize_title(x["title"])
        if u in RUN_SEEN_URLS or t in RUN_SEEN_TITLES:
            continue
        tmp.append(x)
    items = tmp

    # 7) 스코어링 정렬
    items.sort(key=lambda z: score_item(z, cfg), reverse=True)
    return items

def make_batches(cfg):
    """키워드별 검색 -> 필터/랭크 -> 키워드 간 교차중복 제거"""
    keywords = cfg.get("KEYWORDS", [])
    aliases = cfg.get("KEYWORD_ALIASES", {})
    page_size = int(cfg.get("PAGE_SIZE", 30))
    max_per_key = int(cfg.get("MAX_PER_KEYWORD", 10))

    picked_by_bucket = {}
    cross_seen_u = set()
    cross_seen_t = set()

    for bucket in keywords:
        q_terms = [bucket]
        q_terms += aliases.get(bucket, [])
        q = " OR ".join(list(dict.fromkeys(q_terms)))  # dup 제거

        raw = []
        # Naver 우선
        raw += search_naver_news(q, display=page_size)
        # 이후 NewsAPI 보강(있으면)
        raw += search_newsapi(q, page_size=page_size)

        filtered = filter_and_rank(raw, cfg)

        # 키워드 간 교차 중복 제거
        deduped = []
        for it in filtered:
            u = it["link"].strip()
            t = normalize_title(it["title"])
            if u in cross_seen_u or t in cross_seen_t:
                continue
            cross_seen_u.add(u)
            cross_seen_t.add(t)
            deduped.append(it)

        picked_by_bucket[bucket] = deduped[:max_per_key]

    return picked_by_bucket

# =========================
# Telegram
# =========================
def send_telegram_message(text, disable_preview=True):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False, "TELEGRAM env 미설정"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": bool(disable_preview),
    }
    try:
        r = requests.post(url, data=payload, timeout=10)
        ok = r.status_code == 200 and r.json().get("ok", False)
        if not ok:
            return False, f"TG 오류: {r.text[:200]}"
        return True, "ok"
    except Exception as e:
        return False, str(e)

def format_bucket_message(bucket, items):
    if not items:
        return None
    lines = [f"📌 PE 동향 뉴스 ({bucket})"]
    for it in items[:10]:
        src = it.get("source","")
        when = it.get("pub_dt")
        ts = when.strftime("%Y-%m-%d %H:%M") if when else ""
        lines.append(f"• {it['title']} ({it['link']}) — {src} ({ts})")
    return "\n".join(lines)

def transmit_once(cfg, preview=False):
    global RUN_SEEN_URLS, RUN_SEEN_TITLES, last_run_info
    # 근무시간 제한
    if to_bool(cfg.get("ONLY_WORKING_HOURS", False), False):
        if not within_working_hours():
            return {"picked": 0, "sent": 0, "skipped": "off_hours"}

    batches = make_batches(cfg)
    disable_preview = to_bool(cfg.get("TELEGRAM_DISABLE_PREVIEW", True), True)

    total_picked = sum(len(v) for v in batches.values())
    total_sent = 0

    if not preview:
        # 실행 간 중복 기록 업데이트
        for arr in batches.values():
            for it in arr:
                RUN_SEEN_URLS.add(it["link"].strip())
                RUN_SEEN_TITLES.add(normalize_title(it["title"]))

        # 전송
        for bucket, arr in batches.items():
            msg = format_bucket_message(bucket, arr)
            if not msg:
                continue
            ok, _ = send_telegram_message(msg, disable_preview=disable_preview)
            if ok:
                total_sent += len(arr)

    last_run_info = {"ts": now_kst(), "sent": total_sent, "picked": total_picked}
    return {"picked": total_picked, "sent": total_sent, "skipped": None}

# =========================
# 스케줄러
# =========================
SCHED = BackgroundScheduler(timezone=str(KST))
JOB_ID = "pe_monitoring_job"
JOB_LOCK = threading.Lock()

def start_schedule(cfg):
    with JOB_LOCK:
        if SCHED.get_job(JOB_ID):
            return
        interval = int(cfg.get("TRANSMIT_INTERVAL_MIN", 60))
        SCHED.add_job(lambda: transmit_once(cfg, preview=False),
                      "interval", minutes=interval, id=JOB_ID, max_instances=1)
        if not SCHED.running:
            SCHED.start()

def stop_schedule():
    with JOB_LOCK:
        job = SCHED.get_job(JOB_ID)
        if job:
            job.remove()

def scheduler_running():
    return SCHED.get_job(JOB_ID) is not None

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="PE 동향 뉴스 → Telegram", page_icon="📨", layout="wide")

cfg, cfg_meta = load_config()

with st.sidebar:
    st.markdown("### 자격증명 / 설정")
    st.caption(f"CONFIG 경로:\n`{cfg_meta.get('path') or '미발견'}` / 존재: **{cfg_meta.get('exists', False)}**")
    st.divider()
    st.text_input("NewsAPI Key (선택)", value=("●" * 8 if env_ok("NEWSAPI_KEY") else ""), disabled=True)
    st.text_input("Naver Client ID (선택)", value=("●" * 8 if env_ok("NAVER_CLIENT_ID") else ""), disabled=True)
    st.text_input("Naver Client Secret (선택)", value=("●" * 8 if env_ok("NAVER_CLIENT_SECRET") else ""), disabled=True)
    st.text_input("Telegram Bot Token", value=("●" * 8 if env_ok("TELEGRAM_BOT_TOKEN") else ""), disabled=True)
    st.text_input("Telegram Chat ID", value=os.environ.get("TELEGRAM_CHAT_ID",""), disabled=True)

    st.divider()
    st.markdown("### config.json")
    if cfg:
        st.button("구성 리로드", on_click=lambda: load_config.clear())  # cache 초기화
        st.write("**KEYWORDS (읽기전용)**")
        st.code(", ".join(cfg.get("KEYWORDS", [])) or "(none)")
        st.number_input("페이지당 수집 수", min_value=5, max_value=100, step=5,
                        value=int(cfg.get("PAGE_SIZE", 30)), key="ps", disabled=True)
        st.number_input("전송 건수 제한(키워드별)", min_value=1, max_value=50, step=1,
                        value=int(cfg.get("MAX_PER_KEYWORD", 10)), key="mx", disabled=True)
        st.number_input("전송 주기(분)", min_value=10, max_value=360, step=10,
                        value=int(cfg.get("TRANSMIT_INTERVAL_MIN", 60)), key="iv", disabled=True)
        st.number_input("신선도(최근 N시간)", min_value=6, max_value=168, step=6,
                        value=int(cfg.get("RECENCY_HOURS", 48)), key="rc", disabled=True)
        st.checkbox("업무시간(08~20 KST) 내 전송", value=to_bool(cfg.get("ONLY_WORKING_HOURS", True)), disabled=True)
        st.checkbox("링크 프리뷰 비활성화", value=to_bool(cfg.get("TELEGRAM_DISABLE_PREVIEW", True)), disabled=True)
    else:
        st.error("config.json을 읽을 수 없습니다. 경로/JSON 문법을 확인하세요.")

st.title("📬 PE 동향 뉴스 → Telegram 자동 전송")
st.caption("Streamlit + NewsAPI/Naver + Telegram + APScheduler")

col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("지금 한 번 실행", use_container_width=True):
        if not cfg:
            st.error("config.json을 불러오지 못했습니다.")
        else:
            res = transmit_once(cfg, preview=True)
            st.success(f"완료: {res['picked']}건 미리보기, {0}건 전송(미리보기 모드)")
with col2:
    if st.button("스케줄 시작", use_container_width=True):
        if not cfg:
            st.error("config.json을 불러오지 못했습니다.")
        else:
            start_schedule(cfg)
with col3:
    if st.button("스케줄 중지", use_container_width=True):
        stop_schedule()

st.divider()
st.subheader("상태")

st.write(f"Scheduler 실행 중: **{scheduler_running()}**")
if last_run_info["ts"]:
    st.write(f"마지막 수행 시각: **{last_run_info['ts'].strftime('%Y-%m-%d %H:%M:%S')}**")
st.info("config.json의 KEYWORDS가 비어 있습니다." if (cfg and not cfg.get("KEYWORDS")) else "")

# 미리보기 블록
if cfg:
    st.subheader("미리보기: 최신 10건")
    preview = make_batches(cfg)
    # 키워드 섹션별로 상위 10개만 표시
    for bucket in cfg.get("KEYWORDS", []):
        items = preview.get(bucket, [])[:10]
        if not items:
            continue
        with st.expander(f"{bucket} — {len(items)}건", expanded=False):
            for it in items:
                ts = it["pub_dt"].strftime("%Y-%m-%d %H:%M") if it.get("pub_dt") else ""
                st.markdown(f"- [{it['title']}]({it['link']})  \n  <span style='font-size:12px;color:#888'>{it.get('source','')} — {ts}</span>", unsafe_allow_html=True)
