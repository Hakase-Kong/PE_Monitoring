import os
import re
import json
import string
import threading
from datetime import datetime, timedelta
from urllib.parse import urlparse

import requests
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from zoneinfo import ZoneInfo

# =========================
# 전역 상태
# =========================
RUN_SEEN_URLS = set()
RUN_SEEN_TITLES = set()
last_run_info = {"ts": None, "sent": 0, "picked": 0, "note": ""}

# =========================
# 유틸
# =========================
KST = ZoneInfo("Asia/Seoul")

def now_kst():
    return datetime.now(tz=KST)

def normalize_title(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    # 기본 구두점 제거
    return t.translate(str.maketrans("", "", string.punctuation + "“”‘’"))

def get_domain(url):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def title_has_any(title, words):
    tl = title.lower()
    return any(w.lower() in tl for w in words)

def to_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ["1", "true", "yes", "y", "on"]
    return default

def get_config_path():
    candidates = [
        os.environ.get("CONFIG_PATH"),
        "/opt/render/project/src/config.json",  # Render 기본 경로
        os.path.abspath(os.path.join(os.path.dirname(__file__), "config.json")),
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

# 근무시간/휴일 정책
def is_weekend(dt):
    return dt.weekday() >= 5  # 5=토, 6=일

def is_holiday(dt, cfg):
    days = set(cfg.get("HOLIDAYS_KR", []))
    return dt.strftime("%Y-%m-%d") in days

def within_send_window(cfg):
    if not to_bool(cfg.get("ONLY_WORKING_HOURS", True), True):
        return True
    now = now_kst()
    hour_ok = 8 <= now.hour < 20
    if not hour_ok:
        return False
    if to_bool(cfg.get("SKIP_WEEKENDS", True), True) and is_weekend(now):
        return False
    if to_bool(cfg.get("SKIP_HOLIDAYS", True), True) and is_holiday(now, cfg):
        return False
    return True

# 점수
def score_item(item, cfg):
    score = 0.0
    # 도메인 가중치
    weights = cfg.get("DOMAIN_WEIGHTS", {})
    score += float(weights.get(get_domain(item["link"]), 1.0))
    # 워치리스트 부스트
    if title_has_any(item["title"], cfg.get("FIRM_WATCHLIST", [])):
        score += 1.5
    # 최신성 가점(24h 이내 최대 +2)
    if item.get("pub_dt"):
        hrs = (now_kst() - item["pub_dt"]).total_seconds() / 3600.0
        score += max(0.0, 2.0 - (hrs / 24.0))
    return score

# =========================
# 수집기
# =========================
def search_naver_news(query, display=30):
    cid = os.environ.get("NAVER_CLIENT_ID", "")
    csc = os.environ.get("NAVER_CLIENT_SECRET", "")
    if not cid or not csc:
        return []
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csc}
    params = {"query": query, "display": min(display, 100), "start": 1, "sort": "date"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        out = []
        for it in data.get("items", []):
            link = it.get("link") or it.get("originallink") or ""
            title = re.sub("<.*?>", "", it.get("title", ""))
            desc = re.sub("<.*?>", "", it.get("description", ""))
            try:
                pub_dt = datetime.strptime(it.get("pubDate", ""), "%a, %d %b %Y %H:%M:%S %z").astimezone(KST)
            except Exception:
                pub_dt = now_kst()
            out.append({"title": title, "desc": desc, "link": link, "source": "Naver", "pub_dt": pub_dt})
        return out
    except Exception:
        return []

def search_newsapi(query, page_size=30):
    key = os.environ.get("NEWSAPI_KEY", "")
    if not key:
        return []
    url = "https://newsapi.org/v2/everything"
    from_dt = (now_kst() - timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%S")
    params = {"q": query, "pageSize": min(page_size, 100), "from": from_dt,
              "language": "ko", "sortBy": "publishedAt", "apiKey": key}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        out = []
        for a in data.get("articles", []):
            try:
                pub_dt = datetime.fromisoformat(a.get("publishedAt","").replace("Z","+00:00")).astimezone(KST)
            except Exception:
                pub_dt = now_kst()
            out.append({
                "title": a.get("title",""),
                "desc": a.get("description",""),
                "link": a.get("url",""),
                "source": a.get("source",{}).get("name","NewsAPI"),
                "pub_dt": pub_dt
            })
        return out
    except Exception:
        return []

# =========================
# 필터링/집계
# =========================
def filter_and_rank(items, cfg):
    # 1) 제목 제외
    ex = cfg.get("EXCLUDE_TITLE_KEYWORDS", [])
    items = [x for x in items if not title_has_any(x["title"], ex)]
    # 2) 도메인 차단/허용
    allow = set([d.lower() for d in cfg.get("ALLOW_DOMAINS", [])])
    block = set([d.lower() for d in cfg.get("BLOCK_DOMAINS", [])])
    strict = to_bool(cfg.get("ALLOWLIST_STRICT", False), False)
    if strict and allow:
        items = [x for x in items if get_domain(x["link"]) in allow]
    items = [x for x in items if get_domain(x["link"]) not in block]
    # 3) 신선도
    recency_h = int(cfg.get("RECENCY_HOURS", 48))
    since = now_kst() - timedelta(hours=recency_h)
    items = [x for x in items if not x.get("pub_dt") or x["pub_dt"] >= since]
    # 4) 실행 내 중복 제거(제목/URL)
    seen_u, seen_t, uniq = set(), set(), []
    for x in items:
        u = x["link"].strip()
        t = normalize_title(x["title"])
        if u in seen_u or t in seen_t:
            continue
        seen_u.add(u); seen_t.add(t); uniq.append(x)
    items = uniq
    # 5) 실행 간 중복 제거(전역)
    global RUN_SEEN_URLS, RUN_SEEN_TITLES
    tmp = []
    for x in items:
        u = x["link"].strip()
        t = normalize_title(x["title"])
        if u in RUN_SEEN_URLS or t in RUN_SEEN_TITLES:
            continue
        tmp.append(x)
    items = tmp
    # 6) 스코어링
    items.sort(key=lambda z: score_item(z, cfg), reverse=True)
    return items

def make_batches(cfg):
    keywords = cfg.get("KEYWORDS", [])
    aliases = cfg.get("KEYWORD_ALIASES", {})
    page_size = int(cfg.get("PAGE_SIZE", 30))
    max_per = int(cfg.get("MAX_PER_KEYWORD", 10))

    picked_by_bucket = {}
    cross_seen_u, cross_seen_t = set(), set()

    for bucket in keywords:
        q_terms = [bucket] + aliases.get(bucket, [])
        q = " OR ".join(list(dict.fromkeys(q_terms)))
        raw = []
        raw += search_naver_news(q, display=page_size)
        raw += search_newsapi(q, page_size=page_size)

        filtered = filter_and_rank(raw, cfg)

        # 버킷 간 교차 중복 제거
        deduped = []
        for it in filtered:
            u = it["link"].strip()
            t = normalize_title(it["title"])
            if u in cross_seen_u or t in cross_seen_t:
                continue
            cross_seen_u.add(u); cross_seen_t.add(t)
            deduped.append(it)
        picked_by_bucket[bucket] = deduped[:max_per]
    return picked_by_bucket

def aggregate_top_k(batches, cfg):
    """버킷 전체를 합쳐 상위 K개로 정렬"""
    K = int(cfg.get("MAX_OVERALL", 10))
    flat = []
    for arr in batches.values():
        for it in arr:
            flat.append((score_item(it, cfg), it))
    flat.sort(key=lambda x: x[0], reverse=True)
    top = []
    seen_u, seen_t = set(), set()
    for _, it in flat:
        u = it["link"].strip()
        t = normalize_title(it["title"])
        if u in seen_u or t in seen_t:
            continue
        seen_u.add(u); seen_t.add(t)
        top.append(it)
        if len(top) >= K:
            break
    return top

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

def display_source(it):
    d = get_domain(it.get("link",""))
    return d or (it.get("source") or "")

def format_overall_message(items):
    if not items: return None
    lines = ["📌 PE 동향 뉴스 (Top 10)"]
    for it in items:
        when = it.get("pub_dt")
        ts = when.strftime("%Y-%m-%d %H:%M") if when else ""
        src = display_source(it)
        lines.append(f"• {it['title']} ({it['link']}) — {src} ({ts})")
    return "\n".join(lines)

def format_bucket_message(bucket, items):
    if not items: return None
    lines = [f"📌 PE 동향 뉴스 ({bucket})"]
    for it in items[:10]:
        when = it.get("pub_dt")
        ts = when.strftime("%Y-%m-%d %H:%M") if when else ""
        src = display_source(it)
        lines.append(f"• {it['title']} ({it['link']}) — {src} ({ts})")
    return "\n".join(lines)

def transmit_once(cfg, preview=False, ignore_hours=False):
    global RUN_SEEN_URLS, RUN_SEEN_TITLES, last_run_info

    # 근무시간/휴일 제한 (스케줄러에서만 적용)
    if not ignore_hours and not within_send_window(cfg):
        last_run_info = {"ts": now_kst(), "sent": 0, "picked": 0, "note": "off_hours_or_holiday"}
        return {"picked": 0, "sent": 0, "skipped": "off_hours_or_holiday"}

    batches = make_batches(cfg)
    disable_preview = to_bool(cfg.get("TELEGRAM_DISABLE_PREVIEW", True), True)

    # 집계 모드 여부
    aggregate = to_bool(cfg.get("AGGREGATE_MODE", True), True)

    if aggregate:
        top = aggregate_top_k(batches, cfg)
        total_picked = len(top)
        total_sent = 0
        if not preview:
            # 전역 중복 리스트 업데이트
            for it in top:
                RUN_SEEN_URLS.add(it["link"].strip())
                RUN_SEEN_TITLES.add(normalize_title(it["title"]))
            msg = format_overall_message(top)
            if msg:
                ok, err = send_telegram_message(msg, disable_preview=disable_preview)
                if ok: total_sent = len(top)
                else: last_run_info = {"ts": now_kst(), "sent": total_sent, "picked": total_picked, "note": f"TG_FAIL:{err}"}
        last_run_info = {"ts": now_kst(), "sent": total_sent, "picked": total_picked, "note": ""}
        return {"picked": total_picked, "sent": total_sent, "skipped": None}

    # (비집계 모드) 버킷별 전송
    total_picked = sum(len(v) for v in batches.values())
    total_sent = 0
    if not preview:
        for arr in batches.values():
            for it in arr:
                RUN_SEEN_URLS.add(it["link"].strip())
                RUN_SEEN_TITLES.add(normalize_title(it["title"]))
        for bucket, arr in batches.items():
            msg = format_bucket_message(bucket, arr)
            if not msg: continue
            ok, err = send_telegram_message(msg, disable_preview=disable_preview)
            if ok: total_sent += len(arr)
            else: last_run_info = {"ts": now_kst(), "sent": total_sent, "picked": total_picked, "note": f"TG_FAIL:{err}"}

    last_run_info = {"ts": now_kst(), "sent": total_sent, "picked": total_picked, "note": ""}
    return {"picked": total_picked, "sent": total_sent, "skipped": None}

# =========================
# 스케줄러
# =========================
SCHED = BackgroundScheduler(timezone="Asia/Seoul")
JOB_ID = "pe_monitoring_job"
JOB_LOCK = threading.Lock()

def start_schedule(cfg):
    with JOB_LOCK:
        if SCHED.get_job(JOB_ID):
            return
        interval = int(cfg.get("TRANSMIT_INTERVAL_MIN", 60))
        SCHED.add_job(lambda: transmit_once(cfg, preview=False, ignore_hours=False),
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
        st.button("구성 리로드", on_click=lambda: load_config.clear())
        st.write("**KEYWORDS (읽기전용)**")
        st.code(", ".join(cfg.get("KEYWORDS", [])) or "(none)")
        st.number_input("페이지당 수집 수", 5, 100, int(cfg.get("PAGE_SIZE", 30)), 5, key="ps", disabled=True)
        st.number_input("전송 건수 제한(키워드별)", 1, 50, int(cfg.get("MAX_PER_KEYWORD", 10)), 1, key="mx", disabled=True)
        st.number_input("전송 주기(분)", 10, 360, int(cfg.get("TRANSMIT_INTERVAL_MIN", 60)), 10, key="iv", disabled=True)
        st.number_input("신선도(최근 N시간)", 6, 168, int(cfg.get("RECENCY_HOURS", 48)), 6, key="rc", disabled=True)
        st.checkbox("업무시간(08~20 KST) 내 전송", value=to_bool(cfg.get("ONLY_WORKING_HOURS", True)), disabled=True)
        st.checkbox("주말 미전송", value=to_bool(cfg.get("SKIP_WEEKENDS", True)), disabled=True)
        st.checkbox("공휴일 미전송", value=to_bool(cfg.get("SKIP_HOLIDAYS", True)), disabled=True)
        st.checkbox("링크 프리뷰 비활성화", value=to_bool(cfg.get("TELEGRAM_DISABLE_PREVIEW", True)), disabled=True)
        st.checkbox("집계 모드(Top-K 단일 메시지)", value=to_bool(cfg.get("AGGREGATE_MODE", True)), disabled=True)
    else:
        st.error("config.json을 읽을 수 없습니다. 경로/JSON 문법을 확인하세요.")

st.title("📬 PE 동향 뉴스 → Telegram 자동 전송")
st.caption("Streamlit + Naver/NewsAPI + Telegram + APScheduler")

col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    if st.button("지금 한 번 실행(미리보기)", use_container_width=True):
        if not cfg:
            st.error("config.json을 불러오지 못했습니다.")
        else:
            res = transmit_once(cfg, preview=True, ignore_hours=True)
            st.success(f"완료: {res['picked']}건 미리보기, 0건 전송(미리보기)")
with col2:
    if st.button("지금 한 번 전송", use_container_width=True):
        if not cfg:
            st.error("config.json을 불러오지 못했습니다.")
        else:
            res = transmit_once(cfg, preview=False, ignore_hours=True)
            st.success(f"전송 완료: {res['sent']}건 전송 / 선별 {res['picked']}건")
with col3:
    if st.button("스케줄 시작", use_container_width=True):
        if not cfg:
            st.error("config.json을 불러오지 못했습니다.")
        else:
            start_schedule(cfg)
            st.info("스케줄 시작됨.")
with col4:
    if st.button("스케줄 중지", use_container_width=True):
        stop_schedule()
        st.info("스케줄 중지됨.")

st.divider()
st.subheader("상태")
st.write(f"Scheduler 실행 중: **{scheduler_running()}**")
if last_run_info["ts"]:
    note = f" (note: {last_run_info['note']})" if last_run_info.get("note") else ""
    st.write(f"마지막 수행 시각: **{last_run_info['ts'].strftime('%Y-%m-%d %H:%M:%S')}** / 선별: {last_run_info['picked']} / 전송: {last_run_info['sent']}{note}")

# 미리보기
if cfg:
    st.subheader("미리보기")
    preview_batches = make_batches(cfg)
    if to_bool(cfg.get("AGGREGATE_MODE", True), True):
        top = aggregate_top_k(preview_batches, cfg)
        with st.expander(f"Top {int(cfg.get('MAX_OVERALL',10))} — {len(top)}건", expanded=True):
            if not top:
                st.caption("결과 없음")
            for it in top:
                ts = it["pub_dt"].strftime("%Y-%m-%d %H:%M") if it.get("pub_dt") else ""
                src = display_source(it)
                st.markdown(f"- [{it['title']}]({it['link']})  \n"
                            f"  <span style='font-size:12px;color:#888'>{src} — {ts}</span>", unsafe_allow_html=True)
    else:
        for bucket, arr in preview_batches.items():
            with st.expander(f"{bucket} — {len(arr[:10])}건", expanded=False):
                if not arr:
                    st.caption("결과 없음")
                for it in arr[:10]:
                    ts = it["pub_dt"].strftime("%Y-%m-%d %H:%M") if it.get("pub_dt") else ""
                    src = display_source(it)
                    st.markdown(f"- [{it['title']}]({it['link']})  \n"
                                f"  <span style='font-size:12px;color:#888'>{src} — {ts}</span>", unsafe_allow_html=True)
