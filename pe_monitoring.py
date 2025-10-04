import os
import re
import json
import time
import hashlib
import logging
import requests
import datetime as dt
from urllib.parse import urlparse, parse_qs
from typing import Optional, List, Dict, Set

import pytz
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler

# -------------------------
# 기본 설정 / 전역 상태
# -------------------------
APP_TZ = pytz.timezone("Asia/Seoul")
DEFAULT_CONFIG_PATH = os.getenv("PE_CFG", "config.json")
CACHE_FILE = os.getenv("PE_SENT_CACHE", "sent_cache.json")  # 전송 이력 저장(중복 방지)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pe_monitor")

# 스케줄러 잡이 참조할 "현재" 구성/환경 (start_schedule()에서 갱신)
CURRENT_CFG_PATH = DEFAULT_CONFIG_PATH
CURRENT_ENV = {
    "NEWSAPI_KEY": os.getenv("NEWSAPI_KEY", ""),
    "NAVER_CLIENT_ID": os.getenv("NAVER_CLIENT_ID", ""),
    "NAVER_CLIENT_SECRET": os.getenv("NAVER_CLIENT_SECRET", ""),
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", ""),
}

# -------------------------
# 공용 유틸
# -------------------------
def load_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("config 로드 실패(%s): %s", path, e)
        return {}

def now_kst() -> dt.datetime:
    return dt.datetime.now(APP_TZ)

def clamp(n, lo, hi):
    return max(lo, min(hi, n))

def domain_of(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").replace("www.", "")
    except Exception:
        return ""

def _naver_sid(url: str) -> Optional[str]:
    try:
        q = parse_qs(urlparse(url).query).get("sid", [])
        return q[0] if q else None
    except Exception:
        return None

def sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

# -------------------------
# 전송 캐시 (중복 방지)
# -------------------------
def load_sent_cache() -> Set[str]:
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            arr = json.load(f)
            return set(arr if isinstance(arr, list) else [])
    except Exception:
        return set()

def save_sent_cache(hashes: Set[str]) -> None:
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(list(hashes)), f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("전송 캐시 저장 실패: %s", e)

# -------------------------
# 외부 API (Naver / NewsAPI)
# -------------------------
def search_naver_news(keyword: str, client_id: str, client_secret: str, recency_hours=72) -> List[dict]:
    if not client_id or not client_secret or not keyword:
        return []
    base = "https://openapi.naver.com/v1/search/news.json"
    params = {"query": keyword, "display": 30, "sort": "date"}
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    try:
        r = requests.get(base, params=params, headers=headers, timeout=12)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        res = []
        cutoff = now_kst() - dt.timedelta(hours=recency_hours)
        for it in items:
            link = it.get("link") or it.get("originallink") or ""
            if not link:
                continue
            pubdate = it.get("pubDate")
            try:
                pub_kst = dt.datetime.strptime(pubdate, "%a, %d %b %Y %H:%M:%S %z")
            except Exception:
                pub_kst = now_kst()
            if pub_kst < cutoff:
                continue
            title = re.sub("<.*?>", "", it.get("title") or "")
            res.append({
                "title": title.strip(),
                "url": link.strip(),
                "source": domain_of(link),
                "publishedAt": pub_kst.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "origin_keyword": keyword,
                "provider": "naver",
            })
        return res
    except Exception as e:
        log.warning("Naver 오류(%s): %s", keyword, e)
        return []

def search_newsapi(query: str, page_size: int, api_key: str, from_hours: int = 72, cfg: dict | None = None) -> List[dict]:
    if not api_key or not query:
        return []
    base = "https://newsapi.org/v2/everything"
    from_dt = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=from_hours))
    params = {
        "q": (cfg.get("NEWSAPI_QUERY") if (cfg and cfg.get("NEWSAPI_QUERY")) else query),
        "searchIn": "title",
        "pageSize": clamp(page_size, 10, 100),
        "language": "ko",
        "sortBy": "publishedAt",
        "from": from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "apiKey": api_key,
    }
    if cfg and cfg.get("NEWSAPI_DOMAINS"):
        params["domains"] = ",".join(cfg["NEWSAPI_DOMAINS"])
    try:
        r = requests.get(base, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        arts = data.get("articles", [])
        res = []
        for a in arts:
            title = (a.get("title") or "").strip()
            url = (a.get("url") or "").strip()
            if not title or not url:
                continue
            res.append({
                "title": title,
                "url": url,
                "source": domain_of(url) or (a.get("source", {}) or {}).get("name", ""),
                "publishedAt": (a.get("publishedAt") or "").replace(".000Z", "Z"),
                "origin_keyword": "_newsapi",
                "provider": "newsapi",
            })
        return res
    except Exception as e:
        log.warning("NewsAPI 오류: %s", e)
        return []

# -------------------------
# 필터/정렬/중복제거
# -------------------------
def should_drop(item: dict, cfg: dict) -> bool:
    url = item.get("url", "")
    title = (item.get("title") or "").strip()
    if not url or not title:
        return True

    src = domain_of(url)
    allow = set(cfg.get("ALLOW_DOMAINS", []) or [])
    block = set(cfg.get("BLOCK_DOMAINS", []) or [])
    allow_strict = bool(cfg.get("ALLOWLIST_STRICT", False))

    if src in block:
        return True
    if allow_strict and allow and (src not in allow):
        return True

    # 네이버 섹션 제한(예: 경제면=101)
    if "naver.com" in src:
        sids = set(cfg.get("NAVER_ALLOW_SIDS", []) or [])
        if sids:
            sid = _naver_sid(url)
            if sid not in sids:
                return True

    # 제목 포함 키워드 / 제외 키워드
    include = (cfg.get("INCLUDE_TITLE_KEYWORDS", []) or [])
    if include and not any(w.lower() in title.lower() for w in include):
        return True
    for w in (cfg.get("EXCLUDE_TITLE_KEYWORDS", []) or []):
        if w and w.lower() in title.lower():
            return True

    return False

def dedup(items: List[dict], threshold: float = 0.82) -> List[dict]:
    def norm(s: str) -> str:
        s = re.sub(r'[\[\]\(\)【】“”"\'<>]', " ", s or "")
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    seen = []
    out = []
    for it in items:
        t = norm(it.get("title", ""))
        dup = False
        for s in seen:
            a, b = set(t.split()), set(s.split())
            if not a or not b:
                continue
            jac = len(a & b) / max(1, len(a | b))
            if jac >= threshold:
                dup = True
                break
        if not dup:
            seen.append(t)
            out.append(it)
    return out

def score_item(item: dict, cfg: dict) -> float:
    # 간단 가중치: 도메인 가중치 + 신선도
    src = domain_of(item.get("url", ""))
    score = float((cfg.get("DOMAIN_WEIGHTS", {}) or {}).get(src, 1.0))
    try:
        ts = item.get("publishedAt")
        pub = dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except Exception:
        pub = now_kst()
    hours_ago = (now_kst() - pub.astimezone(APP_TZ)).total_seconds() / 3600.0
    score += max(0.0, 6.0 - (hours_ago / 8.0))
    return score

def rank_filtered(items: List[dict], cfg: dict) -> List[dict]:
    arr = [it for it in items if not should_drop(it, cfg)]
    for it in arr:
        it["_score"] = score_item(it, cfg)
    arr.sort(key=lambda x: x["_score"], reverse=True)
    return dedup(arr)

# -------------------------
# 수집/전송
# -------------------------
def collect_all(cfg: dict, env: dict) -> List[dict]:
    keywords = cfg.get("KEYWORDS", []) or []
    page_size = int(cfg.get("PAGE_SIZE", 30))
    recency_hours = int(cfg.get("RECENCY_HOURS", 72))

    all_items: List[dict] = []

    # Naver
    for kw in keywords:
        batch = search_naver_news(kw, env.get("NAVER_CLIENT_ID", ""), env.get("NAVER_CLIENT_SECRET", ""), recency_hours=recency_hours)
        all_items += batch
        time.sleep(0.2)

    # NewsAPI (선택)
    if env.get("NEWSAPI_KEY") and keywords:
        query = " OR ".join(keywords)
        batch = search_newsapi(query, page_size=page_size, api_key=env["NEWSAPI_KEY"], from_hours=recency_hours, cfg=cfg)
        all_items += batch

    return all_items

def format_telegram_text(items: List[dict]) -> str:
    if not items:
        return "📭 신규 뉴스 없음"
    lines = ["📌 <b>국내 PE 동향 관련 뉴스</b>"]
    for it in items:
        t = it.get("title", "").strip()
        u = it.get("url", "")
        try:
            pub = dt.datetime.strptime(it["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).astimezone(APP_TZ)
            when = pub.strftime("%Y-%m-%d %H:%M")
        except Exception:
            when = "-"
        # 제목을 하이퍼링크로, 출처 도메인 표시는 제거
        lines.append(f"• <a href=\"{u}\">{t}</a> ({when})")
    return "\n".join(lines)

def send_telegram(bot_token: str, chat_id: str, text: str) -> bool:
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=12)
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning("텔레그램 전송 실패: %s", e)
        return False

def transmit_once(cfg: dict, env: dict, preview=False) -> dict:
    # 전체 수집 → 필터/정렬 → 전체 리스트
    all_items = collect_all(cfg, env)
    ranked = rank_filtered(all_items, cfg)

    if preview:
        return {"count": len(ranked), "items": ranked}

    # 신규만 전송 (캐시 기준)
    cache = load_sent_cache()
    new_items = [it for it in ranked if sha1(it.get("url", "")) not in cache]

    # 신규 없으면 알림
    if not new_items:
        ok = send_telegram(env.get("TELEGRAM_BOT_TOKEN", ""), env.get("TELEGRAM_CHAT_ID", ""), "📭 신규 뉴스 없음")
        return {"count": 0, "items": []}

    # 텔레그램 4096자 제한 대비 — 30개 단위로 배치 전송(필요시)
    BATCH = 30
    sent_any = False
    for i in range(0, len(new_items), BATCH):
        chunk = new_items[i:i+BATCH]
        text = format_telegram_text(chunk)
        ok = send_telegram(env.get("TELEGRAM_BOT_TOKEN", ""), env.get("TELEGRAM_CHAT_ID", ""), text)
        sent_any = sent_any or ok
        # 너무 빠른 연속 전송 방지
        time.sleep(0.6)

    if sent_any:
        cache |= {sha1(it.get("url", "")) for it in new_items}
        save_sent_cache(cache)

    return {"count": len(new_items), "items": new_items}

# -------------------------
# 스케줄러 (rerun-safe)
# -------------------------
@st.cache_resource(show_spinner=False)
def get_scheduler() -> BackgroundScheduler:
    sched = BackgroundScheduler(timezone=APP_TZ)
    sched.start()
    return sched

def scheduled_job():
    cfg = load_config(CURRENT_CFG_PATH)
    try:
        transmit_once(cfg, CURRENT_ENV, preview=False)
    except Exception as e:
        log.exception("스케줄 작업 실패: %s", e)

def ensure_interval_job(sched: BackgroundScheduler, minutes: int):
    job_id = "pe_news_job"
    try:
        sched.remove_job(job_id)
    except Exception:
        pass
    sched.add_job(scheduled_job, "interval", minutes=minutes, id=job_id,
                  replace_existing=True, next_run_time=now_kst())

def is_running(sched: BackgroundScheduler) -> bool:
    try:
        return any(j.id == "pe_news_job" for j in sched.get_jobs())
    except Exception:
        return False

# 스케줄 시작(버튼 핸들러용) — 여기서만 global 사용
def start_schedule(cfg_path: str, env: dict, minutes: int):
    global CURRENT_CFG_PATH, CURRENT_ENV
    CURRENT_CFG_PATH = cfg_path
    CURRENT_ENV = env
    sched = get_scheduler()
    ensure_interval_job(sched, minutes)
    # 즉시 1회 수행
    scheduled_job()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="PE 동향 뉴스 모니터링", page_icon="📰", layout="wide")

# Config / 자격증명
cfg_path = st.sidebar.text_input("config.json 경로", value=DEFAULT_CONFIG_PATH)
cfg = load_config(cfg_path)
st.sidebar.caption(f"Config 로드 상태: {'✅' if cfg else '❌'}  · 경로: {cfg_path}")

naver_id = st.sidebar.text_input("Naver Client ID", type="password", value=os.getenv("NAVER_CLIENT_ID", ""))
naver_secret = st.sidebar.text_input("Naver Client Secret", type="password", value=os.getenv("NAVER_CLIENT_SECRET", ""))
newsapi_key = st.sidebar.text_input("NewsAPI Key (선택)", type="password", value=os.getenv("NEWSAPI_KEY", ""))
bot_token = st.sidebar.text_input("Telegram Bot Token", type="password", value=os.getenv("TELEGRAM_BOT_TOKEN", ""))
chat_id = st.sidebar.text_input("Telegram Chat ID (채널/그룹)", value=os.getenv("TELEGRAM_CHAT_ID", ""))

# 파라미터
st.sidebar.divider()
st.sidebar.subheader("전송/수집 파라미터")
cfg["PAGE_SIZE"] = int(st.sidebar.number_input("페이지당 수집 수", min_value=10, max_value=100, step=1, value=int(cfg.get("PAGE_SIZE", 30))))
cfg["INTERVAL_MIN"] = int(st.sidebar.number_input("전송 주기(분)", min_value=5, max_value=360, step=5, value=int(cfg.get("INTERVAL_MIN", 60))))
cfg["RECENCY_HOURS"] = int(st.sidebar.number_input("신선도(최근 N시간)", min_value=6, max_value=168, step=6, value=int(cfg.get("RECENCY_HOURS", 72))))

# 필터 토글
cfg["ALLOWLIST_STRICT"] = bool(st.sidebar.checkbox("🧱 ALLOWLIST_STRICT (허용 도메인 외 차단)", value=bool(cfg.get("ALLOWLIST_STRICT", True))))

st.sidebar.divider()
if st.sidebar.button("구성 리로드", use_container_width=True):
    st.experimental_rerun()

st.title("📰 국내 PE 동향 뉴스 자동 모니터링")
st.caption("Streamlit + Naver/NewsAPI + Telegram + APScheduler (Render + UptimeRobot)")

def make_env() -> dict:
    return {
        "NAVER_CLIENT_ID": naver_id,
        "NAVER_CLIENT_SECRET": naver_secret,
        "NEWSAPI_KEY": newsapi_key,
        "TELEGRAM_BOT_TOKEN": bot_token,
        "TELEGRAM_CHAT_ID": chat_id,
    }

col1, col2, col3 = st.columns(3)
sched = get_scheduler()

with col1:
    if st.button("지금 한번 실행(미리보기)", type="primary"):
        res = transmit_once(cfg, make_env(), preview=True)
        st.session_state["preview"] = res

with col2:
    if st.button("지금 한번 전송"):
        res = transmit_once(cfg, make_env(), preview=False)
        st.session_state["preview"] = res

with col3:
    if st.button("스케줄 시작"):
        start_schedule(cfg_path=cfg_path, env=make_env(), minutes=int(cfg["INTERVAL_MIN"]))
        st.success("스케줄 시작됨 (즉시 전송 후 주기 실행)")

# 상태
_running = is_running(sched)
st.subheader("상태")
st.info(f"Scheduler 실행 중: {_running}")

# 미리보기 결과 — 전체 필터링 기사만 표시 (Top10 없음)
st.subheader("📋 필터링된 전체 기사")
res = st.session_state.get("preview", {"items": []})
items = res.get("items", [])
if not items:
    st.write("결과 없음")
else:
    for it in items:
        t = it.get("title", "")
        u = it.get("url", "")
        try:
            pub = dt.datetime.strptime(it["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).astimezone(APP_TZ)
            when = pub.strftime("%Y-%m-%d %H:%M")
        except Exception:
            when = "-"
        st.markdown(f"- <a href='{u}'>{t}</a> ({when})", unsafe_allow_html=True)
