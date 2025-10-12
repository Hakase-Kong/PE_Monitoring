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
from threading import Lock

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
CURRENT_CFG_DICT: Dict = {}     # UI에서 시작할 때 스냅샷 저장 (재시작 전까지 유효)
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

def is_weekend(kst: dt.datetime) -> bool:
    return kst.weekday() >= 5

def is_holiday(kst: dt.datetime, holidays: List[str]) -> bool:
    ymd = kst.strftime("%Y-%m-%d")
    return ymd in set(holidays or [])

def between_working_hours(kst: dt.datetime, start=8, end=20) -> bool:
    return start <= kst.hour < end

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

def search_newsapi(query: str, page_size: int, api_key: str, from_hours: int = 72, cfg: dict = None) -> List[dict]:
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
# 중복 제거 강화 (URL/제목 정규화 + 근사 중복)
# -------------------------
NAVER_ART_RE = re.compile(r"/article/(\d{3})/(\d{10})")
NOISE_TAGS = {"단독","속보","시그널","fn마켓워치","투자360","영상","포토","르포","사설","칼럼","분석"}
BRACKET_RE   = re.compile(r"[\[\(（](.*?)[\]\)）]")
MULTISPACE_RE = re.compile(r"\s+")
SYNONYM_MAP = {
    "imm인베스트먼트": "imm인베",
    "imm 인베스트먼트": "imm인베",
    "imm investment": "imm인베",
    "mergers & acquisitions": "m&a",
    "인베스트먼트": "인베",
}

def canonical_url_id(url: str) -> str:
    """같은 기사를 동일 키로 묶기 위한 정규화 ID 생성."""
    try:
        u = urlparse(url)
        host = (u.netloc or "").replace("www.", "")
        path = u.path or ""
        if host.endswith("naver.com"):
            m = NAVER_ART_RE.search(path)
            if m:
                oid, aid = m.group(1), m.group(2)
                return f"naver:{oid}:{aid}"
        base = f"{host}{path}".rstrip("/")
        return re.sub(r"/+$", "", base)
    except Exception:
        return url

def normalize_title(t: str) -> str:
    if not t:
        return ""
    s = t
    # 괄호/대괄호 안의 태그성 토큰 제거
    def _strip_noise(m):
        inner = (m.group(1) or "").strip()
        return "" if any(tag in inner.replace(" ", "") for tag in NOISE_TAGS) else inner
    s = BRACKET_RE.sub(_strip_noise, s)
    # 머리말 태그 제거
    for tag in NOISE_TAGS:
        s = re.sub(rf"^\s*(?:\[{tag}\]|\({tag}\))\s*", "", s, flags=re.IGNORECASE)
    # 특수문자/말줄임표 정리
    s = s.replace("…", " ").replace("ㆍ", " ").replace("·", " ").replace("—", " ")
    # 동의어 통일 (소문자)
    s_low = s.lower()
    for k, v in SYNONYM_MAP.items():
        s_low = s_low.replace(k, v)
    # 숫자 콤마 정규화
    s_low = re.sub(r"\b(\d{1,3}(,\d{3})+|\d+)\b", lambda m: m.group(0).replace(",", ""), s_low)
    # 공백 정리
    s_low = MULTISPACE_RE.sub(" ", s_low).strip()
    return s_low

def _tokens(s: str) -> set:
    return {w for w in re.split(r"[^0-9a-zA-Z가-힣]+", s) if len(w) >= 2}

def _bigrams(s: str) -> set:
    return {s[i:i+2] for i in range(len(s)-1)} if len(s) > 1 else set()

def is_near_dup(a: str, b: str) -> bool:
    """정규화 제목 a,b 근접 중복 판단."""
    if not a or not b:
        return False
    if a == b:
        return True
    ta, tb = _tokens(a), _tokens(b)
    if ta and tb:
        j_tok = len(ta & tb) / max(1, len(ta | tb))
        if j_tok >= 0.70:
            return True
    ba, bb = _bigrams(a), _bigrams(b)
    if ba and bb:
        j_bg = len(ba & bb) / max(1, len(ba | bb))
        if j_bg >= 0.55:
            return True
    if a in b or b in a:
        return True
    return False

def dedup(items: List[dict]) -> List[dict]:
    """URL 정규화 → 제목 근사 중복 제거 2단계."""
    out, seen_by_id, seen_titles = [], set(), []
    for it in items:
        url = it.get("url", "")
        cid = canonical_url_id(url)
        if cid in seen_by_id:
            continue
        norm_t = normalize_title(it.get("title", ""))
        dup = False
        for prev_norm, _idx in seen_titles:
            if is_near_dup(norm_t, prev_norm):
                dup = True
                break
        if dup:
            continue
        seen_by_id.add(cid)
        seen_titles.append((norm_t, len(out)))
        out.append(it)
    return out

# -------------------------
# 필터/정렬
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

    # 제목 포함/제외 키워드
    include = (cfg.get("INCLUDE_TITLE_KEYWORDS", []) or [])
    if include and not any(w.lower() in title.lower() for w in include):
        return True
    for w in (cfg.get("EXCLUDE_TITLE_KEYWORDS", []) or []):
        if w and w.lower() in title.lower():
            return True

    return False

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
        # 제목=링크, 출처 도메인 미표시
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

def _should_skip_by_time(cfg: dict) -> bool:
    """업무시간/주말/공휴일 옵션에 따라 전송을 건너뛸지 판단"""
    kst_now = now_kst()
    if cfg.get("ONLY_WORKING_HOURS") and not between_working_hours(kst_now, 8, 20):
        return True
    if cfg.get("BLOCK_WEEKEND") and is_weekend(kst_now):
        return True
    if cfg.get("BLOCK_HOLIDAY") and is_holiday(kst_now, cfg.get("HOLIDAYS", [])):
        return True
    return False

# -------------------------
# 실행 겹침 방지 락
# -------------------------
@st.cache_resource(show_spinner=False)
def get_run_lock() -> Lock:
    return Lock()

def transmit_once(cfg: dict, env: dict, preview=False) -> dict:
    # 실행 겹침 방지 (동시에 두 번 이상 돌지 않도록)
    run_lock = get_run_lock()
    if not run_lock.acquire(blocking=False):
        log.info("다른 실행이 진행 중이어서 이번 주기는 스킵합니다.")
        return {"count": 0, "items": []}
    try:
        # 전체 수집 → 필터/정렬 → 전체 리스트
        all_items = collect_all(cfg, env)
        ranked = rank_filtered(all_items, cfg)

        if preview:
            return {"count": len(ranked), "items": ranked}

        # 전송 타임 필터
        if _should_skip_by_time(cfg):
            log.info("시간 정책에 의해 전송 건너뜀 (업무시간/주말/공휴일)")
            return {"count": 0, "items": []}

        # 신규만 전송 (캐시 기준)
        cache = load_sent_cache()
        new_items = [it for it in ranked if sha1(it.get("url", "")) not in cache]

        # 신규 없으면 알림
        if not new_items:
            send_telegram(env.get("TELEGRAM_BOT_TOKEN", ""), env.get("TELEGRAM_CHAT_ID", ""), "📭 신규 뉴스 없음")
            return {"count": 0, "items": []}

        # 텔레그램 4096자 제한 대비 — 30개 단위로 배치 전송
        BATCH = 30
        sent_any = False
        for i in range(0, len(new_items), BATCH):
            chunk = new_items[i:i+BATCH]
            text = format_telegram_text(chunk)
            ok = send_telegram(env.get("TELEGRAM_BOT_TOKEN", ""), env.get("TELEGRAM_CHAT_ID", ""), text)
            sent_any = sent_any or ok
            time.sleep(0.6)

        if sent_any:
            cache |= {sha1(it.get("url", "")) for it in new_items}
            save_sent_cache(cache)

        return {"count": len(new_items), "items": new_items}
    finally:
        run_lock.release()

# -------------------------
# 스케줄러 (rerun-safe)
# -------------------------
@st.cache_resource(show_spinner=False)
def get_scheduler() -> BackgroundScheduler:
    sched = BackgroundScheduler(timezone=APP_TZ)
    sched.start()
    return sched

def scheduled_job():
    # UI에서 스냅샷이 있으면 그것을 우선 사용
    cfg = CURRENT_CFG_DICT or load_config(CURRENT_CFG_PATH)
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
    # next_run_time=now → 스케줄러가 즉시 1회 실행을 트리거 (수동 호출 금지)
    sched.add_job(scheduled_job, "interval", minutes=minutes, id=job_id,
                  replace_existing=True, next_run_time=now_kst())

def is_running(sched: BackgroundScheduler) -> bool:
    try:
        return any(j.id == "pe_news_job" for j in sched.get_jobs())
    except Exception:
        return False

# 스케줄 시작/중지(버튼 핸들러용) — 여기서만 global 사용
def start_schedule(cfg_path: str, cfg_dict: dict, env: dict, minutes: int):
    global CURRENT_CFG_PATH, CURRENT_CFG_DICT, CURRENT_ENV
    CURRENT_CFG_PATH = cfg_path
    CURRENT_CFG_DICT = dict(cfg_dict)  # UI 조정 옵션까지 스냅샷 저장
    CURRENT_ENV = env
    sched = get_scheduler()
    ensure_interval_job(sched, minutes)
    # 주의: 즉시 실행은 스케줄러 next_run_time으로만 유도(수동 scheduled_job() 호출 금지)

def stop_schedule():
    sched = get_scheduler()
    try:
        sched.remove_job("pe_news_job")
    except Exception:
        pass

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="PE 동향 뉴스 모니터링", page_icon="📰", layout="wide")

# Config / 자격증명
cfg_path = st.sidebar.text_input("config.json 경로", value=DEFAULT_CONFIG_PATH)
cfg_file = load_config(cfg_path)
st.sidebar.caption(f"Config 로드 상태: {'✅' if cfg_file else '❌'}  · 경로: {cfg_path}")

# 파일의 기본값을 UI 런타임 cfg로 복사
cfg = dict(cfg_file)

naver_id = st.sidebar.text_input("Naver Client ID", type="password", value=os.getenv("NAVER_CLIENT_ID", ""))
naver_secret = st.sidebar.text_input("Naver Client Secret", type="password", value=os.getenv("NAVER_CLIENT_SECRET", ""))
newsapi_key = st.sidebar.text_input("NewsAPI Key (선택)", type="password", value=os.getenv("NEWSAPI_KEY", ""))
bot_token = st.sidebar.text_input("Telegram Bot Token", type="password", value=os.getenv("TELEGRAM_BOT_TOKEN", ""))
chat_id = st.sidebar.text_input("Telegram Chat ID (채널/그룹)", value=os.getenv("TELEGRAM_CHAT_ID", ""))

# 파라미터
st.sidebar.divider()
st.sidebar.subheader("전송/수집 파라미터")
cfg["PAGE_SIZE"] = int(st.sidebar.number_input("페이지당 수집 수", min_value=10, max_value=100, step=1, value=int(cfg.get("PAGE_SIZE", 30))))
cfg["INTERVAL_MIN"] = int(st.sidebar.number_input("전송 주기(분)", min_value=5, max_value=360, step=5, value=int(cfg.get("INTERVAL_MIN", cfg.get("TRANSMIT_INTERVAL_MIN", 60)))))
cfg["RECENCY_HOURS"] = int(st.sidebar.number_input("신선도(최근 N시간)", min_value=6, max_value=168, step=6, value=int(cfg.get("RECENCY_HOURS", 72))))

# ✅ 시간 정책 토글
st.sidebar.subheader("시간 정책")
cfg["ONLY_WORKING_HOURS"] = bool(st.sidebar.checkbox("✅ 업무시간(08~20 KST) 내 전송", value=bool(cfg.get("ONLY_WORKING_HOURS", True))))
cfg["BLOCK_WEEKEND"]     = bool(st.sidebar.checkbox("🚫 주말 미전송", value=bool(cfg.get("BLOCK_WEEKEND", True))))
cfg["BLOCK_HOLIDAY"]     = bool(st.sidebar.checkbox("🚫 공휴일 미전송", value=bool(cfg.get("BLOCK_HOLIDAY", False))))
holidays_text = st.sidebar.text_area("공휴일(YYYY-MM-DD, 쉼표 또는 줄바꿈 구분)", value=", ".join(cfg.get("HOLIDAYS", [])))
cfg["HOLIDAYS"] = [s.strip() for s in re.split(r"[,\n]", holidays_text) if s.strip()]

# 기타 필터 토글
st.sidebar.subheader("기타 필터")
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

col1, col2, col3, col4 = st.columns(4)
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
        start_schedule(cfg_path=cfg_path, cfg_dict=cfg, env=make_env(), minutes=int(cfg["INTERVAL_MIN"]))
        st.success("스케줄 시작됨 (즉시 1회 전송 후 주기 실행)")

with col4:
    if st.button("스케줄 중지"):
        stop_schedule()
        st.warning("스케줄 중지됨")

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
