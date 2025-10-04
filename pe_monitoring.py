import os
import re
import json
import time
import math
import hashlib
import logging
import requests
import datetime as dt
from urllib.parse import urlparse, parse_qs

import pytz
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler

# -------------------------
# 기본 설정
# -------------------------
APP_TZ = pytz.timezone("Asia/Seoul")
DEFAULT_CONFIG_PATH = "config.json"  # 동일 디렉토리 기준

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pe_monitor_fixed")

# -------------------------
# 유틸
# -------------------------
def load_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("config 로드 실패(%s): %s", path, e)
        return {}

def now_kst():
    return dt.datetime.now(APP_TZ)

def between_working_hours(kst: dt.datetime, start=8, end=20) -> bool:
    return start <= kst.hour < end

def is_weekend(kst: dt.datetime) -> bool:
    return kst.weekday() >= 5

def is_holiday(kst: dt.datetime, holidays: list[str]) -> bool:
    ymd = kst.strftime("%Y-%m-%d")
    return ymd in set(holidays or [])

def clamp(n, lo, hi):
    return max(lo, min(hi, n))

def domain_of(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").replace("www.", "")
    except Exception:
        return ""

def _token_hit(text: str, tokens: list[str]) -> bool:
    t = (text or "").lower()
    return any((tok or "").lower() in t for tok in (tokens or []))

def _alias_flatten(alias_map: dict) -> list[str]:
    vals = []
    for v in (alias_map or {}).values():
        vals.extend(v or [])
    return sorted({s.strip() for s in vals if s and s.strip()})

def _naver_sid(url: str) -> str | None:
    try:
        q = parse_qs(urlparse(url).query).get("sid", [])
        return q[0] if q else None
    except Exception:
        return None

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

# -------------------------
# 외부 API
# -------------------------
def search_newsapi(query: str, page_size: int, api_key: str, from_hours: int = 72, cfg: dict | None = None):
    if not api_key or not query:
        return []
    base = "https://newsapi.org/v2/everything"
    from_dt = (now_kst() - dt.timedelta(hours=from_hours)).astimezone(pytz.utc)
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
                "summary": a.get("description") or "",
                "provider": "newsapi",
                "origin_keyword": "_newsapi",
            })
        return res
    except Exception as e:
        log.warning("NewsAPI 오류: %s", e)
        return []

def search_naver_news(keyword: str, display: int, offset: int, client_id: str, client_secret: str, recency_hours=72):
    if not client_id or not client_secret or not keyword:
        return []
    base = "https://openapi.naver.com/v1/search/news.json"
    params = {
        "query": keyword,
        "display": clamp(display, 1, 30),
        "start": clamp(offset, 1, 1000),
        "sort": "date",
    }
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
            pubdate = it.get("pubDate")
            try:
                pub_kst = dt.datetime.strptime(pubdate, "%a, %d %b %Y %H:%M:%S %z")
            except Exception:
                pub_kst = now_kst()
            if pub_kst < cutoff:
                continue
            title = re.sub("<.*?>", "", it.get("title") or "")
            desc = re.sub("<.*?>", "", it.get("description") or "")
            res.append({
                "title": title,
                "url": link,
                "source": domain_of(link),
                "publishedAt": pub_kst.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "summary": desc,
                "provider": "naver",
                "origin_keyword": keyword,
            })
        return res
    except Exception as e:
        log.warning("Naver 오류(%s): %s", keyword, e)
        return []

# -------------------------
# 필터/스코어/중복
# -------------------------
def should_drop(item: dict, cfg: dict) -> bool:
    url = item.get("url", "")
    title = (item.get("title") or "").strip()
    if not url or not title:
        return True

    src = domain_of(url)
    allow = set(cfg.get("ALLOW_DOMAINS", []) or [])
    block = set(cfg.get("BLOCK_DOMAINS", []) or [])
    allow_strict = bool(cfg.get("ALLOWLIST_STRICT"))

    if src in block:
        return True
    if allow_strict and allow and (src not in allow):
        return True

    # 네이버 섹션 필터(경제면=101 등)
    if "naver.com" in src:
        sids = set(cfg.get("NAVER_ALLOW_SIDS", []) or [])
        if sids:
            sid = _naver_sid(url)
            if sid not in sids:
                return True

    # 제외 키워드
    for w in (cfg.get("EXCLUDE_TITLE_KEYWORDS", []) or []):
        if w and w.lower() in title.lower():
            return True

    # 포함 키워드 + 별칭(필수 적중)
    include = (cfg.get("INCLUDE_TITLE_KEYWORDS", []) or [])
    aliases = _alias_flatten(cfg.get("KEYWORD_ALIASES") or {})
    must = include + aliases
    if must and not _token_hit(title, must):
        return True

    return False

def score_item(item: dict, cfg: dict) -> float:
    src = domain_of(item.get("url", ""))
    title = item.get("title") or ""
    score = 0.0

    # 도메인 가중치
    weights = cfg.get("DOMAIN_WEIGHTS", {}) or {}
    score += float(weights.get(src, 1.0))

    # 키워드 적중 보너스
    hit_bonus = 0.0
    for kw in cfg.get("KEYWORDS", []):
        if kw and kw.lower() in title.lower():
            hit_bonus += 1.0
    for alias in _alias_flatten(cfg.get("KEYWORD_ALIASES") or {}):
        if alias and alias.lower() in title.lower():
            hit_bonus += 0.5
    score += hit_bonus

    # 신선도 보정
    try:
        ts = item.get("publishedAt")
        pub = dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc)
    except Exception:
        pub = now_kst()
    hours_ago = (now_kst() - pub.astimezone(APP_TZ)).total_seconds() / 3600.0
    score += max(0.0, 6.0 - (hours_ago / 8.0))

    return score

def dedup(items: list[dict], threshold: float = 0.82) -> list[dict]:
    def norm(s: str) -> str:
        s = re.sub(r"[\\[\\]\\(\\)【】『』“”\"'<>]", " ", s or "")
        s = re.sub(r"\\s+", " ", s).strip().lower()
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

def pick_top10(items: list[dict], cfg: dict) -> list[dict]:
    filtered = [it for it in items if not should_drop(it, cfg)]
    for it in filtered:
        it["_score"] = score_item(it, cfg)
    filtered.sort(key=lambda x: x["_score"], reverse=True)
    unique = dedup(filtered)
    cap = int(cfg.get("MAX_PER_KEYWORD", 8))
    bucket, out = {}, []
    for it in unique:
        k = it.get("origin_keyword") or "_"
        c = bucket.get(k, 0)
        if c < cap:
            out.append(it)
            bucket[k] = c + 1
        if len(out) >= 10:
            break
    return out

# -------------------------
# Telegram
# -------------------------
def format_telegram_block(title: str, items: list[dict]) -> str:
    if not items:
        return ""
    lines = [f"📌 <b>{title}</b>"]
    for it in items:
        src = domain_of(it.get("url", ""))
        try:
            pub = dt.datetime.strptime(it["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc).astimezone(APP_TZ)
            when = pub.strftime("%Y-%m-%d %H:%M")
        except Exception:
            when = ""
        t = it.get("title", "").strip()
        u = it.get("url", "")
        lines.append(f"• {t} ({u}) — {src} ({when})")
    return "\\n".join(lines)

def send_telegram(bot_token: str, chat_id: str, text: str, disable_web_page_preview: bool = True):
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": disable_web_page_preview,
    }
    try:
        r = requests.post(url, json=payload, timeout=12)
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning("텔레그램 전송 실패: %s", e)
        return False

# -------------------------
# 수집/전송
# -------------------------
def transmit_once(cfg: dict, preview_mode: bool, env: dict):
    keywords = cfg.get("KEYWORDS", []) or []
    page_size = int(cfg.get("PAGE_SIZE", 30))
    recency_hours = int(cfg.get("RECENCY_HOURS", 72))

    newsapi_key = env.get("NEWSAPI_KEY", os.getenv("NEWSAPI_KEY", ""))
    naver_id = env.get("NAVER_CLIENT_ID", os.getenv("NAVER_CLIENT_ID", ""))
    naver_secret = env.get("NAVER_CLIENT_SECRET", os.getenv("NAVER_CLIENT_SECRET", ""))
    bot_token = env.get("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_BOT_TOKEN", ""))
    chat_id = env.get("TELEGRAM_CHAT_ID", os.getenv("TELEGRAM_CHAT_ID", ""))

    all_items = []

    # NAVER — 키워드별 개별 조회
    naver_hits = 0
    for kw in keywords:
        batch = search_naver_news(
            kw, display=max(3, page_size // 3), offset=1,
            client_id=naver_id, client_secret=naver_secret, recency_hours=recency_hours
        )
        naver_hits += len(batch)
        all_items += batch
        time.sleep(0.2)

    # NEWSAPI — OR 쿼리(단, 내부에서 title 제한/도메인 제한)
    newsapi_hits = 0
    if newsapi_key and keywords:
        query = " OR ".join(keywords)
        batch = search_newsapi(query, page_size=page_size, api_key=newsapi_key, from_hours=recency_hours, cfg=cfg)
        newsapi_hits = len(batch)
        all_items += batch

    raw_count = len(all_items)

    # 필터/스코어/Top10
    _stage = [it for it in all_items if not should_drop(it, cfg)]
    after_filter = len(_stage)
    for it in _stage:
        it["_score"] = score_item(it, cfg)
    _stage.sort(key=lambda x: x["_score"], reverse=True)
    top10 = pick_top10(all_items, cfg)

    if preview_mode:
        return {
            "picked": len(top10),
            "sent": 0,
            "items": top10,
            "diag": {
                "naver_hits": naver_hits,
                "newsapi_hits": newsapi_hits,
                "raw": raw_count,
                "after_filter": after_filter
            }
        }

    # 전송 정책
    kst_now = now_kst()
    if cfg.get("ONLY_WORKING_HOURS") and not between_working_hours(kst_now, 8, 20):
        log.info("업무시간 외 — 전송 생략")
        return {"picked": len(top10), "sent": 0, "items": top10}
    if cfg.get("BLOCK_WEEKEND") and is_weekend(kst_now):
        log.info("주말 — 전송 생략")
        return {"picked": len(top10), "sent": 0, "items": top10}
    if cfg.get("BLOCK_HOLIDAY") and is_holiday(kst_now, cfg.get("HOLIDAYS", [])):
        log.info("공휴일 — 전송 생략")
        return {"picked": len(top10), "sent": 0, "items": top10}

    if not top10:
        log.info("전송할 기사 없음")
        return {"picked": 0, "sent": 0, "items": []}

    text = format_telegram_block("Top 10", top10)
    ok = send_telegram(
        bot_token, chat_id, text,
        disable_web_page_preview=bool(cfg.get("TELEGRAM_DISABLE_PREVIEW", True))
    )
    return {"picked": len(top10), "sent": (1 if ok else 0), "items": top10}

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="PE 동향 뉴스 → Telegram 자동 전송 (fixed)", page_icon="📰", layout="wide")

# config 파일 경로 입력 가능(기본: 확장본)
cfg_path = st.sidebar.text_input("config.json 경로", value=DEFAULT_CONFIG_PATH)
cfg = load_config(cfg_path)
st.sidebar.caption(f"CONFIG 경로: {cfg_path} / 존재: {'True' if cfg else 'False'}")

# 자격증명(선택)
newsapi_key = st.sidebar.text_input("NewsAPI Key (선택)", type="password", value=os.getenv("NEWSAPI_KEY", ""))
naver_id = st.sidebar.text_input("Naver Client ID (선택)", type="password", value=os.getenv("NAVER_CLIENT_ID", ""))
naver_secret = st.sidebar.text_input("Naver Client Secret (선택)", type="password", value=os.getenv("NAVER_CLIENT_SECRET", ""))
bot_token = st.sidebar.text_input("Telegram Bot Token", type="password", value=os.getenv("TELEGRAM_BOT_TOKEN", ""))
chat_id = st.sidebar.text_input("Telegram Chat ID (채널/그룹)", value=os.getenv("TELEGRAM_CHAT_ID", ""))

st.sidebar.divider()
st.sidebar.subheader("전송/수집 파라미터")
cfg["PAGE_SIZE"] = st.sidebar.number_input("페이지당 수집 수", min_value=10, max_value=100, step=1, value=int(cfg.get("PAGE_SIZE", 30)))
cfg["MAX_PER_KEYWORD"] = st.sidebar.number_input("전송 건수 제한(키워드별)", min_value=3, max_value=20, step=1, value=int(cfg.get("MAX_PER_KEYWORD", 8)))
cfg["INTERVAL_MIN"] = st.sidebar.number_input("전송 주기(분)", min_value=15, max_value=360, step=5, value=int(cfg.get("INTERVAL_MIN", 60)))
cfg["RECENCY_HOURS"] = st.sidebar.number_input("신선도(최근 N시간)", min_value=6, max_value=168, step=6, value=int(cfg.get("RECENCY_HOURS", 72)))

# ✅ 체크박스: 즉시 반영 (저장은 별도)
cfg["ONLY_WORKING_HOURS"] = st.sidebar.checkbox("✅ 업무시간(08~20 KST) 내 전송", value=bool(cfg.get("ONLY_WORKING_HOURS", True)))
cfg["TELEGRAM_DISABLE_PREVIEW"] = st.sidebar.checkbox("🔗 링크 프리뷰 비활성화", value=bool(cfg.get("TELEGRAM_DISABLE_PREVIEW", True)))
cfg["BLOCK_WEEKEND"] = st.sidebar.checkbox("🚫 주말 미전송", value=bool(cfg.get("BLOCK_WEEKEND", True)))
cfg["BLOCK_HOLIDAY"] = st.sidebar.checkbox("🚫 공휴일 미전송", value=bool(cfg.get("BLOCK_HOLIDAY", False)))
cfg["ALLOWLIST_STRICT"] = st.sidebar.checkbox("🧱 ALLOWLIST_STRICT (허용 도메인 외 차단)", value=bool(cfg.get("ALLOWLIST_STRICT", True)))

st.sidebar.divider()
if st.sidebar.button("구성 리로드", use_container_width=True):
    cfg = load_config(cfg_path)
    st.rerun()

# 키워드 확인
st.sidebar.caption("KEYWORDS (일부)")
st.sidebar.code(", ".join(cfg.get("KEYWORDS", [])[:20]) or "(none)", language="text")

st.title("📰 PE 동향 뉴스 ➜ Telegram 자동 전송 (fixed)")
st.caption("Streamlit + NewsAPI/Naver + Telegram + APScheduler")

col1, col2, col3, col4 = st.columns([1,1,1,1])

def transmit_env():
    return {
        "NEWSAPI_KEY": newsapi_key,
        "NAVER_CLIENT_ID": naver_id,
        "NAVER_CLIENT_SECRET": naver_secret,
        "TELEGRAM_BOT_TOKEN": bot_token,
        "TELEGRAM_CHAT_ID": chat_id,
    }

with col1:
    if st.button("지금 한번 실행(미리보기)", type="primary"):
        res = transmit_once(cfg, preview_mode=True, env=transmit_env())
        st.session_state["preview"] = res

with col2:
    if st.button("지금 한번 전송"):
        res = transmit_once(cfg, preview_mode=False, env=transmit_env())
        st.session_state["preview"] = res

SCHED = BackgroundScheduler(timezone=APP_TZ)
if "sched_started" not in st.session_state:
    st.session_state["sched_started"] = False

def scheduled_job():
    try:
        transmit_once(cfg, preview_mode=False, env={
            "NEWSAPI_KEY": os.getenv("NEWSAPI_KEY", ""),
            "NAVER_CLIENT_ID": os.getenv("NAVER_CLIENT_ID", ""),
            "NAVER_CLIENT_SECRET": os.getenv("NAVER_CLIENT_SECRET", ""),
            "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", ""),
            "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", ""),
        })
    except Exception as e:
        log.exception("스케줄 작업 실패: %s", e)

with col3:
    if not st.session_state["sched_started"]:
        if st.button("스케줄 시작"):
            minutes = int(cfg.get("INTERVAL_MIN", 60))
            SCHED.add_job(scheduled_job, "interval", minutes=minutes, id="job1", replace_existing=True, next_run_time=now_kst()+dt.timedelta(seconds=3))
            SCHED.start()
            st.session_state["sched_started"] = True
    else:
        st.button("스케줄 시작", disabled=True)

with col4:
    if st.button("스케줄 중지"):
        try:
            SCHED.shutdown(wait=False)
        except Exception:
            pass
        st.session_state["sched_started"] = False

st.subheader("상태")
st.info(f"Scheduler 실행 중: {st.session_state['sched_started']}")

# 미리보기
st.subheader("미리보기: 최신 10건")
res = st.session_state.get("preview", {})
items = res.get("items", [])
diag = res.get("diag", {})

with st.expander(f"Top 10 — {len(items)}건", expanded=True):
    if not items:
        st.write("결과 없음")
    else:
        for it in items:
            src = domain_of(it.get("url", ""))
            title = it.get("title", "")
            url = it.get("url", "")
            try:
                pub = dt.datetime.strptime(it["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc).astimezone(APP_TZ)
                when = pub.strftime("%Y-%m-%d %H:%M")
            except Exception:
                when = "-"
            st.markdown(f"- **{title}**  \n  {url}  — *{src}* ({when})")

if diag:
    st.caption(f"수집: Naver {diag.get('naver_hits',0)} / NewsAPI {diag.get('newsapi_hits',0)} • 원시합계: {diag.get('raw',0)} → 필터후: {diag.get('after_filter',0)} → Top10: {len(items)}")
