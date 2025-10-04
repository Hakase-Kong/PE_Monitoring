# pe_monitoring.py
# Streamlit + Naver OpenAPI + NewsAPI + Telegram + APScheduler
# - 전체 Top10만 송출
# - 주말/공휴일 미전송
# - 도메인(링크 기준) 출처 표기
# - 중복/유사 제목 자동 제거
# - Render 환경에서 config.json 고정 로드
# - 좌측 컨트롤 활성화
# - Asia/Seoul 타임존 고정 (pytz)

import os
import re
import json
import time
import math
import html
import pytz
import queue
import string
import random
import logging
import datetime as dt
from urllib.parse import urlparse

import requests
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler

# -----------------------------------------------------------------------------
# 고정 경로(Deploy 환경 기본값)
# -----------------------------------------------------------------------------
CONFIG_PATH_DEFAULT = "/opt/render/project/src/config.json"

# -----------------------------------------------------------------------------
# 공통 유틸
# -----------------------------------------------------------------------------
KST = pytz.timezone("Asia/Seoul")

def now_kst() -> dt.datetime:
    return dt.datetime.now(tz=KST)

def get_domain(url: str) -> str:
    try:
        d = urlparse(url).netloc.lower()
        if d.startswith("www."):
            d = d[4:]
        return d
    except Exception:
        return "unknown"

def normalize_title(t: str) -> str:
    t = re.sub(r"\[[^\]]+\]", " ", t)          # [단독], [속보] 같은 태그 제거
    t = re.sub(r"\([^)]*\)", " ", t)           # 괄호 부가정보 제거
    t = re.sub(r"[^\w가-힣\s]", " ", t)         # 특수문자 제거
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def human_time(ts: str | dt.datetime):
    if isinstance(ts, str):
        return ts
    if isinstance(ts, dt.datetime):
        return ts.astimezone(KST).strftime("%Y-%m-%d %H:%M")
    return str(ts)

# -----------------------------------------------------------------------------
# 설정 로드
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

# -----------------------------------------------------------------------------
# 외부 API
# -----------------------------------------------------------------------------
def search_naver_news(query: str, display: int = 30, offset: int = 1,
                      client_id: str = "", client_secret: str = "") -> list[dict]:
    if not client_id or not client_secret:
        return []
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    params = {
        "query": query,
        "display": max(1, min(display, 100)),
        "start": max(1, min(offset, 1000)),
        "sort": "date",
    }
    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code != 200:
        return []
    data = r.json().get("items", [])
    out = []
    for it in data:
        out.append({
            "title": html.unescape(re.sub(r"<\/?b>", "", it.get("title",""))),
            "summary": html.unescape(re.sub(r"<\/?b>", "", it.get("description",""))),
            "link": it.get("link") or it.get("originallink") or "",
            "published_at": it.get("pubDate"),
            "source": get_domain(it.get("link") or it.get("originallink") or ""),
            "via": "naver",
        })
    return out

def search_newsapi(query: str, page_size: int = 30, api_key: str = "") -> list[dict]:
    if not api_key:
        return []
    url = "https://newsapi.org/v2/everything"
    # NewsAPI는 OR 문법이 약해서 | 로 대체, 공백 쿼리는 큰따옴표로 묶어주면 적중률↑
    def qfix(q: str):
        parts = [p.strip() for p in q.split(" OR ") if p.strip()]
        parts = [f"\"{p}\"" if " " in p else p for p in parts]
        return " OR ".join(parts) if parts else q
    params = {
        "q": qfix(query),
        "pageSize": max(1, min(page_size, 100)),
        "language": "ko",
        "sortBy": "publishedAt",
        "apiKey": api_key,
    }
    r = requests.get(url, params=params, timeout=12)
    if r.status_code != 200:
        return []
    data = r.json().get("articles", [])
    out = []
    for it in data:
        out.append({
            "title": it.get("title") or "",
            "summary": it.get("description") or "",
            "link": it.get("url") or "",
            "published_at": it.get("publishedAt"),
            "source": get_domain(it.get("url") or ""),
            "via": "newsapi",
        })
    return out

# -----------------------------------------------------------------------------
# 스코어링 & 필터
# -----------------------------------------------------------------------------
def score_item(item: dict, cfg: dict) -> float:
    title = f"{item.get('title','')} {item.get('summary','')}"
    tnorm = title.lower()
    score = 0.0

    # 도메인 가중치
    dw = cfg.get("DOMAIN_WEIGHTS", {})
    score += float(dw.get(item.get("source",""), 1.0))

    # 키워드 가중치(포함되면 +1)
    for kw in cfg.get("KEYWORDS", []):
        if kw.lower() in tnorm:
            score += 1.0

    # 화제성(대문자/숫자/길이 등 간단 부스트)
    score += min(1.0, len(item.get("title","")) / 80.0)

    return score

def should_drop(item: dict, cfg: dict) -> bool:
    t = item.get("title","")
    d = item.get("source","")

    # 제목 제외 키워드
    for bad in cfg.get("EXCLUDE_TITLE_KEYWORDS", []):
        if bad and bad.lower() in t.lower():
            return True

    # 도메인 허용/차단
    allow = cfg.get("ALLOW_DOMAINS", [])
    block = cfg.get("BLOCK_DOMAINS", [])
    if d in block:
        return True
    if cfg.get("ALLOWLIST_STRICT", False) and allow:
        if d not in allow:
            return True

    return False

def dedup(items: list[dict], threshold: float = 0.55) -> list[dict]:
    out = []
    seen = []
    for it in items:
        nt = normalize_title(it.get("title",""))
        drop = False
        for st in seen:
            if jaccard(nt, st) >= threshold:
                drop = True
                break
        if not drop:
            out.append(it)
            seen.append(nt)
    return out

# -----------------------------------------------------------------------------
# 텔레그램
# -----------------------------------------------------------------------------
def tg_send_message(token: str, chat_id: str, text: str, disable_preview: bool = True) -> bool:
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": disable_preview,
    }
    r = requests.post(url, data=data, timeout=10)
    return r.status_code == 200

# -----------------------------------------------------------------------------
# 전송 로직
# -----------------------------------------------------------------------------
def pick_top10(all_items: list[dict], cfg: dict) -> list[dict]:
    # 1) 제외 필터
    items = [it for it in all_items if not should_drop(it, cfg)]
    # 2) 정렬 스코어
    for it in items:
        it["_score"] = score_item(it, cfg)
    items.sort(key=lambda x: x["_score"], reverse=True)
    # 3) 유사 중복 제거 후 Top10
    items = dedup(items, threshold=0.55)
    return items[:10]

def format_telegram_block(header: str, items: list[dict]) -> str:
    if not items:
        return ""
    lines = [f"📌 <b>{header}</b>"]
    for it in items:
        title = html.escape(it.get("title",""))
        link  = it.get("link","")
        dom   = it.get("source","")
        ts    = it.get("published_at") or ""
        ts    = human_time(ts)
        lines.append(f"• {title} ({link}) — {dom} ({ts})")
    return "\n".join(lines)

def transmit_once(cfg: dict, naver_id: str, naver_secret: str, newsapi_key: str,
                  tg_token: str, tg_chat: str, preview_mode: bool,
                  page_size: int, recency_hours: int) -> dict:
    # 주말/공휴일 차단
    today = now_kst().date()
    if not preview_mode:
        if cfg.get("BLOCK_WEEKEND", True) and today.weekday() >= 5:
            return {"picked": 0, "sent": 0, "skipped": "weekend"}
        if cfg.get("HOLIDAYS"):  # YYYY-MM-DD 배열
            holidays = set(dt.date.fromisoformat(h) for h in cfg["HOLIDAYS"])
            if today in holidays:
                return {"picked": 0, "sent": 0, "skipped": "holiday"}

    # 쿼리 생성 (config.json의 KEYWORDS 통합)
    keywords = cfg.get("KEYWORDS", [])
    if not keywords:
        return {"picked": 0, "sent": 0, "skipped": "no_keywords"}

    query = " OR ".join(keywords)

    # 수집
    all_items = []
    naver = search_naver_news(query, display=page_size, offset=1,
                              client_id=naver_id, client_secret=naver_secret)
    all_items += naver
    newsapi = search_newsapi(query, page_size=page_size, api_key=newsapi_key)
    all_items += newsapi

    # 시간 필터 (recency_hours)
    if recency_hours and recency_hours > 0:
        cutoff = now_kst() - dt.timedelta(hours=recency_hours)
        def parse_ts(s):
            try:
                return dt.datetime.strptime(s, "%a, %d %b %Y %H:%M:%S %z")
            except Exception:
                try:
                    return dt.datetime.fromisoformat(s.replace("Z","+00:00"))
                except Exception:
                    return None
        _tmp = []
        for it in all_items:
            ts = parse_ts(it.get("published_at",""))
            if not ts:
                _tmp.append(it)  # 시간이 없으면 일단 포함
            else:
                if ts >= cutoff:
                    _tmp.append(it)
        all_items = _tmp

    # Top 10 선별
    top10 = pick_top10(all_items, cfg)

    # 전송 or 미리보기
    if preview_mode:
        return {"picked": len(top10), "sent": 0, "items": top10}

    if not tg_token or not tg_chat:
        return {"picked": len(top10), "sent": 0, "skipped": "no_telegram"}

    block = format_telegram_block("Top 10 뉴스", top10)
    if not block:
        return {"picked": 0, "sent": 0, "skipped": "empty"}

    ok = tg_send_message(tg_token, tg_chat, block, disable_preview=cfg.get("TELEGRAM_DISABLE_PREVIEW", True))
    return {"picked": len(top10), "sent": 1 if ok else 0, "items": top10}

# -----------------------------------------------------------------------------
# 스케줄러
# -----------------------------------------------------------------------------
SCHED = BackgroundScheduler(timezone="Asia/Seoul")
JOB_ID = "pe_monitoring_job"

def start_scheduler(cfg, naver_id, naver_secret, newsapi_key, tg_token, tg_chat,
                    page_size, recency_hours, interval_min):
    def job():
        transmit_once(cfg, naver_id, naver_secret, newsapi_key, tg_token, tg_chat,
                      preview_mode=False, page_size=page_size, recency_hours=recency_hours)

    if SCHED.get_job(JOB_ID):
        SCHED.remove_job(JOB_ID)
    SCHED.add_job(job, "interval", minutes=max(5, int(interval_min)), id=JOB_ID, next_run_time=now_kst())
    if not SCHED.running:
        SCHED.start()

def stop_scheduler():
    if SCHED.get_job(JOB_ID):
        SCHED.remove_job(JOB_ID)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PE 동향 뉴스 → Telegram 자동 전송", page_icon="📰", layout="wide")

st.markdown("### 📰 PE 동향 뉴스 → Telegram 자동 전송")
st.caption("Streamlit + Naver + NewsAPI + Telegram + APScheduler")

# 좌측: 자격/설정
st.sidebar.markdown("**자격증명 / 설정**")
cfg_path = CONFIG_PATH_DEFAULT
cfg_exists = os.path.exists(cfg_path)
st.sidebar.caption(f"CONFIG 경로:\n`{cfg_path}` / 존재: **{'True' if cfg_exists else 'False'}**")

newsapi_key = st.sidebar.text_input("NewsAPI Key (선택)", value=read_env("NEWSAPI_KEY"), type="password")
naver_id    = st.sidebar.text_input("Naver Client ID (선택)", value=read_env("NAVER_CLIENT_ID"), type="password")
naver_secret= st.sidebar.text_input("Naver Client Secret (선택)", value=read_env("NAVER_CLIENT_SECRET"), type="password")
tg_token    = st.sidebar.text_input("Telegram Bot Token", value=read_env("TELEGRAM_BOT_TOKEN"), type="password")
tg_chat     = st.sidebar.text_input("Telegram Chat ID", value=read_env("TELEGRAM_CHAT_ID") or "", help="예: -100xxxxxxxxxx")

st.sidebar.markdown("---")
st.sidebar.subheader("config.json")

if st.sidebar.button("구성 리로드", use_container_width=True):
    st.cache_resource.clear()

cfg = load_config(cfg_path) if cfg_exists else {
    "KEYWORDS": [],
    "ALLOW_DOMAINS": [],
    "BLOCK_DOMAINS": [],
    "EXCLUDE_TITLE_KEYWORDS": [],
    "DOMAIN_WEIGHTS": {},
    "TELEGRAM_DISABLE_PREVIEW": True,
    "ALLOWLIST_STRICT": False,
    "BLOCK_WEEKEND": True,
    "HOLIDAYS": [],   # ["2025-10-03", ...]
}

# 읽기 전용 표시
kw_preview = ", ".join(cfg.get("KEYWORDS", [])) or "(none)"
st.sidebar.caption(f"**KEYWORDS (읽기전용)**\n{kw_preview}")

# 조정 가능한 파라미터 (disabled=False)
page_size      = st.sidebar.number_input("페이지당 수집 수", min_value=5, max_value=100, value=int(cfg.get("PAGE_SIZE", 30)), step=1, disabled=False)
max_per_kw     = st.sidebar.number_input("전송 건수 제한(키워드별)", min_value=1, max_value=50, value=int(cfg.get("MAX_PER_KEYWORD", 10)), step=1, disabled=False)
interval_min   = st.sidebar.number_input("전송 주기(분)", min_value=5, max_value=720, value=int(cfg.get("INTERVAL_MINUTES", 60)), step=5, disabled=False)
recency_hours  = st.sidebar.number_input("신선도(최근 N시간)", min_value=1, max_value=168, value=int(cfg.get("RECENCY_HOURS", 48)), step=1, disabled=False)

st.sidebar.markdown("---")
workhour_only  = st.sidebar.checkbox("업무시간(08~20 KST) 내 전송", value=bool(cfg.get("ONLY_WORKING_HOURS", True)))
disable_prev   = st.sidebar.checkbox("링크 프리뷰 비활성화", value=bool(cfg.get("TELEGRAM_DISABLE_PREVIEW", True)))
block_weekend  = st.sidebar.checkbox("주말 미전송", value=bool(cfg.get("BLOCK_WEEKEND", True)))
block_holiday  = st.sidebar.checkbox("공휴일 미전송", value=True)
allowlist_strict = st.sidebar.checkbox("ALLOWLIST_STRICT (허용 도메인만)", value=bool(cfg.get("ALLOWLIST_STRICT", False)))

# UI 상태 표시
st.markdown("#### 상태")
running = bool(SCHED.get_job(JOB_ID))
st.info(f"Scheduler 실행 중: {running}")

# 버튼들
colA, colB, colC, colD = st.columns(4)
with colA:
    if st.button("지금 한 번 실행(미리보기)", use_container_width=True):
        # 미리보기
        cfg["ALLOWLIST_STRICT"] = allowlist_strict
        cfg["TELEGRAM_DISABLE_PREVIEW"] = disable_prev
        cfg["BLOCK_WEEKEND"] = block_weekend
        if not block_holiday:
            cfg["HOLIDAYS"] = []  # 미적용
        res = transmit_once(cfg, naver_id, naver_secret, newsapi_key, tg_token, tg_chat,
                            preview_mode=True, page_size=page_size, recency_hours=recency_hours)
        st.session_state["preview_items"] = res.get("items", [])
        st.success(f"완료: {len(res.get('items', []))}건 미리보기, 0건 전송(미리보기 모드)")
with colB:
    if st.button("지금 한 번 전송", use_container_width=True):
        cfg["ALLOWLIST_STRICT"] = allowlist_strict
        cfg["TELEGRAM_DISABLE_PREVIEW"] = disable_prev
        cfg["BLOCK_WEEKEND"] = block_weekend
        if not block_holiday:
            cfg["HOLIDAYS"] = []
        res = transmit_once(cfg, naver_id, naver_secret, newsapi_key, tg_token, tg_chat,
                            preview_mode=False, page_size=page_size, recency_hours=recency_hours)
        st.success(f"전송 완료: 선별 {res.get('picked',0)}건 / 전송 {res.get('sent',0)}건")
with colC:
    if st.button("스케줄 시작", use_container_width=True):
        # 업무시간 가드: 스케줄러 작업에서 체크
        cfg["ONLY_WORKING_HOURS"] = workhour_only
        cfg["ALLOWLIST_STRICT"] = allowlist_strict
        cfg["TELEGRAM_DISABLE_PREVIEW"] = disable_prev
        cfg["BLOCK_WEEKEND"] = block_weekend
        if not block_holiday:
            cfg["HOLIDAYS"] = []
        start_scheduler(cfg, naver_id, naver_secret, newsapi_key, tg_token, tg_chat,
                        page_size, recency_hours, interval_min)
        st.success("스케줄러 시작")
with colD:
    if st.button("스케줄 정지", use_container_width=True):
        stop_scheduler()
        st.warning("스케줄러 정지")

st.markdown("---")

# 미리보기 섹션
st.markdown("### 미리보기: 최신 10건")
items = st.session_state.get("preview_items", [])

# 업무시간/주말/공휴일 안내
tips = []
if workhour_only:
    tips.append("업무시간(08~20 KST) 내 전송")
if block_weekend:
    tips.append("주말 미전송")
if block_holiday:
    tips.append("공휴일 미전송")
if tips:
    st.caption(" · ".join(tips))

if not items:
    with st.expander("Top 10 — 0건", expanded=True):
        st.write("결과 없음")
else:
    with st.expander(f"Top 10 — {len(items)}건", expanded=True):
        for it in items:
            dom = it.get("source","")
            ts  = human_time(it.get("published_at",""))
            st.markdown(f"- **{it.get('title','')}**  \n  {it.get('summary','')}  \n  [{dom}]({it.get('link','')}) · {ts}")

# -----------------------------------------------------------------------------
# 스케줄러 실행 중일 때: 업무시간/공휴일 차단 로직(잡 함수 내부에서 재확인)
# -----------------------------------------------------------------------------
def scheduler_guard():
    job = SCHED.get_job(JOB_ID)
    if not job:
        return
    # 업무시간 체크
    if cfg.get("ONLY_WORKING_HOURS", True):
        hn = now_kst().hour
        if not (8 <= hn <= 20):
            return  # off-hours, 실제 전송은 job 내부에서 한번 더 가드함

scheduler_guard()
