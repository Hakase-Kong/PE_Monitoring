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
# 기본 설정
# -------------------------
APP_TZ = pytz.timezone("Asia/Seoul")
DEFAULT_CONFIG_PATH = os.getenv("PE_CFG", "config.json")
CACHE_FILE = os.getenv("PE_SENT_CACHE", "sent_cache.json")  # 전송 이력 저장

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pe_monitor")

# 전역 환경 변수
CURRENT_CFG_PATH = DEFAULT_CONFIG_PATH
CURRENT_ENV = {
    "NEWSAPI_KEY": os.getenv("NEWSAPI_KEY", ""),
    "NAVER_CLIENT_ID": os.getenv("NAVER_CLIENT_ID", ""),
    "NAVER_CLIENT_SECRET": os.getenv("NAVER_CLIENT_SECRET", ""),
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", ""),
}


# -------------------------
# 유틸 함수
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

def domain_of(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").replace("www.", "")
    except Exception:
        return ""

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
# 뉴스 수집
# -------------------------
def search_naver_news(keyword: str, client_id: str, client_secret: str, recency_hours=72):
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
            link = it.get("link") or ""
            pubdate = it.get("pubDate")
            try:
                pub_kst = dt.datetime.strptime(pubdate, "%a, %d %b %Y %H:%M:%S %z")
            except Exception:
                pub_kst = now_kst()
            if pub_kst < cutoff:
                continue
            title = re.sub("<.*?>", "", it.get("title") or "")
            res.append({
                "title": title,
                "url": link,
                "publishedAt": pub_kst.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
        return res
    except Exception as e:
        log.warning("Naver 오류(%s): %s", keyword, e)
        return []


def collect_all(cfg: dict, env: dict) -> List[dict]:
    keywords = cfg.get("KEYWORDS", [])
    all_items = []
    for kw in keywords:
        batch = search_naver_news(kw, env["NAVER_CLIENT_ID"], env["NAVER_CLIENT_SECRET"])
        all_items += batch
        time.sleep(0.2)
    return all_items


# -------------------------
# 텔레그램 전송
# -------------------------
def send_telegram(bot_token: str, chat_id: str, text: str):
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
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning(f"텔레그램 전송 실패: {e}")
        return False


def format_telegram_text(items: List[dict]) -> str:
    if not items:
        return "📭 신규 뉴스 없음"
    lines = ["📌 <b>국내 PE 동향 관련 뉴스</b>"]
    for it in items:
        t = it["title"]
        u = it["url"]
        try:
            pub = dt.datetime.strptime(it["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).astimezone(APP_TZ)
            when = pub.strftime("%Y-%m-%d %H:%M")
        except Exception:
            when = "-"
        lines.append(f"• <a href=\"{u}\">{t}</a> ({when})")
    return "\n".join(lines)


# -------------------------
# 전송 로직
# -------------------------
def transmit_once(cfg: dict, env: dict, preview=False):
    items = collect_all(cfg, env)
    cache = load_sent_cache()
    new_items = [it for it in items if sha1(it["url"]) not in cache]

    if preview:
        return {"count": len(items), "items": items}

    if not new_items:
        send_telegram(env["TELEGRAM_BOT_TOKEN"], env["TELEGRAM_CHAT_ID"], "📭 신규 뉴스 없음")
        return {"count": 0, "items": []}

    text = format_telegram_text(new_items)
    ok = send_telegram(env["TELEGRAM_BOT_TOKEN"], env["TELEGRAM_CHAT_ID"], text)
    if ok:
        cache |= {sha1(it["url"]) for it in new_items}
        save_sent_cache(cache)
    return {"count": len(new_items), "items": new_items}


# -------------------------
# 스케줄러 유지 (Render 호환)
# -------------------------
@st.cache_resource
def get_scheduler() -> BackgroundScheduler:
    sched = BackgroundScheduler(timezone=APP_TZ)
    sched.start()
    return sched


def scheduled_job():
    cfg = load_config(CURRENT_CFG_PATH)
    transmit_once(cfg, CURRENT_ENV, preview=False)


def ensure_interval_job(sched: BackgroundScheduler, minutes: int):
    job_id = "pe_news_job"
    try:
        sched.remove_job(job_id)
    except Exception:
        pass
    sched.add_job(scheduled_job, "interval", minutes=minutes, id=job_id, next_run_time=now_kst())


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="PE 동향 뉴스 모니터링", page_icon="📰", layout="wide")

cfg_path = st.sidebar.text_input("config.json 경로", value=DEFAULT_CONFIG_PATH)
cfg = load_config(cfg_path)
st.sidebar.caption(f"Config 로드 상태: {'✅' if cfg else '❌'}")

bot_token = st.sidebar.text_input("텔레그램 봇 토큰", type="password", value=os.getenv("TELEGRAM_BOT_TOKEN", ""))
chat_id = st.sidebar.text_input("텔레그램 Chat ID", value=os.getenv("TELEGRAM_CHAT_ID", ""))
naver_id = st.sidebar.text_input("Naver Client ID", type="password", value=os.getenv("NAVER_CLIENT_ID", ""))
naver_secret = st.sidebar.text_input("Naver Client Secret", type="password", value=os.getenv("NAVER_CLIENT_SECRET", ""))

cfg["INTERVAL_MIN"] = st.sidebar.number_input("전송 주기(분)", min_value=5, max_value=180, step=5, value=int(cfg.get("INTERVAL_MIN", 60)))

st.title("📰 국내 PE 동향 뉴스 자동 모니터링")
st.caption("Streamlit + Naver API + Telegram + APScheduler")

def make_env():
    return {
        "NAVER_CLIENT_ID": naver_id,
        "NAVER_CLIENT_SECRET": naver_secret,
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
        global CURRENT_CFG_PATH, CURRENT_ENV
        CURRENT_CFG_PATH = cfg_path
        CURRENT_ENV = make_env()
        ensure_interval_job(sched, int(cfg["INTERVAL_MIN"]))
        # 즉시 첫 실행
        scheduled_job()
        st.success("스케줄 시작됨 (즉시 전송 후 주기적 실행)")

st.divider()
st.subheader("📋 필터링된 전체 기사")
res = st.session_state.get("preview", {"items": []})
items = res.get("items", [])

if not items:
    st.write("결과 없음")
else:
    for it in items:
        t = it["title"]
        u = it["url"]
        try:
            pub = dt.datetime.strptime(it["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).astimezone(APP_TZ)
            when = pub.strftime("%Y-%m-%d %H:%M")
        except Exception:
            when = "-"
        st.markdown(f"- <a href='{u}'>{t}</a> ({when})", unsafe_allow_html=True)
