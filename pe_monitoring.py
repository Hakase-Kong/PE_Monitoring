import os
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import requests
import streamlit as st

# ---- Optional: 사용 시 백그라운드 주기 실행 ----
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# ---------- Logging ----------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

# ---------- Secrets/Env helper ----------
def get_secret(key: str, default: str = "") -> str:
    """
    secrets.toml(st.secrets) → 환경변수(os.environ) → default 순으로 읽는다.
    Render에서 Secret Files가 없어도 Environment Variables만으로 동작하도록 함.
    """
    try:
        return st.secrets.get(key, default)  # secrets.toml이 없으면 여기서 예외 가능
    except Exception:
        return os.environ.get(key, default)

# ---------- Utilities ----------
def read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def write_json(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# ---------- Config ----------
STORAGE_DIR = os.environ.get("STORAGE_DIR", ".pe_news_state")
os.makedirs(STORAGE_DIR, exist_ok=True)
SENT_DB_PATH = os.path.join(STORAGE_DIR, "sent_urls.json")

DEFAULT_KEYWORDS = [
    "국내 PE",
    "사모펀드",
    "바이아웃",
    "경쟁입찰",
    "경영권 인수",
    "딜 클로징",
    "매각 본입찰",
    "예비입찰",
    "리캡",
    "구주매출",
    "경영권 프리미엄",
]

# ---------- Providers ----------
@dataclass
class Article:
    title: str
    url: str
    source: str
    published_at: str  # ISO 8601
    description: Optional[str] = None

class NewsProvider:
    def fetch(self, query: str, page_size: int = 20) -> List[Article]:
        raise NotImplementedError

class NewsAPIProvider(NewsProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, query: str, page_size: int = 20) -> List[Article]:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "ko",
            "pageSize": page_size,
            "sortBy": "publishedAt",
        }
        headers = {"X-Api-Key": self.api_key}
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        arts = []
        for a in data.get("articles", []):
            published = a.get("publishedAt") or datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
            src = (a.get("source") or {}).get("name") or "NewsAPI"
            arts.append(Article(
                title=a.get("title") or "",
                url=a.get("url") or "",
                source=src,
                published_at=published,
                description=a.get("description") or "",
            ))
        return arts

class NaverNewsProvider(NewsProvider):
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret

    def fetch(self, query: str, page_size: int = 20) -> List[Article]:
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }
        params = {
            "query": query,
            "display": min(page_size, 100),
            "start": 1,
            "sort": "date",
        }
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        arts = []
        for it in data.get("items", []):
            link = it.get("link") or it.get("originallink") or ""
            pub = it.get("pubDate") or ""
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(pub)
                published = dt.astimezone(timezone.utc).isoformat()
            except Exception:
                published = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
            arts.append(Article(
                title=it.get("title") or "",
                url=link,
                source="Naver",
                published_at=published,
                description=it.get("description"),
            ))
        return arts

# ---------- Telegram ----------
def send_telegram_message(bot_token: str, chat_id: str, text: str, disable_web_page_preview=True):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": disable_web_page_preview,
    }
    r = requests.post(url, data=payload, timeout=20)
    if r.status_code != 200:
        logging.error("Telegram send failed: %s", r.text)
    r.raise_for_status()
    return r.json()

# ---------- Core ----------
def normalize_url(u: str) -> str:
    if not u:
        return u
    return u.strip().replace("http://", "https://")

def compose_message(keyword: str, articles: List[Article]) -> str:
    lines = [f"📌 <b>PE 동향 뉴스 ({keyword})</b>"]
    for a in articles:
        dt = a.published_at[:16].replace("T", " ")
        title = a.title.replace("&quot;", '"').replace("&amp;", "&")
        lines.append(f"• <a href=\"{a.url}\">{title}</a> — <i>{a.source}</i> ({dt})")
    text = "\n".join(lines)
    return text[:4000]  # Telegram limit safe-guard

def fetch_articles(providers: List[NewsProvider], keyword: str, page_size: int) -> List[Article]:
    agg: Dict[str, Article] = {}
    for p in providers:
        try:
            for a in p.fetch(keyword, page_size=page_size):
                u = normalize_url(a.url)
                if not u:
                    continue
                if u not in agg:  # dedup by URL
                    agg[u] = Article(
                        title=a.title,
                        url=u,
                        source=a.source,
                        published_at=a.published_at,
                        description=a.description,
                    )
        except Exception as e:
            logging.warning("Provider fetch error (%s): %s", p.__class__.__name__, e)
    return sorted(agg.values(), key=lambda x: x.published_at, reverse=True)

def filter_unsent(articles: List[Article], sent_db: Dict[str, bool], max_items: int) -> List[Article]:
    out = []
    for a in articles:
        if a.url in sent_db:
            continue
        out.append(a)
        if len(out) >= max_items:
            break
    return out

# ---------- Streamlit UI & Scheduler ----------
if "scheduler" not in st.session_state:
    st.session_state.scheduler = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "last_run" not in st.session_state:
    st.session_state.last_run = None

st.set_page_config(page_title="PE 동향 뉴스 → Telegram", page_icon="📨", layout="wide")
st.title("📨 PE 동향 뉴스 → Telegram 자동 전송")
st.caption("Streamlit + NewsAPI/Naver + Telegram + APScheduler")

with st.sidebar:
    st.subheader("자격증명 / 설정")
    # secrets.toml 또는 환경변수에서 기본값 로드
    default_newsapi      = get_secret("NEWSAPI_KEY")
    default_naver_id     = get_secret("NAVER_CLIENT_ID")
    default_naver_secret = get_secret("NAVER_CLIENT_SECRET")
    default_bot_token    = get_secret("TELEGRAM_BOT_TOKEN")
    default_chat_id      = get_secret("TELEGRAM_CHAT_ID")

    newsapi_key = st.text_input("NewsAPI Key (선택)", value=default_newsapi, type="password")
    naver_client_id = st.text_input("Naver Client ID (선택)", value=default_naver_id, type="password")
    naver_client_secret = st.text_input("Naver Client Secret (선택)", value=default_naver_secret, type="password")

    telegram_bot_token = st.text_input("Telegram Bot Token", value=default_bot_token, type="password")
    telegram_chat_id = st.text_input("Telegram Chat ID (본인/그룹)", value=default_chat_id)

    st.divider()
    st.subheader("키워드 / 빈도 / 전송 옵션")
    keywords_str = st.text_area("키워드(쉼표로 구분)", value=", ".join(DEFAULT_KEYWORDS), height=100)
    page_size = st.number_input("뉴스 소스별 페이지당 수집 수", min_value=5, max_value=100, value=30, step=5)
    max_per_keyword = st.number_input("전송 건수 제한(키워드별)", min_value=1, max_value=50, value=10, step=1)
    interval_min = st.number_input("전송 주기(분)", min_value=5, max_value=720, value=60, step=5)
    disable_preview = st.checkbox("미리보기(링크 프리뷰) 비활성화", value=True)

# Build providers
providers: List[NewsProvider] = []
if newsapi_key:
    providers.append(NewsAPIProvider(newsapi_key))
if naver_client_id and naver_client_secret:
    providers.append(NaverNewsProvider(naver_client_id, naver_client_secret))

if not providers:
    st.warning("최소 하나의 뉴스 제공자(NewsAPI 또는 Naver)를 설정하세요.")

sent_db: Dict[str, bool] = read_json(SENT_DB_PATH, default={})

def do_run_once() -> Tuple[int, int]:
    """모든 키워드에 대해 수집→신규필터→전송→DB기록. (전송한 키워드 수, 기사 수) 반환."""
    if not telegram_bot_token or not telegram_chat_id:
        st.error("Telegram 토큰/채팅 ID가 필요합니다.")
        return 0, 0
    if not providers:
        st.error("설정된 뉴스 제공자가 없습니다.")
        return 0, 0

    total_sent = 0
    kw_sent_cnt = 0
    for kw in [k.strip() for k in keywords_str.split(",") if k.strip()]:
        arts = fetch_articles(providers, kw, page_size=page_size)
        new_arts = filter_unsent(arts, sent_db, max_items=max_per_keyword)
        if not new_arts:
            continue

        msg = compose_message(kw, new_arts)
        try:
            send_telegram_message(telegram_bot_token, telegram_chat_id, msg, disable_web_page_preview=disable_preview)
            kw_sent_cnt += 1
            total_sent += len(new_arts)
            for a in new_arts:
                sent_db[a.url] = True
        except Exception as e:
            st.error(f"텔레그램 전송 실패 ({kw}): {e}")

    write_json(SENT_DB_PATH, sent_db)
    return kw_sent_cnt, total_sent

def ensure_scheduler():
    if st.session_state.scheduler is None:
        st.session_state.scheduler = BackgroundScheduler(timezone="Asia/Seoul")
        st.session_state.scheduler.start()

def start_schedule(every_minutes: int):
    ensure_scheduler()
    for job in st.session_state.scheduler.get_jobs():
        st.session_state.scheduler.remove_job(job.id)
    st.session_state.scheduler.add_job(
        func=lambda: _scheduled_send(),
        trigger=IntervalTrigger(minutes=every_minutes),
        id="pe_news_push",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    st.session_state.is_running = True

def stop_schedule():
    if st.session_state.scheduler:
        for job in st.session_state.scheduler.get_jobs():
            st.session_state.scheduler.remove_job(job.id)
    st.session_state.is_running = False

def _scheduled_send():
    try:
        kw_cnt, sent_cnt = do_run_once()
        logging.info("Scheduled run finished: %s keywords, %s articles sent", kw_cnt, sent_cnt)
    except Exception as e:
        logging.exception("Scheduled run error: %s", e)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("지금 한 번 실행"):
        k, t = do_run_once()
        st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"전송 완료: {k}개 키워드, {t}건 기사 전송")
with col2:
    if st.button("스케줄 시작"):
        start_schedule(int(interval_min))
        st.success(f"스케줄 시작: {interval_min}분 간격")
with col3:
    if st.button("스케줄 중지"):
        stop_schedule()
        st.info("스케줄 중지")

st.divider()
st.subheader("상태")
st.write(f"Scheduler 실행 중: {st.session_state.is_running}")
if st.session_state.last_run:
    st.write(f"마지막 수동 실행: {st.session_state.last_run}")

# Preview (첫 번째 키워드 기준)
if providers:
    try:
        first_kw = [k.strip() for k in keywords_str.split(",") if k.strip()][0]
        preview = fetch_articles(providers, first_kw, page_size=10)
        st.subheader(f"미리보기: “{first_kw}” 최신 10건")
        for a in preview:
            st.markdown(
                f"- [{a.title}]({a.url})  \n  <small>{a.source} • {a.published_at[:16].replace('T',' ')}</small>",
                unsafe_allow_html=True
            )
    except Exception as e:
        st.warning(f"미리보기 로드 실패: {e}")
else:
    st.info("프로바이더 설정 후 미리보기가 표시됩니다.")
