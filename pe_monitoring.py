import os
import re
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse

import requests
import streamlit as st

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# ---------- Logging ----------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

# ---------- Secrets/Env helper ----------
def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)  # secrets.toml 없으면 예외 가능
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

def normalize_url(u: str) -> str:
    if not u:
        return u
    return u.strip().replace("http://", "https://")

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def html_unescape(s: str) -> str:
    return s.replace("&quot;", '"').replace("&amp;", "&").replace("&apos;", "'").replace("&lt;", "<").replace("&gt;", ">")

def now_utc() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc)

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
            published = a.get("publishedAt") or now_utc().isoformat()
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
                published = now_utc().isoformat()
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

# ---------- Ranking / Filtering ----------
QUALITY_WEIGHTS = {
    # 가중치 상향(도메인 포함 시 가산점)
    "thebell.co.kr": 3.0,
    "investchosun.com": 3.0,
    "dealsite.co.kr": 2.5,
    "news.naver.com": 1.0,
    # 가중치 하향(스포츠/연성 기사)
    "sports.naver.com": -5.0,
    "m.sports.naver.com": -5.0,
}

def iso_to_dt(iso_str: str) -> datetime:
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except Exception:
        return now_utc()

def hours_ago(iso_str: str) -> float:
    dt = iso_to_dt(iso_str)
    return (now_utc() - dt).total_seconds() / 3600.0

def should_keep(a: Article,
                allow_domains: List[str],
                block_domains: List[str],
                must_include_terms: List[str],
                must_exclude_terms: List[str],
                recency_hours: int) -> bool:
    d = domain_of(a.url)
    # 도메인 차단
    if d in block_domains:
        return False
    # 도메인 허용 리스트가 있으면 화이트리스트 방식
    if allow_domains and d not in allow_domains:
        return False
    # 신선도
    if recency_hours > 0 and hours_ago(a.published_at) > recency_hours:
        return False
    title = html_unescape(a.title).lower()
    # 포함 키워드
    if must_include_terms:
        if not any(t for t in must_include_terms if t and t.lower() in title):
            return False
    # 제외 키워드
    if any(t for t in must_exclude_terms if t and t.lower() in title):
        return False
    return True

def score_article(a: Article,
                  domain_weights: Dict[str, float],
                  include_terms: List[str],
                  exclude_terms: List[str]) -> float:
    score = 0.0
    d = domain_of(a.url)
    score += domain_weights.get(d, 0.0)
    # 신선도 가산(최대 2점, 0~48시간 선형)
    h = max(0.0, min(48.0, hours_ago(a.published_at)))
    score += (48.0 - h) / 48.0 * 2.0
    title = html_unescape(a.title).lower()
    if any(t for t in include_terms if t and t.lower() in title):
        score += 1.0
    if any(t for t in exclude_terms if t and t.lower() in title):
        score -= 2.0
    # 제목 최소 길이 패널티
    if len(title) < 10:
        score -= 1.0
    return score

# ---------- Core ----------
def compose_message(keyword: str, articles: List[Article]) -> str:
    lines = [f"📌 <b>PE 동향 뉴스 ({keyword})</b>"]
    for a in articles:
        dt = a.published_at[:16].replace("T", " ")
        lines.append(f"• <a href=\"{a.url}\">{html_unescape(a.title)}</a> — <i>{a.source}</i> ({dt})")
    text = "\n".join(lines)
    return text[:4000]  # Telegram limit guard

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
    return list(agg.values())

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
    disable_preview = st.checkbox("링크 프리뷰 비활성화", value=True)
    only_working_hours = st.checkbox("업무시간(08–20 KST) 내에서만 전송", value=True)
    test_mode = st.checkbox("테스트 모드(텔레그램 전송 생략, 로그/미리보기만)", value=False)

    st.divider()
    st.subheader("필터링/랭킹 고급 옵션")
    recency_hours = st.number_input("신선도 제한(최근 N시간 기사만)", min_value=0, max_value=168, value=48, step=1)
    allow_domains_text = st.text_input("허용 도메인(쉼표; 비우면 전체 허용)", value="")
    block_domains_text = st.text_input("차단 도메인(쉼표)", value="sports.naver.com, m.sports.naver.com")
    include_terms_text = st.text_input("포함 키워드(제목에 하나 이상 포함되면 통과; 비우면 미적용)", value="")
    exclude_terms_text = st.text_input("제외 키워드(제목에 포함되면 제외)", value="축구,야구,농구,배구,케인,손흥민")

# Build providers
providers: List[NewsProvider] = []
if newsapi_key:
    providers.append(NewsAPIProvider(newsapi_key))
if naver_client_id and naver_client_secret:
    providers.append(NaverNewsProvider(naver_client_id, naver_client_secret))

if not providers:
    st.warning("최소 하나의 뉴스 제공자(NewsAPI 또는 Naver)를 설정하세요.")

# Parse filters
def csv_to_list(s: str) -> List[str]:
    return [x.strip().lower() for x in s.split(",") if x.strip()]

allow_domains = csv_to_list(allow_domains_text)
block_domains = csv_to_list(block_domains_text)
include_terms = csv_to_list(include_terms_text)
exclude_terms = csv_to_list(exclude_terms_text)

sent_db: Dict[str, bool] = read_json(SENT_DB_PATH, default={})

def do_run_once() -> Tuple[int, int]:
    """모든 키워드에 대해 수집→필터/랭킹→신규필터→(전송|테스트)→DB기록. (전송한 키워드 수, 기사 수) 반환."""
    if not telegram_bot_token or not telegram_chat_id:
        if not test_mode:
            st.error("Telegram 토큰/채팅 ID가 필요합니다.")
            return 0, 0
    if not providers:
        st.error("설정된 뉴스 제공자가 없습니다.")
        return 0, 0

    # 업무시간 제한
    if only_working_hours:
        kst_hour = (datetime.utcnow() + timedelta(hours=9)).hour  # KST = UTC+9
        if kst_hour < 8 or kst_hour >= 20:
            logging.info("Skipped by working-hours window (KST %02d)", kst_hour)
            return 0, 0

    total_sent = 0
    kw_sent_cnt = 0

    for kw in [k.strip() for k in keywords_str.split(",") if k.strip()]:
        # 1) 수집
        raw = fetch_articles(providers, kw, page_size=page_size)

        # 2) 필터링
        filt = [
            a for a in raw
            if should_keep(
                a,
                allow_domains=allow_domains,
                block_domains=block_domains,
                must_include_terms=include_terms,
                must_exclude_terms=exclude_terms,
                recency_hours=recency_hours,
            )
        ]

        # 3) 랭킹
        ranked = sorted(
            filt,
            key=lambda x: (score_article(x, QUALITY_WEIGHTS, include_terms, exclude_terms), x.published_at),
            reverse=True,
        )

        # 4) 중복/발송수 제한
        new_arts = filter_unsent(ranked, sent_db, max_items=max_per_keyword)
        if not new_arts:
            continue

        # 5) 전송/테스트
        msg = compose_message(kw, new_arts)
        try:
            if test_mode:
                logging.info("[TEST MODE] Would send %d items for kw '%s'", len(new_arts), kw)
            else:
                send_telegram_message(telegram_bot_token, telegram_chat_id, msg, disable_web_page_preview=disable_preview)
            kw_sent_cnt += 1
            total_sent += len(new_arts)
            for a in new_arts:
                sent_db[a.url] = True
        except Exception as e:
            st.error(f"텔레그램 전송 실패 ({kw}): {e}")

    write_json(SENT_DB_PATH, sent_db)
    return kw_sent_cnt, total_sent

# ---------- Scheduler ----------
if "scheduler" not in st.session_state:
    st.session_state.scheduler = None

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

# ---------- UI ----------
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("지금 한 번 실행"):
        k, t = do_run_once()
        st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"완료: {k}개 키워드, {t}건 기사 처리{' (전송 생략: 테스트 모드)' if test_mode else ''}")
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
        preview_raw = fetch_articles(providers, first_kw, page_size=30)
        preview_filt = [
            a for a in preview_raw
            if should_keep(
                a,
                allow_domains=allow_domains,
                block_domains=block_domains,
                must_include_terms=include_terms,
                must_exclude_terms=exclude_terms,
                recency_hours=recency_hours,
            )
        ]
        preview_ranked = sorted(
            preview_filt,
            key=lambda x: (score_article(x, QUALITY_WEIGHTS, include_terms, exclude_terms), x.published_at),
            reverse=True,
        )[:10]
        st.subheader(f"미리보기: “{first_kw}” 최신·필터 통과 상위 10건")
        for a in preview_ranked:
            st.markdown(
                f"- [{html_unescape(a.title)}]({a.url})  \n  <small>{domain_of(a.url)} • {a.published_at[:16].replace('T',' ')}</small>",
                unsafe_allow_html=True
            )
    except Exception as e:
        st.warning(f"미리보기 로드 실패: {e}")
else:
    st.info("프로바이더 설정 후 미리보기가 표시됩니다.")
