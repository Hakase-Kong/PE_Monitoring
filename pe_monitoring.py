import os
import re
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, urlunparse, parse_qsl

import requests
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime

# =========================
# 로깅
# =========================
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

# =========================
# 경로/상수
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.environ.get("STORAGE_DIR", os.path.join(BASE_DIR, ".pe_news_state"))
os.makedirs(STORAGE_DIR, exist_ok=True)
SENT_DB_PATH = os.path.join(STORAGE_DIR, "sent_urls.json")

CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.join(BASE_DIR, "config.json"))

DEFAULT_CONFIG = {
    "KEYWORDS": [],
    "KEYWORD_ALIASES": {},
    "FIRM_WATCHLIST": [],
    "ALLOW_DOMAINS": [],
    "BLOCK_DOMAINS": ["sports.naver.com", "m.sports.naver.com"],
    "INCLUDE_TITLE_KEYWORDS": [],
    "EXCLUDE_TITLE_KEYWORDS": [],
    "DOMAIN_WEIGHTS": {
        "thebell.co.kr": 3.0,
        "investchosun.com": 3.0,
        "dealsite.co.kr": 2.5,
        "news.naver.com": 1.0,
        "sports.naver.com": -5.0,
        "m.sports.naver.com": -5.0
    },
    "RECENCY_HOURS": 48,
    "ONLY_WORKING_HOURS": True,
    "TELEGRAM_DISABLE_PREVIEW": True,
    "MAX_PER_KEYWORD": 10,
    "PAGE_SIZE": 30,
    "CREDENTIALS": {}
}

# =========================
# 공통 유틸
# =========================
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

def load_config(path: str = CONFIG_PATH) -> Dict:
    cfg = DEFAULT_CONFIG.copy()
    file_cfg = read_json(path, default=None)
    if file_cfg is None:
        logging.warning("config.json not found. Using DEFAULT_CONFIG. path=%s", path)
        return cfg
    cfg.update(file_cfg)
    return cfg

CONFIG = load_config()

def get_secret(key: str, default: str = "") -> str:
    """환경변수 → config.json(CREDENTIALS) 순으로 조회."""
    env_val = os.environ.get(key)
    if env_val:
        return env_val
    creds = CONFIG.get("CREDENTIALS", {})
    return creds.get(key, default)

def now_utc() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc)

def html_unescape(s: str) -> str:
    return (s or "").replace("&quot;", '"').replace("&amp;", "&").replace("&apos;", "'")\
                    .replace("&lt;", "<").replace("&gt;", ">")

# URL 정규화(추적 파라미터 제거)
TRACKING_PARAMS = {
    "utm_source","utm_medium","utm_campaign","utm_term","utm_content",
    "inflow","sid","oid","aid","mode","ref","feature","from"
}
def normalize_url(u: str) -> str:
    u = (u or "").strip().replace("http://", "https://")
    try:
        p = urlparse(u)
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if k.lower() not in TRACKING_PARAMS]
        return urlunparse((
            p.scheme,
            p.netloc.lower(),
            p.path,
            p.params,
            "&".join([f"{k}={v}" for k, v in q]),
            ""  # fragment 제거
        ))
    except Exception:
        return u

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def iso_to_dt(iso_str: str) -> datetime:
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except Exception:
        return now_utc()

def hours_ago(iso_str: str) -> float:
    return (now_utc() - iso_to_dt(iso_str)).total_seconds() / 3600.0

# =========================
# 데이터 모델/공급자
# =========================
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
        params = {"q": query, "language": "ko", "pageSize": page_size, "sortBy": "publishedAt"}
        r = requests.get(url, params=params, headers={"X-Api-Key": self.api_key}, timeout=20)
        r.raise_for_status()
        data = r.json()
        arts = []
        for a in data.get("articles", []):
            arts.append(Article(
                title=a.get("title") or "",
                url=a.get("url") or "",
                source=(a.get("source") or {}).get("name") or "NewsAPI",
                published_at=(a.get("publishedAt") or now_utc().isoformat()),
                description=a.get("description") or "",
            ))
        return arts

class NaverNewsProvider(NewsProvider):
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
    def fetch(self, query: str, page_size: int = 20) -> List[Article]:
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {"X-Naver-Client-Id": self.client_id, "X-Naver-Client-Secret": self.client_secret}
        params = {"query": query, "display": min(page_size, 100), "start": 1, "sort": "date"}
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        arts = []
        for it in data.get("items", []):
            pub = it.get("pubDate") or ""
            try:
                published = parsedate_to_datetime(pub).astimezone(timezone.utc).isoformat()
            except Exception:
                published = now_utc().isoformat()
            arts.append(Article(
                title=it.get("title") or "",
                url=(it.get("link") or it.get("originallink") or ""),
                source="Naver",
                published_at=published,
                description=it.get("description") or "",
            ))
        return arts

# =========================
# 랭킹/필터/중복제거
# =========================
WORD_RE = re.compile(r"[가-힣A-Za-z0-9]+")

def title_key(title: str) -> str:
    t = html_unescape(title).lower()
    tokens = WORD_RE.findall(t)
    tokens = [w for w in tokens if len(w) >= 2 and not w.isdigit()]
    return " ".join(tokens[:30])

def is_similar(t1: str, t2: str, threshold: float = 0.86) -> bool:
    return SequenceMatcher(None, title_key(t1), title_key(t2)).ratio() >= threshold

def dedup_by_title(articles: List[Article], score_fn) -> List[Article]:
    reps: List[Article] = []
    for a in articles:
        dup = False
        for i, r in enumerate(reps):
            if is_similar(a.title, r.title):
                if score_fn(a) > score_fn(r):
                    reps[i] = a
                dup = True
                break
        if not dup:
            reps.append(a)
    return reps

def should_keep(a: Article,
                allow_domains: List[str],
                block_domains: List[str],
                must_include_terms: List[str],
                must_exclude_terms: List[str],
                recency_hours: int) -> bool:
    d = domain_of(a.url)
    if d in block_domains:
        return False
    if allow_domains and d not in allow_domains:
        return False
    if recency_hours > 0 and hours_ago(a.published_at) > recency_hours:
        return False
    title = html_unescape(a.title).lower()
    if must_include_terms and not any(t for t in must_include_terms if t in title):
        return False
    if any(t for t in must_exclude_terms if t in title):
        return False
    return True

# =========================
# Telegram
# =========================
def send_telegram_message(bot_token: str, chat_id: str, text: str, disable_web_page_preview=True):
    r = requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        data={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": disable_web_page_preview
        },
        timeout=20,
    )
    if r.status_code != 200:
        logging.error("Telegram send failed: %s", r.text)
    r.raise_for_status()
    return r.json()

def compose_message(keyword: str, articles: List[Article]) -> str:
    lines = [f"📌 <b>PE 동향 뉴스 ({keyword})</b>"]
    for a in articles:
        dt = a.published_at[:16].replace("T", " ")
        lines.append(
            f"• <a href=\"{a.url}\">{html_unescape(a.title)}</a> — <i>{a.source}</i> ({dt})"
        )
    return "\n".join(lines)[:4000]  # 텔레그램 4096자 제한 가드

# =========================
# Streamlit 상태
# =========================
if "scheduler" not in st.session_state:
    st.session_state.scheduler = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "last_run" not in st.session_state:
    st.session_state.last_run = None

st.set_page_config(page_title="PE 동향 뉴스 → Telegram", page_icon="📨", layout="wide")
st.title("📨 PE 동향 뉴스 → Telegram 자동 전송")
st.caption("키워드는 전부 config.json에서 관리합니다.")

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.subheader("자격증명 / 설정")
    st.caption(f"CONFIG 경로: {CONFIG_PATH} / 존재: {os.path.isfile(CONFIG_PATH)}")

    newsapi_key = st.text_input("NewsAPI Key (선택)", value=get_secret("NEWSAPI_KEY"), type="password")
    naver_client_id = st.text_input("Naver Client ID (선택)", value=get_secret("NAVER_CLIENT_ID"), type="password")
    naver_client_secret = st.text_input("Naver Client Secret (선택)", value=get_secret("NAVER_CLIENT_SECRET"), type="password")

    telegram_bot_token = st.text_input("Telegram Bot Token", value=get_secret("TELEGRAM_BOT_TOKEN"), type="password")
    telegram_chat_id   = st.text_input("Telegram Chat ID (본인/그룹)", value=get_secret("TELEGRAM_CHAT_ID"))

    st.divider()
    st.subheader("config.json")
    if st.button("구성 리로드"):
        global CONFIG
        CONFIG = load_config()
        st.success("config.json을 다시 불러왔습니다.")

    st.text("KEYWORDS (읽기전용)")
    st.code("\n".join(CONFIG.get("KEYWORDS", [])) or "(none)", language="text")

    def csv(v): return ", ".join([x for x in v if isinstance(x, str)])
    page_size        = st.number_input("페이지당 수집 수", min_value=5, max_value=100, value=int(CONFIG.get("PAGE_SIZE", 30)), step=5)
    max_per_keyword  = st.number_input("전송 건수 제한(키워드별)", min_value=1, max_value=50, value=int(CONFIG.get("MAX_PER_KEYWORD", 10)), step=1)
    interval_min     = st.number_input("전송 주기(분)", min_value=5, max_value=720, value=60, step=5)
    recency_hours    = st.number_input("신선도(최근 N시간)", min_value=0, max_value=168, value=int(CONFIG.get("RECENCY_HOURS", 48)), step=1)
    only_working     = st.checkbox("업무시간(08–20 KST) 내 전송", value=bool(CONFIG.get("ONLY_WORKING_HOURS", True)))
    disable_preview  = st.checkbox("링크 프리뷰 비활성화", value=bool(CONFIG.get("TELEGRAM_DISABLE_PREVIEW", True)))
    test_mode        = st.checkbox("테스트 모드(전송 생략)", value=False)

    allow_domains_txt = st.text_input("허용 도메인(쉼표; 비우면 전체 허용)", value=csv(CONFIG.get("ALLOW_DOMAINS", [])))
    block_domains_txt = st.text_input("차단 도메인(쉼표)", value=csv(CONFIG.get("BLOCK_DOMAINS", [])))
    include_terms_txt = st.text_input("포함 키워드(제목)", value=csv(CONFIG.get("INCLUDE_TITLE_KEYWORDS", [])))
    exclude_terms_txt = st.text_input("제목 제외 키워드", value=csv(CONFIG.get("EXCLUDE_TITLE_KEYWORDS", [])))

def csv_to_list(s: str) -> List[str]:
    return [x.strip().lower() for x in s.split(",") if x.strip()]

allow_domains = csv_to_list(allow_domains_txt)
block_domains = csv_to_list(block_domains_txt)
include_terms = csv_to_list(include_terms_txt)
exclude_terms = csv_to_list(exclude_terms_txt)

DOMAIN_WEIGHTS = CONFIG.get("DOMAIN_WEIGHTS", {})
FIRM_WATCH = [s.lower() for s in CONFIG.get("FIRM_WATCHLIST", [])]

# =========================
# 공급자 구성
# =========================
providers: List[NewsProvider] = []
if newsapi_key:
    providers.append(NewsAPIProvider(newsapi_key))
if naver_client_id and naver_client_secret:
    providers.append(NaverNewsProvider(naver_client_id, naver_client_secret))
if not providers:
    st.warning("최소 하나의 뉴스 제공자(NewsAPI 또는 Naver)를 설정하세요.")

# =========================
# 수집 파이프라인
# =========================
def expand_queries(kw: str) -> List[str]:
    aliases = CONFIG.get("KEYWORD_ALIASES", {}).get(kw, [])
    base = [kw] + [a for a in aliases if a != kw]
    enriched: List[str] = []
    for q in base:
        ql = q.lower()
        if ql in {"buyout", "m&a", "private equity"}:
            enriched.append(f"{q} AND (인수 OR 매각 OR 딜 OR PEF)")
        else:
            enriched.append(q)
    seen = set(); uniq=[]
    for q in enriched:
        if q not in seen:
            uniq.append(q); seen.add(q)
    return uniq

def fetch_articles(providers: List[NewsProvider], query: str, page_size: int) -> List[Article]:
    agg: Dict[str, Article] = {}
    for p in providers:
        try:
            for a in p.fetch(query, page_size=page_size):
                u = normalize_url(a.url)
                if u and u not in agg:
                    agg[u] = Article(a.title, u, a.source, a.published_at, a.description)
        except Exception as e:
            logging.warning("Provider fetch error (%s, %s): %s", p.__class__.__name__, query, e)
    return list(agg.values())

def score_article(a: Article) -> float:
    score = DOMAIN_WEIGHTS.get(domain_of(a.url), 0.0)
    h = max(0.0, min(48.0, hours_ago(a.published_at)))
    score += (48.0 - h) / 48.0 * 2.0  # 최신 가점
    title = html_unescape(a.title).lower()
    if any(t for t in include_terms if t in title):
        score += 1.0
    if any(t for t in exclude_terms if t in title):
        score -= 2.0
    if any(f in title for f in FIRM_WATCH):
        score += 0.8
    if len(title) < 10:
        score -= 1.0
    return score

def do_run_once() -> Tuple[int, int]:
    if not providers:
        st.error("설정된 뉴스 제공자가 없습니다.")
        return 0, 0
    if (not telegram_bot_token or not telegram_chat_id) and not test_mode:
        st.error("Telegram 토큰/채팅 ID가 필요합니다.")
        return 0, 0
    if only_working:
        kst_hour = (datetime.utcnow() + timedelta(hours=9)).hour
        if kst_hour < 8 or kst_hour >= 20:
            logging.info("Skipped by working-hours window (KST %02d)", kst_hour)
            return 0, 0

    sent_db: Dict[str, bool] = read_json(SENT_DB_PATH, default={})
    total_sent = 0
    kw_sent_cnt = 0

    keywords = CONFIG.get("KEYWORDS", [])
    for kw in [k.strip() for k in keywords if k.strip()]:
        raw: List[Article] = []
        for q in expand_queries(kw):
            raw.extend(fetch_articles(providers, q, page_size=int(CONFIG.get("PAGE_SIZE", 30))))

        url_seen: Dict[str, Article] = {}
        for a in raw:
            u = normalize_url(a.url)
            if u and u not in url_seen:
                url_seen[u] = a
        raw = list(url_seen.values())

        filt = [a for a in raw if should_keep(
            a,
            allow_domains,
            block_domains,
            include_terms,
            exclude_terms,
            int(CONFIG.get("RECENCY_HOURS", 48))
        )]

        filt = dedup_by_title(filt, score_article)

        ranked = sorted(
            filt,
            key=lambda x: (score_article(x), x.published_at),
            reverse=True
        )

        new_arts: List[Article] = []
        for a in ranked:
            if a.url in sent_db:
                continue
            new_arts.append(a)
            if len(new_arts) >= int(CONFIG.get("MAX_PER_KEYWORD", 10)):
                break

        if not new_arts:
            continue

        msg = compose_message(kw, new_arts)
        try:
            if test_mode:
                logging.info("[TEST MODE] Would send %d items for '%s'", len(new_arts), kw)
            else:
                send_telegram_message(
                    telegram_bot_token, telegram_chat_id, msg,
                    disable_web_page_preview=bool(CONFIG.get("TELEGRAM_DISABLE_PREVIEW", True))
                )
            kw_sent_cnt += 1
            total_sent += len(new_arts)
            for a in new_arts:
                sent_db[a.url] = True
        except Exception as e:
            st.error(f"텔레그램 전송 실패 ({kw}): {e}")

        write_json(SENT_DB_PATH, sent_db)

    return kw_sent_cnt, total_sent

# =========================
# 스케줄러
# =========================
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
        coalesce=True
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

# =========================
# 메인 UI
# =========================
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("지금 한 번 실행"):
        k, t = do_run_once()
        st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"완료: {k}개 키워드, {t}건 기사 처리{' (전송 생략: 테스트 모드)' if st.session_state.get('test_mode', False) else ''}")
with c2:
    if st.button("스케줄 시작"):
        start_schedule(60)  # 기본 60분 (사이드바 interval_min을 사용하려면 연결)
        st.success("스케줄 시작")
with c3:
    if st.button("스케줄 중지"):
        stop_schedule()
        st.info("스케줄 중지")

st.divider()
st.subheader("상태")
st.write(f"Scheduler 실행 중: {st.session_state.is_running}")
if st.session_state.last_run:
    st.write(f"마지막 수동 실행: {st.session_state.last_run}")

# 미리보기(첫 키워드 상위 10건)
if providers and CONFIG.get("KEYWORDS"):
    try:
        first_kw = CONFIG["KEYWORDS"][0]
        preview: List[Article] = []
        for q in expand_queries(first_kw):
            preview.extend(fetch_articles(providers, q, page_size=int(CONFIG.get("PAGE_SIZE", 30))))

        url_seen = {}
        for a in preview:
            u = normalize_url(a.url)
            if u and u not in url_seen:
                url_seen[u] = a
        preview = list(url_seen.values())
        preview = [a for a in preview if should_keep(
            a, csv_to_list(""), csv_to_list(",".join(CONFIG.get("BLOCK_DOMAINS", []))),
            csv_to_list(",".join(CONFIG.get("INCLUDE_TITLE_KEYWORDS", []))),
            csv_to_list(",".join(CONFIG.get("EXCLUDE_TITLE_KEYWORDS", []))),
            int(CONFIG.get("RECENCY_HOURS", 48))
        )]
        preview = dedup_by_title(preview, score_article)
        preview = sorted(preview, key=lambda x: (score_article(x), x.published_at), reverse=True)[:10]

        st.subheader(f"미리보기: “{first_kw}” 상위 10건")
        for a in preview:
            st.markdown(
                f"- [{html_unescape(a.title)}]({a.url})  \n  <small>{domain_of(a.url)} • {a.published_at[:16].replace('T',' ')}</small>",
                unsafe_allow_html=True
            )
    except Exception as e:
        st.warning(f"미리보기 로드 실패: {e}")
else:
    st.info("config.json의 KEYWORDS가 비어 있습니다.")
