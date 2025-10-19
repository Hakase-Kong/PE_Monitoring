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
    # NEW: OpenAI
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
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

# ===== [Scheduler Helpers] =====
def _map_days_for_cron(days_ui: list[str]) -> str:
    """
    UI에서 받은 ["매일"] 또는 ["월","수","금"] 을 APScheduler 'cron' day_of_week 포맷으로 변환
    """
    if not days_ui or "매일" in days_ui:
        return "*"  # 매일
    m = {"월":"mon", "화":"tue", "수":"wed", "목":"thu", "금":"fri", "토":"sat", "일":"sun"}
    mapped = [m[d] for d in days_ui if d in m]
    return ",".join(mapped) if mapped else "*"

def ensure_cron_job(sched: BackgroundScheduler, cfg: dict):
    job_id = "pe_news_job"
    try:
        sched.remove_job(job_id)
    except Exception:
        pass
    day_of_week = _map_days_for_cron(cfg.get("CRON_DAYS_UI", ["매일"]))
    hour = int(cfg.get("CRON_HOUR", 9))
    minute = int(cfg.get("CRON_MINUTE", 0))
    sched.add_job(
        scheduled_job, "cron",
        id=job_id, replace_existing=True,
        day_of_week=day_of_week, hour=hour, minute=minute
    )

def ensure_interval_job(sched: BackgroundScheduler, minutes: int):
    job_id = "pe_news_job"
    try:
        sched.remove_job(job_id)
    except Exception:
        pass
    sched.add_job(
        scheduled_job, "interval",
        id=job_id, replace_existing=True,
        minutes=int(minutes),
        next_run_time=now_kst()    # 시작 즉시 1회 실행
    )

def ensure_scheduled_job(sched: BackgroundScheduler, cfg: dict):
    """
    cfg["SCHEDULE_MODE"]:
      - '주기(분)': interval 스케줄, 시작 즉시 1회 전송
      - '요일/시각(주간/매일)': cron 스케줄, 시작 즉시 전송하지 않음
    """
    mode = cfg.get("SCHEDULE_MODE", "주기(분)")
    if mode == "주기(분)":
        ensure_interval_job(sched, int(cfg.get("INTERVAL_MIN", 60)))
    else:
        ensure_cron_job(sched, cfg)

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

# ===== [NEW] Story Key & Enhanced Cache (v2) =====
def story_key(item: dict, cfg: dict | None = None) -> str:
    """
    회차(시간) 간에도 동일 이슈를 1회만 전송하기 위한 키.
    - Naver: naver:OID:AID
    - 기타: normalize_title 기반
    """
    url = item.get("url", "")
    cid = canonical_url_id(url)
    if cid.startswith("naver:"):
        return cid
    norm_t = normalize_title(item.get("title", ""), cfg)
    return f"title:{sha1(norm_t)}"

def _utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _parse_iso(s: str) -> dt.datetime:
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except Exception:
        return dt.datetime.now(dt.timezone.utc)

def load_sent_cache_v2(retention_hours: int = 72) -> (dict, dict):
    """
    캐시파일 구조(신규): {"url": {"<hash>": "iso"}, "story": {"<keyhash>":"iso"}}
    구버전(list 또는 단순 list[str])도 호환.
    반환: (url_dict, story_dict)
    """
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        raw = {}

    now = dt.datetime.now(dt.timezone.utc)
    limit = now - dt.timedelta(hours=max(6, retention_hours))

    url_map = {}
    story_map = {}

    # 구버전(list) 호환: URL 해시만 존재
    if isinstance(raw, list):
        for h in raw:
            url_map[h] = _utcnow_iso()
    elif isinstance(raw, dict):
        url_map = dict(raw.get("url", {}))
        story_map = dict(raw.get("story", {}))

    # 만료 정리
    def _prune(d: dict) -> dict:
        out = {}
        for k, ts in d.items():
            try:
                ts_dt = _parse_iso(ts)
            except Exception:
                continue
            if ts_dt >= limit:
                out[k] = ts
        return out

    return _prune(url_map), _prune(story_map)

def save_sent_cache_v2(url_map: dict, story_map: dict) -> None:
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"url": url_map, "story": story_map}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("전송 캐시 저장 실패(v2): %s", e)

def search_naver_news(keyword: str, client_id: str, client_secret: str, recency_hours=72, page_size: int = 30) -> List[dict]:
    if not client_id or not client_secret or not keyword:
        return []
    base = "https://openapi.naver.com/v1/search/news.json"
    params = {"query": keyword, "display": clamp(int(page_size), 10, 100), "sort": "date"}
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
        "pageSize": clamp(int(cfg.get("PAGE_SIZE", 30)), 10, 100),
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


# fallback 사전(혹시 config에 SYNONYM_MAP이 없을 때만 사용)
_FALLBACK_SYNONYM_MAP = {
    "mergers & acquisitions": "m&a",
    "merger": "m&a",
    "acquisition": "인수",
    "tender offer": "공개매수",
    "takeover": "인수",
    "sell-down": "지분매각",
    "spin-off": "스핀오프",
    "carve-out": "카브아웃",
    "imm인베스트먼트": "imm인베",
    "imm investment": "imm인베",
    "private equity": "pe",
    "사모펀드": "pe",
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

def normalize_title(t: str, cfg: dict | None = None) -> str:
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
# ▶ config의 SYNONYM_MAP 우선 적용, 없으면 fallback
    try:
        synonyms = (cfg or {}).get("SYNONYM_MAP") if isinstance(cfg, dict) else None
    except Exception:
        synonyms = None
    if not isinstance(synonyms, dict) or not synonyms:
        synonyms = _FALLBACK_SYNONYM_MAP
    for k, v in (synonyms or {}).items():
        try:
            s_low = s_low.replace(k, v)
        except Exception:
            pass

    # 숫자 콤마 정규화
    s_low = re.sub(r"\b(\d{1,3}(,\d{3})+|\d+)\b", lambda m: m.group(0).replace(",", ""), s_low)
    # 공백 정리
    s_low = MULTISPACE_RE.sub(" ", s_low).strip()
    return s_low

def _tokens(s: str) -> set:
    return {w for w in re.split(r"[^0-9a-zA-Z가-힣]+", s) if len(w) >= 2}

def _bigrams(s: str) -> set:
    return {s[i:i+2] for i in range(len(s)-1)} if len(s) > 1 else set()

def is_near_dup(a: str, b: str, time_a=None, time_b=None, src_a=None, src_b=None) -> bool:
    """정규화 제목 a,b 근접 중복 판단 (출처+시간 보강)."""
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
    # ✅ 추가: 동일 출처+12시간 이내 기사 → 동일 이슈 간주
    try:
        if src_a and src_b and src_a == src_b and time_a and time_b:
            tdiff = abs((time_a - time_b).total_seconds()) / 3600.0
            if tdiff <= 12 and j_bg >= 0.45:
                return True
    except Exception:
        pass
    return False

def _jaccard_bigrams(s: str) -> float:
    a = _bigrams(s); 
    return 0.0 if not a else len(a)

def _sim_norm_title(a: str, b: str) -> float:
    # 토큰/바이그램 혼합 유사도 (0~1)
    ta, tb = _tokens(a), _tokens(b)
    ja = len(ta & tb) / max(1, len(ta | tb)) if ta and tb else 0.0
    ba, bb = _bigrams(a), _bigrams(b)
    jb = len(ba & bb) / max(1, len(ba | bb)) if ba and bb else 0.0
    # 토큰 0.6, 바이그램 0.4 가중
    return 0.6 * ja + 0.4 * jb

def dedup(items: List[dict], cfg: dict | None = None) -> List[dict]:
    """
    점수 높은 순으로 정렬 후, 동일 이슈(제목 유사 + 동일 출처 + 12h)에 속하면 제거.
    """
    def _ts_kst(it):
        try:
            return dt.datetime.strptime(it["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")\
                     .replace(tzinfo=dt.timezone.utc).astimezone(APP_TZ)
        except Exception:
            return now_kst()

    cfg = cfg or {}
    xs_th = float(cfg.get("TITLE_SIM_XSRC", 0.56))
    ss_th = float(cfg.get("TITLE_SIM_SAMESRC", 0.52))
    same_src_hours = int(cfg.get("SAME_SOURCE_WINDOW_HOURS", 12))

    work = sorted(items, key=lambda x: x.get("_score", 0.0), reverse=True)
    out, seen = [], []  # seen: dict(t_norm, src, ts)

    for it in work:
        t_norm = normalize_title(it.get("title", ""), cfg)
        src = domain_of(it.get("url", ""))
        ts = _ts_kst(it)
        is_dup = False

        for s in seen:
            # 동일 출처 & 12시간 이내 & 제목유사도 0.58↑ → 동일 이슈 간주
            if s["src"] == src and abs((ts - s["ts"]).total_seconds()) <= same_src_hours*3600:
                if _sim_norm_title(t_norm, s["t_norm"]) >= ss_th:
                    is_dup = True
                    break
            # 출처 달라도 제목이 거의 동일(0.68↑)이면 중복
            if _sim_norm_title(t_norm, s["t_norm"]) >= 0.68:
                is_dup = True
                break

        if not is_dup:
            out.append(it)
            seen.append({"t_norm": t_norm, "src": src, "ts": ts})

    return out

# -------------------------
# 규칙 기반 필터/정렬
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

    # 네이버 섹션 제한
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

    # -------------------------------
    # 🔽 여기부터 수정/추가 부분
    # -------------------------------
    # PEF 맥락 필수 조건을 기본 적용하되,
    # '신뢰 도메인 + 모호하지만 중요한 토큰(매각/공개매각/인수 등)'이면 LLM으로 넘기도록 우회 허용
    context_any = cfg.get("CONTEXT_REQUIRE_ANY", []) or []
    context = (title + " " + item.get("description", "")).lower()

    has_context = any(k.lower() in context for k in context_any)

    trusted = set(cfg.get("TRUSTED_SOURCES_FOR_FI", cfg.get("ALLOW_DOMAINS", [])) or [])
    amb_tokens = set(t.lower() for t in (cfg.get("STRICT_AMBIGUOUS_TOKENS", []) or []))
    has_ambiguous = any(tok in title.lower() for tok in amb_tokens)

    # 맥락 단어가 없다면 → (신뢰 도메인 AND 모호토큰)일 때만 통과시켜 LLM에서 판단
    if not has_context:
        if not (src in trusted and has_ambiguous):
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
    return dedup(arr, cfg)

# -------------------------
# LLM 기반 2차 필터
# -------------------------
def _flatten_aliases(cfg: dict) -> List[str]:
    out = []
    for k, v in (cfg.get("KEYWORD_ALIASES", {}) or {}).items():
        out.append(k)
        out.extend(v or [])
    return sorted(set(out))

def _llm_prompt_for_item(item: dict, cfg: dict) -> str:
    kw = cfg.get("KEYWORDS", []) or []
    aliases = _flatten_aliases(cfg)
    firms = cfg.get("FIRM_WATCHLIST", []) or []
    context_any = cfg.get("CONTEXT_REQUIRE_ANY", []) or []
    return f"""
당신은 '국내 PE 동향' 관련 기사를 분류하는 전문가입니다.
다음 기사가 사모펀드(PEF), GP/LP, 재무적 투자자(FI)의 투자·인수·매각·리파이낸싱 활동과 관련이 있거나,
그들이 관여할 가능성이 높은 거래(M&A, 매각, 대형 자금조달, 공개매수)인지 판단하세요.
단순 산업 내 전략적 인수나 일반 기업 인사·운영 보도는 제외합니다.

판단 기준:
- 핵심 키워드: {', '.join(kw)}
- 동의어: {', '.join(aliases)}
- 운용사/FI 워치리스트: {', '.join(firms)}
- 맥락 키워드(있으면 강한 근거): {', '.join(context_any)}

출력은 반드시 JSON 한 줄로 반환:
{{
  "relevant": true|false,
  "confidence": 0.0~1.0,
  "category": "PE deal"|"finance general"|"industry M&A"|"irrelevant",
  "matched": ["매칭된 단어들"],
  "reason": "한 줄 근거"
}}

기사:
- 제목: {item.get('title','')}
- 출처: {domain_of(item.get('url',''))}
- 링크: {item.get('url','')}
"""

def _openai_chat(messages: List[Dict], api_key: str, model: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    r = requests.post(url, headers=headers, json=payload, timeout=25)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def llm_filter_items(items: List[dict], cfg: dict, env: dict) -> List[dict]:
    if not items:
        return items
    if not bool(cfg.get("USE_LLM_FILTER", False)):
        return items

    api_key = env.get("OPENAI_API_KEY", "")
    if not api_key:
        log.warning("LLM 필터 활성화되어 있으나 OPENAI_API_KEY 미설정 → 규칙기반 결과 사용")
        return items

    model = cfg.get("LLM_MODEL", "gpt-4o-mini")
    conf_th = float(cfg.get("LLM_CONF_THRESHOLD", 0.7))  # ✅ 완화
    out = []

    for it in items:
        try:
            user_prompt = _llm_prompt_for_item(it, cfg)
            messages = [
                {"role": "system", "content":
                    "You are a professional news classifier for Private Equity (KR). "
                    "Classify only deals/events where a financial investor (PEF/GP/LP, co-invest, secondary, structured/NAV/pref-equity, PIPE, mezzanine, recap/refi) "
                    "is involved or is plausibly involved. "
                    "If a strategic investor (SI) conducts a pure industry M&A without FI involvement, mark relevant=false (category='industry M&A'). "
                    "Return JSON only."
                },
                {"role": "user", "content": user_prompt},
            ]
            resp = _openai_chat(messages, api_key, model, max_tokens=int(cfg.get("LLM_MAX_TOKENS", 400)))
            j = None
            try:
                j = json.loads(resp.strip())
            except Exception:
                import re as _re
                m = _re.search(r"\{[\s\S]*\}$", resp.strip())
                if m:
                    j = json.loads(m.group(0))

            # ✅ 완화된 조건: PE deal or finance general 둘 다 허용
            cat = (j or {}).get("category", "").lower()
            if (
                isinstance(j, dict)
                and j.get("relevant") is True
                and float(j.get("confidence", 0.0)) >= conf_th
                and cat in {"pe deal", "finance general"}
            ):
                it["_llm"] = j
                out.append(it)

        except Exception as e:
            log.warning("LLM 필터 처리 실패: %s", e)
            out.append(it)

    return out

# -------------------------
# LLM 기반 2차 중복 제거 (동일 기사/이슈 판정)
# -------------------------
def _llm_is_same_story(a: dict, b: dict, env: dict, cfg: dict) -> Optional[bool]:
    """
    두 기사가 '같은 스토리(동일 기사/동일 이슈 재보도)'인지 LLM으로 판정.
    True: 동일 스토리, False: 다른 스토리, None: 판정 실패(보류)
    """
    api_key = env.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    def _fmt(it: dict) -> str:
        t = (it.get("title") or "").strip()
        u = (it.get("url") or "").strip()
        s = (domain_of(u) or it.get("source") or "").strip()
        when = it.get("publishedAt") or ""
        return f"- 제목: {t}\n- 출처: {s}\n- 시각(UTC): {when}\n- 링크: {u}"

    sys = (
        "You are a professional news deduplication judge for Korean finance news.\n"
        "Task: Decide if two news items are about the SAME story (same underlying article/issue), "
        "even if titles differ slightly (follow-ups, minor edits, copy on portal vs. source). "
        "Consider: title meaning, named entities, seller/buyer, price/size, and timing.\n"
        "Answer strictly in JSON: {\"same\": true|false, \"reason\": \"...\"}"
    )
    usr = "기사 A\n" + _fmt(a) + "\n\n기사 B\n" + _fmt(b) + "\n\n판정:"

    try:
        resp = _openai_chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            api_key,
            cfg.get("LLM_MODEL", "gpt-4o-mini"),
            max_tokens=200,
            temperature=0.0,
        )
        j = None
        try:
            j = json.loads(resp.strip())
        except Exception:
            m = re.search(r"\{[\s\S]*\}$", resp.strip())
            if m:
                j = json.loads(m.group(0))
        if isinstance(j, dict) and "same" in j:
            return bool(j.get("same"))
    except Exception as e:
        log.warning("LLM dedup 오류: %s", e)
    return None


def _utc_to_kst(ts: str) -> dt.datetime:
    try:
        return dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).astimezone(APP_TZ)
    except Exception:
        return now_kst()


def llm_dedup_items(items: List[dict], cfg: dict, env: dict) -> List[dict]:
    """
    1차 규칙기반 dedup 이후에도 남는 '유사하지만 애매한' 케이스를 LLM으로 다시 한 번 정리.
    - 후보 페어: 제목유사도 0.50~0.72 구간 또는 동일출처±시간창 내 기사
    - 동일 스토리로 판정되면 _score 낮은 쪽을 제거
    """
    if not items or not cfg.get("USE_LLM_DEDUP", False):
        return items
    if not env.get("OPENAI_API_KEY"):
        return items

    win_hours = int(cfg.get("LLM_DEDUP_WINDOW_HOURS", 24))
    keep_mask = [True] * len(items)

    # 사전 계산: 정규화 제목/출처/시각
    meta = []
    for it in items:
        tnorm = normalize_title(it.get("title", ""), cfg)
        src = domain_of(it.get("url", ""))
        ts_kst = _utc_to_kst(it.get("publishedAt", ""))
        sc = float(it.get("_score", 0.0))
        meta.append((tnorm, src, ts_kst, sc))

    n = len(items)
    for i in range(n):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, n):
            if not keep_mask[j]:
                continue

            a_t, a_s, a_ts, a_sc = meta[i]
            b_t, b_s, b_ts, b_sc = meta[j]

            # 시간창 & 출처 조건
            time_ok = abs((a_ts - b_ts).total_seconds()) <= win_hours * 3600
            src_same = (a_s == b_s)

            # 제목 유사도
            sim = _sim_norm_title(a_t, b_t)

            # LLM 판정 후보 조건(너무 명확/너무 다른 것은 제외)
            if (0.50 <= sim < 0.72) or (src_same and time_ok and sim >= 0.45):
                same = _llm_is_same_story(items[i], items[j], env, cfg)
                if same is True:
                    # 낮은 점수 쪽을 제거
                    if a_sc >= b_sc:
                        keep_mask[j] = False
                    else:
                        keep_mask[i] = False
                        break  # i가 제거되었으므로 내부 루프 탈출
                elif same is False:
                    continue
                else:
                    # 판정 실패는 보류
                    continue

    return [it for k, it in enumerate(items) if keep_mask[k]]

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
        batch = search_naver_news(
            kw, env.get("NAVER_CLIENT_ID",""), env.get("NAVER_CLIENT_SECRET",""),
            recency_hours=recency_hours, page_size=int(cfg.get("PAGE_SIZE", 30))
        )
        all_items += batch
    
    # NewsAPI (선택)
    if env.get("NEWSAPI_KEY") and keywords:
        query = " OR ".join(keywords)
        batch = search_newsapi(query, page_size=page_size, api_key=env["NEWSAPI_KEY"], from_hours=recency_hours, cfg=cfg)
        all_items += batch

    return all_items

def format_telegram_text(items: List[dict], cfg: dict = {} ) -> str:
    if not items:
        return "📭 신규 뉴스 없음"
    lines = ["📌 <b>국내 PE 동향 관련 뉴스</b>"]
    for it in items:
        t = it.get("title", "").strip()
        u = it.get("url", "")
        src = domain_of(u)
        try:
            pub = dt.datetime.strptime(it["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).astimezone(APP_TZ)
            when = pub.strftime("%Y-%m-%d %H:%M")
        except Exception:
            when = "-"
        if bool(cfg.get("SHOW_SOURCE_DOMAIN", False)):
            lines.append(f"• <a href=\"{u}\">{t}</a> — {src} ({when})")
        else:
            lines.append(f"• <a href=\"{u}\">{t}</a> ({when})")
    return "\n".join(lines)

def send_telegram(bot_token: str, chat_id: str, text: str, disable_preview: bool = True) -> bool:
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML",
               "disable_web_page_preview": disable_preview}
    try:
        r = requests.post(url, json=payload, timeout=12)
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning("텔레그램 전송 실패: %s", e)
        return False

def _should_skip_by_time(cfg: dict) -> bool:
    kst_now = now_kst()
    if cfg.get("ONLY_WORKING_HOURS") and not between_working_hours(kst_now, 6, 20):  # 08→06
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
    run_lock = get_run_lock()
    if not run_lock.acquire(blocking=False):
        log.info("다른 실행이 진행 중이어서 이번 주기는 스킵합니다.")
        return {"count": 0, "items": []}
    try:
        all_items = collect_all(cfg, env)
        ranked = rank_filtered(all_items, cfg)  # 1차: 규칙 기반 필터
        ranked = pe_focus_filter(ranked, cfg) # PE 선택 필터
        ranked = llm_filter_items(ranked, cfg, env)  # 2차: LLM 필터 (옵션)
        ranked = llm_dedup_items(ranked, cfg, env)   # 3차: LLM 중복판정 (옵션) ← ✅ 여기 추가

        if preview:
            return {"count": len(ranked), "items": ranked}

        if _should_skip_by_time(cfg):
            log.info("시간 정책에 의해 전송 건너뜀 (업무시간/주말/공휴일)")
            return {"count": 0, "items": []}

        retention = int(cfg.get("CACHE_RETENTION_HOURS", cfg.get("RECENCY_HOURS", 72)))
        url_cache, story_cache = load_sent_cache_v2(retention_hours=retention)
        now_iso = _utcnow_iso()

        new_items = []
        for it in ranked:
            uhash = sha1(it.get("url", ""))
            skey  = story_key(it, cfg)
            if (uhash in url_cache) or (skey in story_cache):
                continue
            new_items.append(it)

        if not new_items:
            if bool(cfg.get("NO_NEWS_SILENT", True)):
                log.info("신규 뉴스 없음 (무소식 알림 억제 옵션으로 미전송)")
                return {"count": 0, "items": []}
            else:
                send_telegram(env.get("TELEGRAM_BOT_TOKEN", ""), env.get("TELEGRAM_CHAT_ID", ""), "📭 신규 뉴스 없음")
                return {"count": 0, "items": []}

        BATCH = 30
        sent_any = False
        for i in range(0, len(new_items), BATCH):
            chunk = new_items[i:i+BATCH]
            text = format_telegram_text(chunk, cfg)
            ok = send_telegram(env.get("TELEGRAM_BOT_TOKEN", ""), env.get("TELEGRAM_CHAT_ID", ""), text, disable_preview=bool(cfg.get("TELEGRAM_DISABLE_PREVIEW", True)))
            sent_any = sent_any or ok
            time.sleep(0.6)

        if sent_any:
            for it in new_items:
                url_cache[sha1(it.get("url", ""))] = now_iso
                story_cache[story_key(it, cfg)] = now_iso
            save_sent_cache_v2(url_cache, story_cache)

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

def is_running(_: BackgroundScheduler = None) -> bool:
    try:
        sched = get_scheduler()
        return any(j.id == "pe_news_job" for j in sched.get_jobs())
    except Exception:
        return False

# 스케줄 시작/중지(버튼 핸들러용) — 여기서만 global 사용
def start_schedule(cfg_path: str, cfg_dict: dict, env: dict, minutes: int):
    """
    버튼 클릭 시 스케줄 시작.
    - 주기(분) 모드: 스케줄 등록 + 즉시 1회 전송
    - 요일/시각 모드: 스케줄 등록만 (즉시 전송하지 않음)
    """
    global CURRENT_CFG_PATH, CURRENT_CFG_DICT, CURRENT_ENV
    CURRENT_CFG_PATH = cfg_path
    CURRENT_CFG_DICT = dict(cfg_dict)
    CURRENT_ENV = env

    sched = get_scheduler()
    ensure_scheduled_job(sched, CURRENT_CFG_DICT)

def stop_schedule():
    sched = get_scheduler()
    try:
        sched.remove_job("pe_news_job")
    except Exception:
        pass

# ===== [Filter] 특정 PE 포커스 =====
def pe_focus_filter(items: list[dict], cfg: dict) -> list[dict]:
    """
    cfg['PE_FOCUS']가 비어 있지 않으면, 제목/요약/본문에
    해당 키워드(PE명)가 하나라도 포함된 기사만 통과시킵니다.
    items: 각 원소는 {"title","summary","content","url",...}
    """
    focus = [s.strip() for s in cfg.get("PE_FOCUS", []) if isinstance(s, str) and s.strip()]
    if not focus:
        return items

    def _hit(text: str) -> bool:
        if not text:
            return False
        low = text.lower()
        for kw in focus:
            if kw.lower() in low:
                return True
        return False

    out = []
    for it in items:
        text = f"{it.get('title','')} {it.get('summary','')} {it.get('content','')}"
        if _hit(text):
            out.append(it)
    return out

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
# NEW: OpenAI
openai_key = st.sidebar.text_input("OpenAI API Key (선택)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
bot_token = st.sidebar.text_input("Telegram Bot Token", type="password", value=os.getenv("TELEGRAM_BOT_TOKEN", ""))
chat_id = st.sidebar.text_input("Telegram Chat ID (채널/그룹)", value=os.getenv("TELEGRAM_CHAT_ID", ""))

# 파라미터
st.sidebar.divider()
st.sidebar.subheader("전송/수집 파라미터")
cfg["PAGE_SIZE"] = int(st.sidebar.number_input("페이지당 수집 수", min_value=10, max_value=100, step=1, value=int(cfg.get("PAGE_SIZE", 30))))
cfg["RECENCY_HOURS"] = int(st.sidebar.number_input("신선도(최근 N시간)", min_value=6, max_value=168, step=6, value=int(cfg.get("RECENCY_HOURS", 72))))

# ✅ 시간 정책 토글
st.sidebar.subheader("시간 정책")
cfg["ONLY_WORKING_HOURS"] = bool(st.sidebar.checkbox("✅ 업무시간(06~20 KST) 내 전송", value=bool(cfg.get("ONLY_WORKING_HOURS", True))))
cfg["BLOCK_WEEKEND"]     = bool(st.sidebar.checkbox("🚫 주말 미전송", value=bool(cfg.get("BLOCK_WEEKEND", False))))
cfg["BLOCK_HOLIDAY"]     = bool(st.sidebar.checkbox("🚫 공휴일 미전송", value=bool(cfg.get("BLOCK_HOLIDAY", False))))
holidays_text = st.sidebar.text_area("공휴일(YYYY-MM-DD, 쉼표 또는 줄바꿈 구분)", value=", ".join(cfg.get("HOLIDAYS", [])))
cfg["HOLIDAYS"] = [s.strip() for s in re.split(r"[,\n]", holidays_text) if s.strip()]

st.sidebar.subheader("전송 스케줄")

cfg["SCHEDULE_MODE"] = st.sidebar.radio("스케줄 방식", options=["주기(분)", "요일/시각(주간/매일)"], index=0 if cfg.get("SCHEDULE_MODE", "주기(분)") == "주기(분)" else 1, horizontal=False)
if cfg["SCHEDULE_MODE"] == "주기(분)":
    cfg["INTERVAL_MIN"] = int(st.sidebar.number_input(
        "전송 주기(분)",
        min_value=5, max_value=10080, step=5,
        value=int(cfg.get("INTERVAL_MIN", 60))
    ))
    st.sidebar.caption("최대 1주일(=10080분)까지 설정 가능. 스케줄 시작 시 즉시 1회 전송 후 주기적으로 전송합니다.")

# (신규) 요일/시각 스케줄: 주 1회 또는 매일
else:
    # 선택 가능한 요일
    WEEKDAY_LABELS = ["매일", "월", "화", "수", "목", "금", "토", "일"]
    # 기본 선택값
    default_days = cfg.get("CRON_DAYS_UI", ["매일"])
    # UI: 다중 선택
    selected_days = st.sidebar.multiselect(
        "전송 요일 선택 (매일을 선택하면 다른 요일은 무시됩니다)",
        options=WEEKDAY_LABELS,
        default=default_days
    )
    # 결과 저장 (UI 표기 그대로도 보존)
    cfg["CRON_DAYS_UI"] = selected_days[:] if selected_days else ["매일"]

    # 시간 선택
    t = st.sidebar.time_input(
        "전송 시각 (KST)",
        value=dt.time(hour=int(cfg.get("CRON_HOUR", 9)), minute=int(cfg.get("CRON_MINUTE", 0)))
    )
    cfg["CRON_HOUR"] = int(t.hour)
    cfg["CRON_MINUTE"] = int(t.minute)

    st.sidebar.caption("요일/시각 모드에서는 '스케줄 시작'을 눌러도 즉시 전송하지 않고 지정 시각에만 전송합니다.")

# 기타 필터 토글
st.sidebar.subheader("기타 필터")
cfg["ALLOWLIST_STRICT"] = bool(st.sidebar.checkbox("🧱 ALLOWLIST_STRICT (허용 도메인 외 차단)", value=bool(cfg.get("ALLOWLIST_STRICT", False))))

# NEW: LLM 필터 옵션
st.sidebar.subheader("LLM 필터(선택)")
cfg["USE_LLM_FILTER"] = bool(st.sidebar.checkbox("🤖 OpenAI로 2차 필터링", value=bool(cfg.get("USE_LLM_FILTER", False))))
cfg["LLM_MODEL"] = st.sidebar.text_input("모델", value=cfg.get("LLM_MODEL", "gpt-4o-mini"))
cfg["LLM_CONF_THRESHOLD"] = float(st.sidebar.slider("채택 임계치(신뢰도)", min_value=0.0, max_value=1.0, value=float(cfg.get("LLM_CONF_THRESHOLD", 0.7)), step=0.05))
cfg["LLM_MAX_TOKENS"] = int(st.sidebar.number_input("max_tokens", min_value=64, max_value=1000, step=10, value=int(cfg.get("LLM_MAX_TOKENS", 300))))

# ===== [UI] 특정 PE 선택 =====
st.sidebar.subheader("🎯 특정 PE 선택(선택)")
# 후보 목록은 config.json의 "PE_CANDIDATES"를 우선 사용, 없으면 아래 기본
pe_candidates_default = [
    "MBK", "IMM", "Hahn&Co", "VIG", "글랜우드", "베인", "KKR", "Carlyle", "한앤코",
    "스틱", "H&Q", "브릿지", "JIP", "Affinity", "TPG", "KCGI", "한화", "맥쿼리"
]
pe_candidates = cfg.get("PE_CANDIDATES", pe_candidates_default)
cfg["PE_FOCUS"] = st.sidebar.multiselect(
    "특정 PE만 선별(비워두면 전체)",
    options=pe_candidates,
    default=cfg.get("PE_FOCUS", [])
)

st.sidebar.divider()
if st.sidebar.button("구성 리로드", use_container_width=True):
    st.rerun()

st.title("📰 국내 PE 동향 뉴스 자동 모니터링")
st.caption("Streamlit + Naver/NewsAPI + OpenAI Filter + Telegram + APScheduler (Render + UptimeRobot)")

def make_env() -> dict:
    return {
        "NAVER_CLIENT_ID": naver_id,
        "NAVER_CLIENT_SECRET": naver_secret,
        "NEWSAPI_KEY": newsapi_key,
        "OPENAI_API_KEY": openai_key,
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
        if cfg.get("SCHEDULE_MODE","주기(분)") == "주기(분)":
            st.success("스케줄 시작됨: 즉시 1회 전송 후 주기적으로 실행")
        else:
            st.success("스케줄 시작됨: 지정된 요일/시각에만 전송(시작 즉시 전송 없음)")
        st.rerun()

with col4:
    if st.button("스케줄 중지"):
        stop_schedule()
        st.warning("스케줄 중지됨")
        st.rerun()

# 상태
_running = is_running()
st.subheader("상태")
sched = get_scheduler()
jobs = []
try:
    jobs = sched.get_jobs()
except Exception:
    jobs = []
st.info(f"Scheduler 실행 중: {_running}")
for j in jobs:
    st.caption(f"• Job: {j.id} / 다음 실행: {j.next_run_time}")

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
        meta = it.get("_llm")
        if meta:
            _m = ", ".join([str(x) for x in (meta.get("matched") or [])][:6])
            st.markdown(
                f"- <a href='{u}'>{t}</a> ({when})  "
                f"<span style='color:gray'>LLM: {meta.get('confidence',0):.2f}"
                + (f" · {_m}" if _m else "")
                + "</span>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"- <a href='{u}'>{t}</a> ({when})", unsafe_allow_html=True)
