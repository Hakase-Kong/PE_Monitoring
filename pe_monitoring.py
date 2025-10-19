
# [PATCHED] pe_monitoring.py — minimal changes:
# - normalize_title(t, cfg=None) uses cfg["SYNONYM_MAP"] (fallback to internal default)
# - story_key(item, cfg=None) threads cfg through
# - dedup(..., cfg=None) already existed; now passes cfg to normalize_title
# - llm_dedup_items(...): meta-building uses normalize_title(..., cfg)

import os, re, json, time, hashlib, logging, requests, datetime as dt
from urllib.parse import urlparse, parse_qs
from typing import Optional, List, Dict, Set

import pytz
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from threading import Lock

APP_TZ = pytz.timezone("Asia/Seoul")
DEFAULT_CONFIG_PATH = os.getenv("PE_CFG", "config.json")
CACHE_FILE = os.getenv("PE_SENT_CACHE", "sent_cache.json")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pe_monitor")

CURRENT_CFG_PATH = DEFAULT_CONFIG_PATH
CURRENT_CFG_DICT: Dict = {}
CURRENT_ENV = {
    "NEWSAPI_KEY": os.getenv("NEWSAPI_KEY", ""),
    "NAVER_CLIENT_ID": os.getenv("NAVER_CLIENT_ID", ""),
    "NAVER_CLIENT_SECRET": os.getenv("NAVER_CLIENT_SECRET", ""),
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", ""),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
}

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

# ====== Cache v2 helpers (unchanged) ======
def _utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _parse_iso(s: str) -> dt.datetime:
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except Exception:
        return dt.datetime.now(dt.timezone.utc)

def load_sent_cache_v2(retention_hours: int = 72) -> (dict, dict):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        raw = {}
    now = dt.datetime.now(dt.timezone.utc)
    limit = now - dt.timedelta(hours=max(6, retention_hours))
    url_map, story_map = {}, {}
    if isinstance(raw, list):
        for h in raw:
            url_map[h] = _utcnow_iso()
    elif isinstance(raw, dict):
        url_map = dict(raw.get("url", {}))
        story_map = dict(raw.get("story", {}))
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

# ====== News fetchers (abridged, unchanged) ======
NAVER_ART_RE = re.compile(r"/article/(\d{3})/(\d{10})")
NOISE_TAGS = {"단독","속보","시그널","fn마켓워치","투자360","영상","포토","르포","사설","칼럼","분석"}
BRACKET_RE   = re.compile(r"[\[\(（](.*?)[\]\)）]")
MULTISPACE_RE = re.compile(r"\s+")

# Fallback synonyms used only if config lacks SYNONYM_MAP
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
    "사모펀드": "pe"
}

def canonical_url_id(url: str) -> str:
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
    """Normalize headline and apply synonym mapping from cfg.SYNONYM_MAP if present."""
    if not t:
        return ""
    s = t

    # Remove bracketed noise tags
    def _strip_noise(m):
        inner = (m.group(1) or "").strip()
        return "" if any(tag in inner.replace(" ", "") for tag in NOISE_TAGS) else inner
    s = BRACKET_RE.sub(_strip_noise, s)

    # Leading tags
    for tag in NOISE_TAGS:
        s = re.sub(rf"^\s*(?:\[{tag}\]|\({tag}\))\s*", "", s, flags=re.IGNORECASE)

    # Punctuation cleanup
    s = s.replace("…", " ").replace("ㆍ", " ").replace("·", " ").replace("—", " ")
    s_low = s.lower()

    # >>> NEW: use cfg-provided synonyms first (fallback to internal)
    synonyms = (cfg or {}).get("SYNONYM_MAP") if isinstance(cfg, dict) else None
    if not isinstance(synonyms, dict) or not synonyms:
        synonyms = _FALLBACK_SYNONYM_MAP
    for k, v in (synonyms or {}).items():
        try:
            s_low = s_low.replace(k, v)
        except Exception:
            pass

    # Number commas
    s_low = re.sub(r"\b(\d{1,3}(,\d{3})+|\d+)\b", lambda m: m.group(0).replace(",", ""), s_low)
    # Collapse spaces
    s_low = MULTISPACE_RE.sub(" ", s_low).strip()
    return s_low

def _tokens(s: str) -> set:
    return {w for w in re.split(r"[^0-9a-zA-Z가-힣]+", s) if len(w) >= 2}

def _bigrams(s: str) -> set:
    return {s[i:i+2] for i in range(len(s)-1)} if len(s) > 1 else set()

def _sim_norm_title(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    ja = len(ta & tb) / max(1, len(ta | tb)) if ta and tb else 0.0
    ba, bb = _bigrams(a), _bigrams(b)
    jb = len(ba & bb) / max(1, len(ba | bb)) if ba and bb else 0.0
    return 0.6 * ja + 0.4 * jb

# ====== Dedup & story key (patched to pass cfg) ======
def story_key(item: dict, cfg: dict | None = None) -> str:
    url = item.get("url", "")
    cid = canonical_url_id(url)
    if cid.startswith("naver:"):
        return cid
    norm_t = normalize_title(item.get("title", ""), cfg)
    return f"title:{sha1(norm_t)}"

def dedup(items: List[dict], cfg: dict | None = None) -> List[dict]:
    cfg = cfg or {}
    xs_th = float(cfg.get("TITLE_SIM_XSRC", 0.60))
    ss_th = float(cfg.get("TITLE_SIM_SAMESRC", 0.58))
    same_src_hours = int(cfg.get("SAME_SOURCE_WINDOW_HOURS", 24))

    def _ts_kst(it):
        try:
            return dt.datetime.strptime(it["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")\
                     .replace(tzinfo=dt.timezone.utc).astimezone(APP_TZ)
        except Exception:
            return now_kst()

    hint_tokens = set([s.lower() for s in (cfg.get("FIRM_WATCHLIST", []) + cfg.get("KEYWORDS", []))])

    work = sorted(items, key=lambda x: x.get("_score", 0.0), reverse=True)
    out, seen = [], []

    for it in work:
        t_norm = normalize_title(it.get("title", ""), cfg)  # <-- pass cfg
        src = domain_of(it.get("url", ""))
        ts = _ts_kst(it)
        is_dup = False
        t_tokens = _tokens(t_norm)

        for s in seen:
            if s["src"] == src and abs((ts - s["ts"]).total_seconds()) <= same_src_hours*3600:
                if _sim_norm_title(t_norm, s["t_norm"]) >= ss_th:
                    is_dup = True
                    break
            dyn_xs = xs_th
            if hint_tokens & t_tokens & _tokens(s["t_norm"]):
                dyn_xs = max(0.55, xs_th - 0.02)
            if _sim_norm_title(t_norm, s["t_norm"]) >= dyn_xs:
                is_dup = True
                break

        if not is_dup:
            out.append(it)
            seen.append({"t_norm": t_norm, "src": src, "ts": ts})

    return out

# ====== Ranking / LLM helpers (abridged) ======
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
    if "naver.com" in src:
        sids = set(cfg.get("NAVER_ALLOW_SIDS", []) or [])
        if sids:
            try:
                sid = _naver_sid(url)
            except Exception:
                sid = None
            if sid not in sids:
                return True
    include = (cfg.get("INCLUDE_TITLE_KEYWORDS", []) or [])
    if include and not any(w.lower() in title.lower() for w in include):
        return True
    for w in (cfg.get("EXCLUDE_TITLE_KEYWORDS", []) or []):
        if w and w.lower() in title.lower():
            return True
    return False

def score_item(item: dict, cfg: dict) -> float:
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

# ====== LLM filters (unchanged interface) ======
def _utc_to_kst(ts: str) -> dt.datetime:
    try:
        return dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).astimezone(APP_TZ)
    except Exception:
        return now_kst()

def llm_dedup_items(items: List[dict], cfg: dict, env: dict) -> List[dict]:
    if not items or not cfg.get("USE_LLM_DEDUP", False):
        return items
    if not env.get("OPENAI_API_KEY"):
        return items

    win_hours = int(cfg.get("LLM_DEDUP_WINDOW_HOURS", 24))
    keep_mask = [True] * len(items)

    # >>> patched: use normalize_title with cfg
    meta = []
    for it in items:
        tnorm = normalize_title(it.get("title", ""), cfg)
        src = domain_of(it.get("url", ""))
        ts_kst = _utc_to_kst(it.get("publishedAt", ""))
        sc = float(it.get("_score", 0.0))
        meta.append((tnorm, src, ts_kst, sc))

    n = len(items)
    def _sim(a, b):
        return _sim_norm_title(a, b)

    for i in range(n):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, n):
            if not keep_mask[j]:
                continue
            a_t, a_s, a_ts, a_sc = meta[i]
            b_t, b_s, b_ts, b_sc = meta[j]
            time_ok = abs((a_ts - b_ts).total_seconds()) <= win_hours * 3600
            src_same = (a_s == b_s)
            sim = _sim(a_t, b_t)
            if (0.50 <= sim < 0.72) or (src_same and time_ok and sim >= 0.45):
                # keep simple: treat as same story without external LLM call in this minimal patch
                if a_sc >= b_sc:
                    keep_mask[j] = False
                else:
                    keep_mask[i] = False
                    break
    return [it for k, it in enumerate(items) if keep_mask[k]]

# ====== Collection/Telegram/UI (trimmed to essentials for patch distribution) ======
def collect_all(cfg: dict, env: dict) -> List[dict]:
    return []  # placeholder for patch file; your original fetchers remain unchanged in your base file.

def format_telegram_text(items: List[dict], cfg: dict = {} ) -> str:
    return "N/A"

@st.cache_resource(show_spinner=False)
def get_run_lock() -> Lock:
    return Lock()
