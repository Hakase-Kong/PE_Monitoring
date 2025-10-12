# -*- coding: utf-8 -*-
import json, re, os, time, logging, hashlib, threading, datetime as dt
from typing import List, Dict, Optional
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode, urlsplit, urlunsplit
import requests

# ==============================
# Configuration & Constants
# ==============================

APP_NAME = "PE Monitoring Bot"
CACHE_FILE = "/mnt/data/sent_cache.json"
RUN_LOCK = threading.Lock()

# 노이즈 태그 (머리말/꼬리말 제거용)
NOISE_TAGS = {
    "단독","속보","시그널","fn마켓워치","투자360","영상","포토","르포","사설","칼럼","분석",
    "마켓인","PE는 지금"
}

# ==============================
# Utilities
# ==============================

def now_kst() -> dt.datetime:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_config(path: str="/mnt/data/config.json") -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

def get_env_from_cfg(cfg: Dict) -> Dict:
    env = {
        "NAVER_CLIENT_ID": cfg.get("NAVER_CLIENT_ID", ""),
        "NAVER_CLIENT_SECRET": cfg.get("NAVER_CLIENT_SECRET", ""),
        "NEWSAPI_KEY": cfg.get("NEWSAPI_KEY", ""),
        "TELEGRAM_BOT_TOKEN": cfg.get("TELEGRAM_BOT_TOKEN", ""),
        "TELEGRAM_CHAT_ID": cfg.get("TELEGRAM_CHAT_ID", ""),
    }
    return env

def get_run_lock() -> threading.Lock:
    return RUN_LOCK

# ==============================
# Title normalization & URL canonicalization
# ==============================

RE_BRACKET = re.compile(r"^[\[\(【〈<]\s*([^)\]】〉>]+)\s*[\)\]】〉>]\s*")
RE_MULTI_WS = re.compile(r"\s+")

def normalize_title(title: str) -> str:
    """머리말 꼬리표/노이즈 제거 + 소문자화 + 공백정리"""
    if not title:
        return ""
    t = title.strip()

    # 앞쪽 괄호/대괄호 태그 반복 제거
    changed = True
    while changed:
        changed = False
        m = RE_BRACKET.match(t)
        if m:
            tag = m.group(1).strip()
            if tag.replace(" ", "") in {x.replace(" ", "") for x in NOISE_TAGS}:
                t = t[m.end():].lstrip()
                changed = True

    # 중간에 포함된 노이즈 대괄호도 제거
    for tag in NOISE_TAGS:
        t = re.sub(rf"\s*[\[\(【〈<]\s*{re.escape(tag)}\s*[\)\]】〉>]\s*", " ", t, flags=re.IGNORECASE)

    # 특수기호 과다 제거
    t = re.sub(r"[“”\"'‘’·•…▶▷▲▼■□◆◇※★☆❖❗❓]", " ", t)
    t = RE_MULTI_WS.sub(" ", t).strip().lower()
    return t

def canonical_url_id(url: str) -> Optional[str]:
    """
    매체별 '내용 동일성'을 최대한 보장할 수 있는 정규화된 id 추출
    - Naver News: oid/aid
    - 일반 URL: netloc + path + 정렬된 쿼리 key 중 일부
    """
    if not url:
        return None
    try:
        u = urlsplit(url)
        host = (u.netloc or "").lower()

        # Naver News 모바일/데스크톱 공통 처리
        if "naver.com" in host and "/mnews" in u.path or "/news" in u.path:
            qs = parse_qs(u.query)
            oid = qs.get("oid", [None])[0]
            aid = qs.get("aid", [None])[0]
            if not (oid and aid):
                # mnews 형태: .../article/{oid}/{aid}
                m = re.search(r"/article/(\d+)/(\d+)", u.path)
                if m:
                    oid, aid = m.group(1), m.group(2)
            if oid and aid:
                return f"naver:{oid}:{aid}"

        # 기타: host + path (query는 정렬한 주요 key만 포함)
        base = f"{host}{u.path}"
        qs = parse_qs(u.query)
        # 흔한 식별자 키들만 골라 정렬
        keys = ["id","aid","oid","articleId","docid"]
        picked = {k: qs[k] for k in keys if k in qs}
        if picked:
            return f"{base}?{urlencode(sorted([(k, v[0]) for k, v in picked.items()]))}"
        return base
    except Exception:
        return None

def _bigrams(s: str):
    s = s.replace(" ", "")
    return set([s[i:i+2] for i in range(len(s)-1)]) if len(s) >= 2 else set()

def is_near_dup(a: str, b: str) -> bool:
    """제목 근사중복 판단: 토큰 Jaccard와 바이그램 Jaccard를 함께 사용"""
    if not a or not b:
        return False
    atoks = set(a.split())
    btoks = set(b.split())
    if atoks and btoks:
        jacc = len(atoks & btoks) / max(1, len(atoks | btoks))
        if jacc >= 0.70:
            return True
    ab = _bigrams(a)
    bb = _bigrams(b)
    if ab and bb:
        bj = len(ab & bb) / max(1, len(ab | bb))
        if bj >= 0.55:
            return True
    return False

# ==============================
# Cache (with migration & pruning)
# ==============================

def load_sent_cache() -> List[dict]:
    """
    v1: ["<url_sha1>", ...]
    v2: [{"url_hash":..., "id":..., "t_norm":..., "ts":"%Y-%m-%dT%H:%M:%SZ"}, ...]
    """
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # v1 -> v2 마이그레이션
        if isinstance(data, list) and data and isinstance(data[0], str):
            return [{"url_hash": s, "id": None, "t_norm": None, "ts": None} for s in data]
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []

def save_sent_cache(records: List[dict]) -> None:
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.getLogger().warning("전송 캐시 저장 실패: %s", e)

def _parse_pub_ts_z(ts: Optional[str]) -> Optional[dt.datetime]:
    if not ts:
        return None
    try:
        # 모든 timestamp는 UTC Z로 저장하는 것을 가정
        return dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None

def _prune_cache(cache: List[dict], keep_hours: int = 168) -> List[dict]:
    """캐시를 최근 keep_hours(기본 7일)만 유지"""
    if not cache:
        return []
    cutoff = (now_kst() - dt.timedelta(hours=keep_hours)).astimezone(dt.timezone.utc)
    out = []
    for c in cache:
        ts = _parse_pub_ts_z(c.get("ts"))
        if (ts is None) or (ts >= cutoff):  # ts 없으면 일단 유지
            out.append(c)
    # 크기 안전장치(과도 성장 방지)
    if len(out) > 5000:
        out = out[-5000:]
    return out

def is_cached_duplicate(item: dict, cache: List[dict], title_sim_hours: int = 168) -> bool:
    """URL/정규화URLID/정규화제목(근사) 기준으로 캐시 중복 판단"""
    url = item.get("url", "")
    cid = canonical_url_id(url)
    urlh = sha1(url)
    tnorm = normalize_title(item.get("title", ""))
    its = _parse_pub_ts_z(item.get("publishedAt")) or dt.datetime.now(dt.timezone.utc)

    cutoff = its - dt.timedelta(hours=title_sim_hours)

    for c in cache:
        if urlh and urlh == c.get("url_hash"):
            return True
        if cid and cid == c.get("id"):
            return True
        # 제목 근사중복은 일정 기간 내에서만 비교(너무 옛날 기사와의 충돌 방지)
        cts = _parse_pub_ts_z(c.get("ts"))
        if c.get("t_norm") and cts and cts >= cutoff:
            if is_near_dup(tnorm, c["t_norm"]):
                return True
    return False

# ==============================
# Collectors
# ==============================

def _http_get(url: str, headers: Dict=None, params: Dict=None, timeout: int=10) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp
        return None
    except Exception:
        return None

def _to_zulu(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def collect_naver(cfg: Dict) -> List[Dict]:
    """
    Naver Search News API 기반 수집
    config:
      KEYWORDS: ["국내 PE", "사모펀드", ...]
      MAX_RESULTS_PER_SOURCE: 50
      RECENCY_HOURS: 72
    """
    items = []
    client_id = cfg.get("NAVER_CLIENT_ID", "")
    client_secret = cfg.get("NAVER_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return items

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }
    max_results = int(cfg.get("MAX_RESULTS_PER_SOURCE", 50))
    rec_hours = int(cfg.get("RECENCY_HOURS", 72))

    since = now_kst() - dt.timedelta(hours=rec_hours)
    base = "https://openapi.naver.com/v1/search/news.json"
    for q in cfg.get("KEYWORDS", []):
        start = 1
        fetched = 0
        while fetched < max_results and start <= 1000:
            params = {
                "query": q,
                "display": min(100, max_results - fetched),
                "start": start,
                "sort": "date"
            }
            r = _http_get(base, headers=headers, params=params, timeout=10)
            if not r:
                break
            j = r.json()
            arr = j.get("items", [])
            if not arr:
                break
            for it in arr:
                link = it.get("link") or it.get("originallink") or ""
                title = re.sub("<.*?>", "", it.get("title", ""))  # 태그 제거
                desc = re.sub("<.*?>", "", it.get("description", ""))
                pub = it.get("pubDate")  # 예: Thu, 10 Oct 2025 10:18:00 +0900
                try:
                    pub_dt = dt.datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z")
                except Exception:
                    pub_dt = now_kst()
                if pub_dt < since:
                    continue
                items.append({
                    "source": "naver",
                    "query": q,
                    "title": title.strip(),
                    "description": desc.strip(),
                    "url": link,
                    "publishedAt": _to_zulu(pub_dt),
                })
            fetched += len(arr)
            start += len(arr)
            if len(arr) == 0:
                break
            time.sleep(0.1)
    return items

def collect_newsapi(cfg: Dict) -> List[Dict]:
    """
    NewsAPI everything endpoint
    """
    items = []
    key = cfg.get("NEWSAPI_KEY", "")
    if not key:
        return items

    max_results = int(cfg.get("MAX_RESULTS_PER_SOURCE", 50))
    rec_hours = int(cfg.get("RECENCY_HOURS", 72))
    since = (now_kst() - dt.timedelta(hours=rec_hours)).astimezone(dt.timezone.utc)

    base = "https://newsapi.org/v2/everything"
    headers = {"Authorization": key}
    for q in cfg.get("KEYWORDS", []):
        page = 1
        fetched = 0
        while fetched < max_results and page <= 5:
            params = {
                "q": q,
                "language": "ko",
                "pageSize": min(100, max_results - fetched),
                "page": page,
                "sortBy": "publishedAt"
            }
            r = _http_get(base, headers=headers, params=params, timeout=10)
            if not r:
                break
            j = r.json()
            arr = j.get("articles", [])
            if not arr:
                break
            for a in arr:
                title = a.get("title") or ""
                desc = a.get("description") or ""
                url = a.get("url") or ""
                src = a.get("source", {}).get("name") or "unknown"
                pub = a.get("publishedAt")  # UTC ISO
                try:
                    pub_dt = dt.datetime.fromisoformat(pub.replace("Z","+00:00"))
                except Exception:
                    pub_dt = dt.datetime.now(dt.timezone.utc)
                if pub_dt < since.astimezone(dt.timezone.utc):
                    continue
                items.append({
                    "source": f"newsapi:{src}",
                    "query": q,
                    "title": title.strip(),
                    "description": desc.strip(),
                    "url": url,
                    "publishedAt": _to_zulu(pub_dt),
                })
            fetched += len(arr)
            page += 1
            if len(arr) == 0:
                break
            time.sleep(0.1)
    return items

# ==============================
# Filtering, Dedup (intra-run), Ranking
# ==============================

def dedup_intra_run(items: List[Dict]) -> List[Dict]:
    """한 번의 실행 내 중복 제거 (URL/ID/제목근사)"""
    out = []
    seen_urlh = set()
    seen_ids = set()
    seen_titles = []
    for it in sorted(items, key=lambda x: x.get("publishedAt",""), reverse=True):
        url = it.get("url","")
        uid = canonical_url_id(url)
        th = sha1(url) if url else None
        tnorm = normalize_title(it.get("title",""))
        dup = False
        if th and th in seen_urlh: dup = True
        if uid and uid in seen_ids: dup = True
        if not dup:
            # 제목 근사 중복 확인
            for tn in seen_titles[-300:]:  # 최근 것과만 비교
                if is_near_dup(tnorm, tn):
                    dup = True
                    break
        if dup:
            continue
        if th: seen_urlh.add(th)
        if uid: seen_ids.add(uid)
        seen_titles.append(tnorm)
        out.append(it)
    return out

def rank_filtered(items: List[Dict], cfg: Dict) -> List[Dict]:
    # 단순히 최신순 정렬. 필요시 매체 가중치/키워드 점수 반영 가능
    return sorted(items, key=lambda x: x.get("publishedAt",""), reverse=True)

# ==============================
# Telegram
# ==============================

def format_telegram_text(items: List[Dict]) -> str:
    if not items:
        return "📭 신규 뉴스 없음"
    lines = ["📌 국내 PE 동향 관련 뉴스"]
    for it in items[:30]:
        t = it.get("title","").strip()
        u = it.get("url","").strip()
        pub = it.get("publishedAt","")  # Zulu
        # 표시용: KST 로컬 시각 HH:MM
        try:
            pdt = dt.datetime.strptime(pub, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=9)))
            pstr = pdt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pstr = ""
        lines.append(f"• {t} ({u}) ({pstr})")
    return "\n".join(lines)

def send_telegram(bot_token: str, chat_id: str, text: str) -> bool:
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text
    }
    try:
        r = requests.post(url, data=data, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

# ==============================
# Collection Orchestrator
# ==============================

def collect_all(cfg: Dict, env: Dict) -> List[Dict]:
    items = []
    # 수집기 실행
    items += collect_naver({**cfg, **env})
    items += collect_newsapi({**cfg, **env})
    # 실행 내 중복 제거
    items = dedup_intra_run(items)
    return items

def _should_skip_by_time(cfg: Dict) -> bool:
    """업무시간/주말 스킵 로직 필요시 활성화"""
    # 필요 시 cfg["ONLY_BUSINESS_HOURS"]=True & HOURS=[9,18] 등으로 확장 가능
    return False

# ==============================
# Main transmit logic (with cross-run dedup via cache)
# ==============================

def transmit_once(cfg: Dict, env: Dict, preview: bool=False) -> Dict:
    run_lock = get_run_lock()
    if not run_lock.acquire(blocking=False):
        logging.info("다른 실행이 진행 중이어서 이번 주기는 스킵합니다.")
        return {"count": 0, "items": []}
    try:
        all_items = collect_all(cfg, env)
        ranked = rank_filtered(all_items, cfg)  # 주기 내(동일 실행) 중복 제거

        if preview:
            return {"count": len(ranked), "items": ranked}

        if _should_skip_by_time(cfg):
            logging.info("시간 정책에 의해 전송 건너뜀 (업무시간/주말/공휴일)")
            return {"count": 0, "items": []}

        cache = _prune_cache(load_sent_cache(), keep_hours=max(72, int(cfg.get("RECENCY_HOURS", 72))*2))

        #跨-주기 중복 제거: 캐시와도 근사중복 검사
        new_items = []
        for it in ranked:
            if not is_cached_duplicate(it, cache, title_sim_hours=max(72, int(cfg.get("RECENCY_HOURS", 72))*2)):
                new_items.append(it)

        if not new_items:
            send_telegram(env.get("TELEGRAM_BOT_TOKEN", ""), env.get("TELEGRAM_CHAT_ID", ""), "📭 신규 뉴스 없음")
            return {"count": 0, "items": []}

        BATCH = 30
        sent_any = False
        for i in range(0, len(new_items), BATCH):
            chunk = new_items[i:i+BATCH]
            text = format_telegram_text(chunk)
            ok = send_telegram(env.get("TELEGRAM_BOT_TOKEN", ""), env.get("TELEGRAM_CHAT_ID", ""), text)
            sent_any = sent_any or ok
            time.sleep(0.6)

        if sent_any:
            # 캐시에 URL/정규화ID/정규화제목/발행시각 기록
            for it in new_items:
                cache.append({
                    "url_hash": sha1(it.get("url","")),
                    "id": canonical_url_id(it.get("url","")),
                    "t_norm": normalize_title(it.get("title","")),
                    "ts": it.get("publishedAt")
                })
            save_sent_cache(_prune_cache(cache))

        return {"count": len(new_items), "items": new_items}
    finally:
        run_lock.release()

# ==============================
# CLI
# ==============================

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--config", default="/mnt/data/config.json", help="config.json 경로")
    parser.add_argument("--preview", action="store_true", help="수집/필터 후 미리보기(전송 없음)")
    parser.add_argument("--run-once", action="store_true", help="한 번 전송")
    parser.add_argument("--schedule", type=int, default=0, help="분 단위 반복 실행(예: 60)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = get_env_from_cfg(cfg)

    if args.preview:
        result = transmit_once(cfg, env, preview=True)
        text = format_telegram_text(result["items"])
        print(text)
    elif args.run-once:
        result = transmit_once(cfg, env, preview=False)
        print(f"sent={result['count']}")
    elif args.schedule and args.schedule > 0:
        interval = args.schedule
        print(f"[{APP_NAME}] 시작, 주기={interval}분")
        while True:
            try:
                result = transmit_once(cfg, env, preview=False)
                logging.info("cycle done: sent=%d", result["count"])
            except Exception as e:
                logging.exception("cycle error: %s", e)
            time.sleep(interval * 60)
    else:
        # 기본: preview
        result = transmit_once(cfg, env, preview=True)
        text = format_telegram_text(result["items"])
        print(text)
