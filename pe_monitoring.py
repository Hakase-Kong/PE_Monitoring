
# -*- coding: utf-8 -*-
"""
pe_monitoring.py — 국내 PE 동향 뉴스 모니터링 (Telegram 전송)

핵심 변경점
- 중복 제거 강화: "정규화 URL-ID" + "정규화 제목 시그니처"를 캐시에 함께 저장/조회
- 주기(시간) 간에도 동일/유사 기사 재전송 억제
- 캐시 보존기간 슬라이딩 윈도우(기본 7일, 설정 가능)
- 네이버/일반 URL canonicalization 강화 (모바일/데스크톱, 쿼리스트링 차이 허용)
- 간단한 스케줄러 옵션(--schedule-minutes) 및 단발 실행(--once)

필요 파일
- config.json: 키워드/제외키워드/텔레그램 토큰/채팅ID/뉴스 API 키 등

실행 예시
- 한 번만 실행:   python pe_monitoring.py --once
- 60분 주기로:    python pe_monitoring.py --schedule-minutes 60
"""

import os
import re
import json
import time
import math
import hmac
import hashlib
import logging as log
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from urllib.parse import urlparse, parse_qs, urlunparse

import requests

# ---------- 설정 ----------

DEFAULT_CONFIG_PATH = os.environ.get("PE_MONITOR_CONFIG", "config.json")
CACHE_FILE = os.environ.get("PE_MONITOR_CACHE", "sent_cache.json")
CACHE_RETENTION_HOURS = int(os.environ.get("PE_MONITOR_CACHE_RETENTION_HOURS", "168"))  # 7일
MAX_TELEGRAM_ITEMS_PER_MESSAGE = int(os.environ.get("PE_MONITOR_MAX_ITEMS", "30"))
REQUEST_TIMEOUT = (6.0, 12.0)  # (connect timeout, read timeout)

SEOUL_TZ = dt.timezone(dt.timedelta(hours=9))  # Asia/Seoul

# ---------- 유틸 ----------

def _utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _to_seoul_str(when: dt.datetime) -> str:
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return when.astimezone(SEOUL_TZ).strftime("%Y-%m-%d %H:%M")

def load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("JSON 로드 실패(%s): %s", path, e)
        return default

def save_json(path: str, data: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("JSON 저장 실패(%s): %s", path, e)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

# ---------- 제목/URL 정규화 ----------

_RE_NEWS_SUFFIXES = re.compile(r"""
    # 흔한 접미사/태그/대괄호 표시 제거
    [\[\(【]?\s?(단독|종합|속보|마켓인|특징주|시그널|알쓸신잡|PE는\s*지금)\s?[\]\)】]?
""", re.IGNORECASE | re.VERBOSE)

_RE_WHITES = re.compile(r"\s+")

def normalize_title(title: str) -> str:
    """
    전송 중복 억제를 위한 보수적 제목 정규화
    - 괄호 내 수식어 제거([단독], (종합) 등)
    - 공백/연속기호 정리
    - 대소문자 표준화
    """
    if not title:
        return ""
    t = title
    # 괄호/브라켓 태그류 일부 제거 (필요 시 패턴 확장)
    t = _RE_NEWS_SUFFIXES.sub("", t)
    # 하이픈/콜론 주변 공백 정리
    t = t.replace(" - ", " ").replace(" : ", ": ").replace("…", "...")
    # 공백 정규화
    t = _RE_WHITES.sub(" ", t).strip()
    # 대소문자 표준화 (한글엔 영향 거의 없음)
    t = t.lower()
    return t

_RE_NAVER_OID = re.compile(r"[?&]oid=(\d+)")
_RE_NAVER_AID = re.compile(r"[?&]aid=(\d+)")

def canonical_url_id(url: str) -> str:
    """
    기사 URL을 출처 불문하고 안정적인 ID로 정규화
    - 네이버: oid/aid 추출 → naver:oid:aid
    - 기타: (scheme 제거) netloc + path (쿼리 제거), 모바일 서브도메인은 netloc 정규화
    """
    try:
        if not url:
            return ""
        u = url.strip()
        # 네이버 뉴스: oid / aid 기반
        if "naver.com" in u and ("/mnews/" in u or "/news/" in u):
            # 모바일/데스크톱 도메인 구분 무시
            try:
                qs = parse_qs(urlparse(u).query)
                oid = qs.get("oid", [None])[0]
                aid = qs.get("aid", [None])[0]
                if (not oid or not aid):
                    # 일부 경로형 URL의 경우 path에서 추출
                    # 예: https://n.news.naver.com/mnews/article/018/0006134096
                    p = urlparse(u).path
                    m = re.search(r"/article/(\d+)/(\d+)", p)
                    if m:
                        oid, aid = m.group(1), m.group(2)
                if oid and aid:
                    return f"naver:{oid}:{aid}"
            except Exception:
                pass

        # 그 외: 스킴 제거 + 쿼리 제거 + 모바일 서브도메인 정리
        parts = urlparse(u)
        netloc = parts.netloc.lower()
        # 모바일 하위도메인 정규화 (예: m.xxx.com → xxx.com)
        if netloc.startswith("m."):
            netloc = netloc[2:]
        path = parts.path
        # 일부 사이트는 /amp, /m 등 변형 제거
        path = re.sub(r"/amp/?$", "/", path, flags=re.IGNORECASE)
        path = re.sub(r"/m(/|$)", "/", path, flags=re.IGNORECASE)
        return f"{netloc}{path}".rstrip("/")
    except Exception:
        return url

# ---------- 수집 ----------

def fetch_naver_news(keywords: List[str], size_per_kw: int = 15) -> List[Dict[str, Any]]:
    """
    Naver News(검색) 크롤 기반 간이 수집.
    - 공식 OpenAPI(검색뉴스)가 있다면 그 엔드포인트 사용 권장.
    - 여기서는 웹검색 HTML을 사용하지 않고 naver 기사 고정 패턴을 대상으로 기사 본문 URL 형태를 우선 수집.
    주: 실서비스에선 자체 구현/사내 프록시 등 권장.
    """
    # 이 예제는 간소화. 실제로는 News API / 네이버 검색 API를 함께 사용하시길 권장.
    items: List[Dict[str, Any]] = []
    for kw in keywords:
        # news API가 없다고 가정하고, 대체로 사용자의 기존 파이프라인에서 수집된 결과를 이 함수로 합친다고 생각하세요.
        # 필요 시 이 자리를 사용자의 기존 모듈 호출로 교체하세요.
        # 예시에서는 빈 구현 (실 배포 환경에서 기존 수집 함수 이용)
        _ = kw  # placate linter
        pass
    return items

def fetch_newsapi_everything(api_key: str, keywords: List[str], language: str = "ko", page_size: int = 50) -> List[Dict[str, Any]]:
    """
    NewsAPI.org Everything 엔드포인트 사용 (키워드 OR 수집).
    """
    if not api_key:
        return []
    headers = {"User-Agent": "Mozilla/5.0 (PE-Monitor/1.0)"}
    items: List[Dict[str, Any]] = []
    query = " OR ".join([f"\"{kw}\"" for kw in keywords if kw.strip()])
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": api_key,
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        for a in data.get("articles", []):
            title = a.get("title") or ""
            url_ = a.get("url") or ""
            published_at = a.get("publishedAt")
            if published_at:
                try:
                    when = dt.datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                except Exception:
                    when = dt.datetime.now(dt.timezone.utc)
            else:
                when = dt.datetime.now(dt.timezone.utc)
            items.append({
                "title": title.strip(),
                "url": url_.strip(),
                "published_at": when.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": (a.get("source") or {}).get("name"),
            })
    except Exception as e:
        log.warning("NewsAPI 수집 실패: %s", e)
    return items

def collect_all(cfg: Dict[str, Any], env: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    다양한 소스에서 수집해 하나의 리스트로 결합.
    - 사용 중인 파이프라인에 맞춰 이 함수를 확장/교체하세요.
    """
    keywords = cfg.get("keywords", ["PEF", "사모펀드", "M&A", "인수합병"])
    use_newsapi = cfg.get("use_newsapi", True)

    items: List[Dict[str, Any]] = []

    if use_newsapi:
        items += fetch_newsapi_everything(env.get("NEWSAPI_KEY", ""), keywords=keywords)

    # 여기에 Naver News API/사내수집 모듈 결과를 병합
    # items += fetch_naver_news(keywords)

    # 간단 정리: title/url 없는 항목 제거
    items = [it for it in items if it.get("title") and it.get("url")]
    return items

# ---------- 필터링/랭킹 ----------

DEFAULT_INCLUDE_KEYWORDS = [
    "PEF", "사모펀드", "프라이빗에쿼티", "M&A", "바이아웃", "딜", "인수합병", "경영권 분쟁",
    "지분 인수", "투자 유치", "통매각", "공개매각", "예비입찰", "본입찰", "매각", "출자",
]

DEFAULT_EXCLUDE_KEYWORDS = [
    "연예", "스포츠", "날씨", "사설", "칼럼", "오피니언", "증시 급등락", "특징주", "리뷰",
]

def _contains_any(text: str, keywords: List[str]) -> bool:
    t = text or ""
    return any(kw.lower() in t.lower() for kw in keywords if kw.strip())

def filter_items(raw_items: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    inc = cfg.get("include_keywords", DEFAULT_INCLUDE_KEYWORDS)
    exc = cfg.get("exclude_keywords", DEFAULT_EXCLUDE_KEYWORDS)

    out = []
    for it in raw_items:
        title = it.get("title", "")
        if not _contains_any(title, inc):
            continue
        if _contains_any(title, exc):
            continue
        out.append(it)
    return out

def dedup_within_batch(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    같은 배치 내 유사제목/동일 URL-ID 제거 (보수적)
    """
    seen_ids = set()
    seen_titles = set()
    out = []
    for it in items:
        t_sig = normalize_title(it.get("title", ""))
        cid = canonical_url_id(it.get("url", ""))
        if cid in seen_ids or t_sig in seen_titles:
            continue
        seen_ids.add(cid)
        seen_titles.add(t_sig)
        out.append(it)
    return out

def rank_filtered(items: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    단순 정렬: 발행시각 desc → 제목 길이 안정성
    """
    def _k(it):
        ts = it.get("published_at")
        try:
            when = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            when = dt.datetime.now(dt.timezone.utc)
        return (when, -len(it.get("title", "")))
    items = filter_items(items, cfg)
    items = dedup_within_batch(items)
    return sorted(items, key=_k, reverse=True)

# ---------- 캐시(전송 이력) ----------

@dataclass
class SentEntry:
    id: str        # canonical_url_id
    title_sig: str # normalize_title
    ts_utc: str    # ISO8601 (UTC)

def load_sent_cache() -> Dict[str, Any]:
    """
    캐시 형식:
    {
      "entries": [
        {"id": "...", "title_sig": "...", "ts_utc": "....Z"},
        ...
      ],
      "legacy_url_hashes": ["abc...", ...]   # 구버전 호환
    }
    """
    data = load_json(CACHE_FILE, default={"entries": [], "legacy_url_hashes": []})
    # 구버전(list)일 경우 마이그레이션
    if isinstance(data, list):
        return {"entries": [], "legacy_url_hashes": data}
    if "entries" not in data:
        data["entries"] = []
    if "legacy_url_hashes" not in data:
        data["legacy_url_hashes"] = []
    return data

def save_sent_cache(cache: Dict[str, Any]) -> None:
    save_json(CACHE_FILE, cache)

def _prune_cache(cache: Dict[str, Any]) -> None:
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=CACHE_RETENTION_HOURS)
    kept = []
    for e in cache.get("entries", []):
        try:
            ts = dt.datetime.fromisoformat(e.get("ts_utc", "").replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=dt.timezone.utc)
        except Exception:
            ts = cutoff
        if ts >= cutoff:
            kept.append(e)
    # 안전 상한
    cache["entries"] = kept[-5000:]

def _is_seen(cache: Dict[str, Any], url: str, title: str) -> bool:
    can_id = canonical_url_id(url)
    t_sig  = normalize_title(title)
    for e in cache.get("entries", []):
        if e.get("id") == can_id or e.get("title_sig") == t_sig:
            return True
    # 구버전 호환 (sha1(url))
    legacy = set(cache.get("legacy_url_hashes", []))
    if sha1(url) in legacy:
        return True
    return False

def _cache_add_all(cache: Dict[str, Any], items: List[Dict[str, Any]]) -> None:
    for it in items:
        cache.setdefault("entries", []).append({
            "id": canonical_url_id(it.get("url","")),
            "title_sig": normalize_title(it.get("title","")),
            "ts_utc": _utcnow_iso(),
        })
    # 구버전 해시도 유지(후방호환)
    legacy = set(cache.get("legacy_url_hashes", []))
    legacy |= {sha1(it.get("url","")) for it in items}
    cache["legacy_url_hashes"] = sorted(list(legacy))[-10000:]
    _prune_cache(cache)
    save_sent_cache(cache)

# ---------- 포맷/텔레그램 ----------

def format_telegram_text(items: List[Dict[str, Any]], header: Optional[str] = None) -> str:
    if not header:
        header = "📌 국내 PE 동향 관련 뉴스"
    lines = [header]
    for it in items:
        title = it.get("title", "").strip()
        url = it.get("url", "").strip()
        ts = it.get("published_at")
        try:
            when = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            when = dt.datetime.now(dt.timezone.utc)
        seoul = _to_seoul_str(when)
        lines.append(f"• {title} ({url}) ({seoul})")
    return "\n".join(lines)

def send_telegram(bot_token: str, chat_id: str, text: str) -> bool:
    if not bot_token or not chat_id:
        log.warning("텔레그램 설정 누락(bot_token/chat_id)")
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning("텔레그램 전송 실패: %s", e)
        return False

# ---------- 메인 실행 ----------

def transmit_once(cfg: Dict[str, Any], env: Dict[str, str]) -> Dict[str, Any]:
    raw = collect_all(cfg, env)
    ranked = rank_filtered(raw, cfg)

    cache = load_sent_cache()
    new_items = [it for it in ranked if not _is_seen(cache, it.get("url",""), it.get("title",""))]

    if not new_items:
        send_telegram(env.get("TELEGRAM_BOT_TOKEN",""), env.get("TELEGRAM_CHAT_ID",""), "📭 신규 뉴스 없음")
        return {"count": 0, "items": []}

    # 텔레그램 분할 전송
    sent_any = False
    for i in range(0, len(new_items), MAX_TELEGRAM_ITEMS_PER_MESSAGE):
        chunk = new_items[i:i+MAX_TELEGRAM_ITEMS_PER_MESSAGE]
        text = format_telegram_text(chunk)
        ok = send_telegram(env.get("TELEGRAM_BOT_TOKEN",""), env.get("TELEGRAM_CHAT_ID",""), text)
        sent_any = sent_any or ok
        time.sleep(0.6)

    if sent_any:
        _cache_add_all(cache, new_items)

    return {"count": len(new_items), "items": new_items}

def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    cfg = load_json(path, default={})
    # 기본값 보강
    cfg.setdefault("keywords", DEFAULT_INCLUDE_KEYWORDS[:])
    cfg.setdefault("include_keywords", DEFAULT_INCLUDE_KEYWORDS[:])
    cfg.setdefault("exclude_keywords", DEFAULT_EXCLUDE_KEYWORDS[:])
    cfg.setdefault("use_newsapi", True)
    return cfg

def _env() -> Dict[str, str]:
    # 환경변수 + config.json의 telegram, newsapi 설정 병합
    env = {
        "TELEGRAM_BOT_TOKEN": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "TELEGRAM_CHAT_ID": os.environ.get("TELEGRAM_CHAT_ID", ""),
        "NEWSAPI_KEY": os.environ.get("NEWSAPI_KEY", ""),
    }
    # config.json에 별도 지정이 있으면 보강
    cfg = load_config()
    t = cfg.get("telegram", {})
    if t:
        env["TELEGRAM_BOT_TOKEN"] = env["TELEGRAM_BOT_TOKEN"] or t.get("bot_token", "")
        env["TELEGRAM_CHAT_ID"] = env["TELEGRAM_CHAT_ID"] or t.get("chat_id", "")
    n = cfg.get("newsapi", {})
    if n:
        env["NEWSAPI_KEY"] = env["NEWSAPI_KEY"] or n.get("api_key", "")
    return env

def run_scheduler(minutes: int) -> None:
    cfg = load_config()
    env = _env()
    interval = max(5, int(minutes))  # 최소 5분
    log.info("스케줄 시작: %d분 간격", interval)
    while True:
        try:
            transmit_once(cfg, env)
        except Exception as e:
            log.exception("주기 실행 오류: %s", e)
        finally:
            time.sleep(interval * 60)

def main():
    global DEFAULT_CONFIG_PATH
    parser = argparse.ArgumentParser(description="국내 PE 동향 뉴스 모니터링 (Telegram)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="설정 파일 경로 (기본: config.json)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true", help="한 번만 실행")
    group.add_argument("--schedule-minutes", type=int, help="N분 간격으로 무한 실행")
    args = parser.parse_args()

    # 로깅 설정
    log.basicConfig(
        level=log.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 설정 로드
    global DEFAULT_CONFIG_PATH
    DEFAULT_CONFIG_PATH = args.config
    cfg = load_config(DEFAULT_CONFIG_PATH)
    env = _env()

    if args.once:
        res = transmit_once(cfg, env)
        log.info("전송 결과: %s", res.get("count"))
    else:
        run_scheduler(args.schedule_minutes)

if __name__ == "__main__":
    main()
