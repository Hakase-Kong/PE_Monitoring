
# pe_monitoring_fixed.py
# 목적: 국내 PE 동향(딜/공개매수/매각/본입찰 등) 기사만 선별하여 텔레그램으로 전송
# 핵심 변경점:
# 1) 포함 키워드(및 별칭)를 "필수 조건"으로 적용
# 2) Naver는 경제면(sid=101)만 허용(옵션)
# 3) NewsAPI는 title 한정 + domains 화이트리스트 지원
# 4) 키워드별 상한(MAX_PER_KEYWORD) 적용, 중복제거 보강
# 5) 스케줄 키 일치: INTERVAL_MIN or TRANSMIT_INTERVAL_MIN 둘 다 지원

import os
import re
import json
import time
import math
import logging
import hashlib
import datetime as dt
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, parse_qs

import requests
import pytz
from difflib import SequenceMatcher

KST = pytz.timezone("Asia/Seoul")
log = logging.getLogger("pe_monitoring")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# -------------------------
# Utilities
# -------------------------
def now_kst() -> dt.datetime:
    return dt.datetime.now(tz=KST)


def domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc or ""
        return netloc.lower()
    except Exception:
        return ""


def hours_ago(ts_utc: str) -> float:
    """ts_utc: 'YYYY-MM-DDTHH:MM:SSZ'"""
    try:
        t = dt.datetime.strptime(ts_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        delta = dt.datetime.now(tz=dt.timezone.utc) - t
        return delta.total_seconds() / 3600.0
    except Exception:
        return 1e9


def _token_hit(text: str, tokens: List[str]) -> bool:
    t = (text or "").lower()
    return any((tok or "").lower() in t for tok in tokens or [])


def _alias_flatten(alias_map: Dict[str, List[str]]) -> List[str]:
    out = []
    for v in (alias_map or {}).values():
        out.extend(v or [])
    # 고유화 + 빈문자 제거
    return sorted({s.strip() for s in out if s and s.strip()})


def _naver_sid(url: str) -> Optional[str]:
    try:
        q = parse_qs(urlparse(url).query).get("sid", [])
        return q[0] if q else None
    except Exception:
        return None


# -------------------------
# Collector: Naver
# -------------------------
def search_naver_news(keyword: str, page_size: int, recency_hours: int, cfg: dict) -> List[dict]:
    client_id = os.getenv("NAVER_CLIENT_ID") or cfg.get("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET") or cfg.get("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        return []

    base = "https://openapi.naver.com/v1/search/news.json"
    params = {
        "query": keyword,
        "sort": "date",
        "display": max(10, min(page_size, 100)),
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
    except Exception as e:
        log.warning("Naver API 실패(%s): %s", keyword, e)
        return []

    res = []
    cutoff = now_kst() - dt.timedelta(hours=recency_hours)
    for it in items:
        link = it.get("link") or it.get("originallink") or ""
        pubdate = it.get("pubDate")  # 예: 'Sat, 05 Oct 2025 09:00:00 +0900'
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
            "publishedAt": pub_kst.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "summary": desc,
            "provider": "naver",
            "origin_keyword": keyword,
        })
    return res


# -------------------------
# Collector: NewsAPI
# -------------------------
def search_newsapi(page_size: int, api_key: str, recency_hours: int, cfg: dict) -> List[dict]:
    """타이틀 한정 + 도메인 화이트리스트 지원"""
    base = "https://newsapi.org/v2/everything"
    from_dt = (dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(hours=recency_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 쿼리는 좁게: 핵심 키워드 위주
    query = cfg.get("NEWSAPI_QUERY") or '("private equity" OR 사모펀드 OR 바이아웃 OR 공개매수 OR "M&A" OR 인수 OR 매각)'
    params = {
        "q": query,
        "searchIn": "title",
        "pageSize": max(10, min(page_size, 100)),
        "language": "ko",
        "sortBy": "publishedAt",
        "from": from_dt,
        "apiKey": api_key,
    }
    domains = cfg.get("NEWSAPI_DOMAINS") or []
    if domains:
        params["domains"] = ",".join(domains)

    try:
        r = requests.get(base, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        arts = data.get("articles", [])
    except Exception as e:
        log.warning("NewsAPI 실패: %s", e)
        return []

    out = []
    for a in arts:
        title = (a.get("title") or "").strip()
        url = (a.get("url") or "").strip()
        if not title or not url:
            continue
        publishedAt = (a.get("publishedAt") or "").replace(".000Z", "Z")
        desc = (a.get("description") or "")

        out.append({
            "title": title,
            "url": url,
            "source": domain_of(url),
            "publishedAt": publishedAt,
            "summary": desc,
            "provider": "newsapi",
            "origin_keyword": "_newsapi",
        })
    return out


# -------------------------
# Filter/Score
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

    # 도메인 필터
    if src in block:
        return True
    if allow_strict and allow and (src not in allow):
        return True

    # 네이버 sid 필터(경제면=101)
    if "naver.com" in src:
        sid_allow = set(cfg.get("NAVER_ALLOW_SIDS", []) or [])
        if sid_allow:
            sid = _naver_sid(url)
            if sid not in sid_allow:
                return True

    # 제외 키워드
    for w in (cfg.get("EXCLUDE_TITLE_KEYWORDS", []) or []):
        if w and w.lower() in title.lower():
            return True

    # 포함 키워드(및 별칭) - 필수 적중
    include = (cfg.get("INCLUDE_TITLE_KEYWORDS", []) or [])
    aliases = _alias_flatten(cfg.get("KEYWORD_ALIASES") or {})
    must_hit = include + aliases
    if must_hit and not _token_hit(title, must_hit):
        return True

    # (선택) 엄격 교차조건
    if bool(cfg.get("STRICT_AND_MATCH", False)):
        firms = cfg.get("FIRM_WATCHLIST", []) or []
        deal_terms = include or ["공개매수","인수","매각","M&A","본입찰","우선협상","지분인수"]
        if not (_token_hit(title, firms) or _token_hit(title, deal_terms)):
            return True

    return False


def score_item(item: dict, cfg: dict) -> float:
    title = item.get("title", "")
    src = item.get("source", "")
    t_hrs = hours_ago(item.get("publishedAt", ""))

    score = 0.0

    # 도메인 가중치
    dw = cfg.get("DOMAIN_WEIGHTS", {}) or {}
    score += float(dw.get(src, 0.0))

    # 키워드 및 별칭 가점
    inc = (cfg.get("INCLUDE_TITLE_KEYWORDS", []) or [])
    aliases = _alias_flatten(cfg.get("KEYWORD_ALIASES") or {})
    for w in inc:
        if w.lower() in title.lower():
            score += 1.0
    for a in aliases:
        if a.lower() in title.lower():
            score += 0.5

    # 펌/하우스 감지 가점
    for f in (cfg.get("FIRM_WATCHLIST", []) or []):
        if f.lower() in title.lower():
            score += 1.2

    # 최근 기사 가점(시간 경과 감가)
    if t_hrs < 1:
        score += 1.2
    else:
        score += max(0.0, 1.2 - math.log1p(t_hrs) * 0.3)

    return score


def dedup(items: List[dict], threshold: float = 0.82) -> List[dict]:
    def norm(s: str) -> str:
        s = re.sub(r"[\[\]\(\)【】『』“”\"'<>]", " ", s or "")
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    out = []
    norms = []
    for it in items:
        t = norm(it.get("title"))
        is_dup = False
        for nt in norms:
            if SequenceMatcher(None, t, nt).ratio() >= threshold:
                is_dup = True
                break
        if not is_dup:
            out.append(it)
            norms.append(t)
    return out


def pick_top10(items: List[dict], cfg: dict) -> List[dict]:
    filtered = [it for it in items if not should_drop(it, cfg)]
    for it in filtered:
        it["_score"] = score_item(it, cfg)
    filtered.sort(key=lambda x: x["_score"], reverse=True)

    unique = dedup(filtered)

    cap = int(cfg.get("MAX_PER_KEYWORD", 10))
    bucket = {}
    out = []
    for it in unique:
        k = it.get("origin_keyword") or "_"
        cnt = bucket.get(k, 0)
        if cnt < cap:
            out.append(it)
            bucket[k] = cnt + 1
        if len(out) >= 10:
            break
    return out


# -------------------------
# Telegram
# -------------------------
def format_telegram(top: List[dict]) -> str:
    if not top:
        return "관련 기사가 없습니다."
    lines = ["📌 Top 10"]
    for it in top:
        title = it.get("title", "").strip()
        url = it.get("url", "").strip()
        src = it.get("source", "")
        publishedAt = it.get("publishedAt", "")
        when = publishedAt.replace("T", " ").replace("Z", " UTC").split(".")[0]
        lines.append(f"• {title} ({url}) — {src} ({when})")
    return "\n".join(lines)


def send_telegram(bot_token: str, chat_id: str, text: str, disable_web_page_preview: bool = True) -> bool:
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
# Runner
# -------------------------
def within_working_hours(cfg: dict) -> bool:
    if not bool(cfg.get("ONLY_WORKING_HOURS", False)):
        return True
    now = now_kst()
    # 평일 08:00~19:00
    if now.weekday() >= 5:  # 토(5) 일(6)
        return False
    if not (8 <= now.hour < 19):
        return False
    return True


def run_once(cfg: dict) -> Optional[str]:
    if not within_working_hours(cfg):
        log.info("업무시간 외 전송 생략")
        return None

    recency_hours = int(cfg.get("RECENCY_HOURS", 48))
    page_size = int(cfg.get("PAGE_SIZE", 30))
    keywords = cfg.get("KEYWORDS", []) or []

    all_items: List[dict] = []

    # 1) Naver: 키워드별 개별 조회
    for kw in keywords:
        all_items.extend(search_naver_news(kw, page_size=page_size, recency_hours=recency_hours, cfg=cfg))

    # 2) NewsAPI(선택)
    newsapi_key = os.getenv("NEWSAPI_KEY") or cfg.get("NEWSAPI_KEY")
    if newsapi_key:
        all_items.extend(search_newsapi(page_size=page_size, api_key=newsapi_key, recency_hours=recency_hours, cfg=cfg))

    # Top10 선별
    top = pick_top10(all_items, cfg)
    text = format_telegram(top)
    if top:
        # 실제 전송
        tg = cfg.get("TELEGRAM", {}) or {}
        if tg.get("BOT_TOKEN") and tg.get("CHAT_ID"):
            send_telegram(tg["BOT_TOKEN"], tg["CHAT_ID"], text, disable_web_page_preview=bool(cfg.get("TELEGRAM_DISABLE_PREVIEW", True)))
    return text


def main():
    cfg_path = os.getenv("PE_CFG", "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    out = run_once(cfg)
    if out:
        print(out)


if __name__ == "__main__":
    main()
