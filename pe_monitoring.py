# -*- coding: utf-8 -*-
"""
pe_monitoring_llm.py
--------------------
뉴스 수집 → 규칙 필터/랭킹 → OpenAI LLM 재평가 → 텔레그램 전송(한 줄 근거 표시 제외)

사용 전 준비
1) 배포 환경 변수 등록
   - NEWSAPI_KEY, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET (선택), TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, OPENAI_API_KEY
2) config.json 스키마 예시 (필수 키만 나열)
{
  "KEYWORDS": ["PEF", "사모펀드", "바이아웃", "공개매수", "M&A", "VIG", "MBK", "IMM"],
  "EXCLUDE_KEYWORDS": ["연예", "스포츠", "브랜드평판"],
  "DOMAIN_WHITELIST": ["www.thebell.co.kr", "dealsite.co.kr", "www.investchosun.com", "n.news.naver.com"],
  "DOMAIN_BLACKLIST": ["entertain.naver.com"],
  "MIN_PUBLISHED_HOURS": 120,
  "PAGE_SIZE": 30,
  "MAX_ITEMS": 60,
  "SHOW_SOURCE_DOMAIN": true,

  "LLM_ENABLE": true,
  "LLM_MODEL": "gpt-4o-mini",
  "LLM_MIN_SCORE": 70,
  "LLM_BATCH_SIZE": 12,
  "LLM_SYSTEM_PROMPT": "너는 국내 사모펀드/PE 동향을 선별하는 리서치 어시스턴트다...",
  "LLM_USER_TEMPLATE": "아래 기사들을 0~100점으로 채점하고, 60점 미만은 drop하라. 각 항목은 JSONL로 답하라: {\"keep\":true|false, \"score\":0-100, \"reason\":\"한 줄 근거\"}. 참고 키워드: {{KEYWORDS}}. 참고 운용사 watchlist: {{FIRM_WATCHLIST}}.\n\n기사목록(JSONL):\n{{ITEMS_JSONL}}",

  "FIRM_WATCHLIST": ["MBK", "IMM", "Hahn&Company", "VIG", "한앤컴퍼니"]
}
"""

import os
import io
import re
import json
import time
import copy
import logging as log
import datetime as dt
from typing import List, Dict, Any

import requests

CONFIG_PATH = "/mnt/data/config.json"
SENT_CACHE_PATH = "/mnt/data/sent_cache.json"

APP_TZ = dt.timezone(dt.timedelta(hours=9))  # Asia/Seoul

# --------------------------- 유틸 ---------------------------

def now_kst() -> dt.datetime:
    return dt.datetime.now(tz=APP_TZ)

def load_json(path: str, default: Any) -> Any:
    try:
        with io.open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return copy.deepcopy(default)

def save_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def domain_of(url: str) -> str:
    try:
        return re.sub(r"^www\.", "", requests.utils.urlparse(url).netloc)
    except Exception:
        return ""

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def hours_from_utc_now(iso_utc: str) -> float:
    # iso_utc: "2025-10-04T07:30:00Z" 또는 비슷한 형식
    try:
        if iso_utc.endswith("Z"):
            t = dt.datetime.strptime(iso_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        else:
            t = dt.datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        diff = dt.datetime.now(tz=dt.timezone.utc) - t.astimezone(dt.timezone.utc)
        return diff.total_seconds() / 3600.0
    except Exception:
        return 1e9

# --------------------------- 환경/설정 ---------------------------

def load_env() -> Dict[str, str]:
    return {
        "NEWSAPI_KEY": os.getenv("NEWSAPI_KEY", ""),
        "NAVER_CLIENT_ID": os.getenv("NAVER_CLIENT_ID", ""),
        "NAVER_CLIENT_SECRET": os.getenv("NAVER_CLIENT_SECRET", ""),
        "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", ""),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    }

def load_cfg() -> Dict[str, Any]:
    default_cfg = {
        "KEYWORDS": [],
        "EXCLUDE_KEYWORDS": [],
        "DOMAIN_WHITELIST": [],
        "DOMAIN_BLACKLIST": [],
        "MIN_PUBLISHED_HOURS": 120,
        "PAGE_SIZE": 30,
        "MAX_ITEMS": 60,
        "SHOW_SOURCE_DOMAIN": True,

        "LLM_ENABLE": False,
        "LLM_MODEL": "gpt-4o-mini",
        "LLM_MIN_SCORE": 70,
        "LLM_BATCH_SIZE": 12,
        "LLM_SYSTEM_PROMPT": "너는 국내 사모펀드/PE 동향을 선별하는 리서치 어시스턴트다.",
        "LLM_USER_TEMPLATE": "아래 기사들을 0~100점으로 채점하고, 60점 미만은 drop하라. 각 항목은 JSONL로 답하라: {\"keep\":true|false, \"score\":0-100, \"reason\":\"한 줄 근거\"}.\n\n기사목록(JSONL):\n{{ITEMS_JSONL}}",
        "FIRM_WATCHLIST": []
    }
    cfg = load_json(CONFIG_PATH, default_cfg)
    # 안전 범위
    cfg["PAGE_SIZE"] = clamp(int(cfg.get("PAGE_SIZE", 30)), 10, 100)
    cfg["MAX_ITEMS"] = clamp(int(cfg.get("MAX_ITEMS", 60)), 10, 200)
    cfg["LLM_MIN_SCORE"] = clamp(int(cfg.get("LLM_MIN_SCORE", 70)), 0, 100)
    cfg["LLM_BATCH_SIZE"] = clamp(int(cfg.get("LLM_BATCH_SIZE", 12)), 1, 50)
    cfg["MIN_PUBLISHED_HOURS"] = max(0, int(cfg.get("MIN_PUBLISHED_HOURS", 120)))
    return cfg

# --------------------------- 수집기 ---------------------------

def search_newsapi(query: str, page_size: int, api_key: str) -> List[Dict[str, Any]]:
    if not api_key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": clamp(int(page_size), 10, 100),
        "language": "ko",
        "sortBy": "publishedAt",
    }
    r = requests.get(url, params=params, headers={"X-Api-Key": api_key}, timeout=20)
    r.raise_for_status()
    data = r.json()
    out = []
    for a in data.get("articles", []):
        out.append({
            "title": a.get("title") or "",
            "url": a.get("url") or "",
            "publishedAt": (a.get("publishedAt") or "").replace(".000Z", "Z"),
            "source": (a.get("source") or {}).get("name") or domain_of(a.get("url") or ""),
        })
    return out

def collect_all(cfg: Dict[str, Any], env: Dict[str, str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    # 간단히 키워드 조합 1~2개만 사용(운영 환경에서 필요시 확장)
    q = " OR ".join([f"\"{k}\"" for k in cfg.get("KEYWORDS", [])]) or "사모펀드 OR PEF OR 공개매수 OR 바이아웃"
    try:
        items.extend(search_newsapi(q, cfg["PAGE_SIZE"], env.get("NEWSAPI_KEY","")))
    except Exception as e:
        log.warning("NewsAPI 수집 실패: %s", e)

    # 중복 제거 (url 기준)
    seen = set()
    uniq = []
    for it in items:
        u = it.get("url","")
        if not u or u in seen:
            continue
        seen.add(u)
        uniq.append(it)
    return uniq[:cfg["MAX_ITEMS"]]

# --------------------------- 규칙 필터/랭킹 ---------------------------

def contains_any(text: str, keywords: List[str]) -> bool:
    text = text or ""
    for k in keywords:
        if k and (k.lower() in text.lower()):
            return True
    return False

def rule_filter(items: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    keep = []
    wl = set([re.sub(r"^www\.", "", d) for d in cfg.get("DOMAIN_WHITELIST", [])])
    bl = set([re.sub(r"^www\.", "", d) for d in cfg.get("DOMAIN_BLACKLIST", [])])
    inc = cfg.get("KEYWORDS", [])
    exc = cfg.get("EXCLUDE_KEYWORDS", [])
    min_hours = cfg.get("MIN_PUBLISHED_HOURS", 120)

    for it in items:
        title = it.get("title","")
        url = it.get("url","")
        dom = domain_of(url)
        age_h = hours_from_utc_now(it.get("publishedAt",""))
        if bl and dom in bl:
            continue
        if wl and dom and dom not in wl:
            # 화이트리스트가 지정된 경우 화이트리스트 외는 컷
            continue
        if exc and contains_any(title, exc):
            continue
        if inc and (not contains_any(title, inc)):
            continue
        if age_h > min_hours:
            continue
        it["_score"] = 0.0
        # 간단 가중치: 최신 + 도메인 가산
        it["_score"] += max(0.0, 120.0 - age_h) * 0.2
        if dom in wl:
            it["_score"] += 10.0
        keep.append(it)

    # 근사 중복 제거(제목 유사도 낮게)
    out = []
    seen_title = set()
    for it in sorted(keep, key=lambda x: x.get("_score",0.0), reverse=True):
        tnorm = re.sub(r"[\s\W]+", "", it.get("title","").lower())
        if tnorm in seen_title:
            continue
        seen_title.add(tnorm)
        out.append(it)
    return out

def rank_filtered(items: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    # 여기서는 이미 _score가 부여되어 있음. 최신순/스코어순 정렬
    items.sort(key=lambda x: (x.get("_score",0.0), x.get("publishedAt","")), reverse=True)
    return items

# --------------------------- LLM 재평가 ---------------------------

def _jsonl(items: List[Dict[str, Any]]) -> str:
    lines = []
    for it in items:
        lines.append(json.dumps({
            "title": it.get("title",""),
            "url": it.get("url",""),
            "source": domain_of(it.get("url","")),
            "publishedAt": it.get("publishedAt",""),
        }, ensure_ascii=False))
    return "\n".join(lines)

def llm_filter(items: List[Dict[str, Any]], cfg: Dict[str, Any], env: Dict[str, str]) -> List[Dict[str, Any]]:
    if not items:
        return []
    if not cfg.get("LLM_ENABLE", False):
        return items
    api_key = env.get("OPENAI_API_KEY","")
    if not api_key:
        log.info("OPENAI_API_KEY 없음 → LLM 필터 생략")
        return items

    model = cfg.get("LLM_MODEL", "gpt-4o-mini")
    min_score = int(cfg.get("LLM_MIN_SCORE", 70))
    bsz = int(cfg.get("LLM_BATCH_SIZE", 12))
    sys_prompt = cfg.get("LLM_SYSTEM_PROMPT","")
    user_tpl = cfg.get("LLM_USER_TEMPLATE","")
    keywords = ", ".join(cfg.get("KEYWORDS", []))
    firms = ", ".join(cfg.get("FIRM_WATCHLIST", []))

    kept: List[Dict[str, Any]] = []
    for i in range(0, len(items), bsz):
        chunk = items[i:i+bsz]
        user_prompt = user_tpl.replace("{{KEYWORDS}}", keywords).replace("{{FIRM_WATCHLIST}}", firms).replace("{{ITEMS_JSONL}}", _jsonl(chunk))
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role":"system","content": sys_prompt},
                        {"role":"user","content": user_prompt}
                    ],
                    "temperature": 0.2
                },
                timeout=40
            )
            r.raise_for_status()
            txt = (r.json()["choices"][0]["message"]["content"] or "").strip()
            # JSONL 기대
            sel = []
            for line, it in zip(txt.splitlines(), chunk):
                keep, score = False, 0
                try:
                    obj = json.loads(line)
                    keep = bool(obj.get("keep", False))
                    score = int(obj.get("score", 0))
                    # reason 저장은 하되 메시지에는 사용하지 않음(요청사항)
                    it["_llm_score"] = score
                    it["_llm_reason"] = obj.get("reason", "")
                except Exception:
                    # 비정형이면 보수적으로 드랍
                    keep, score = False, 0
                if keep and score >= min_score:
                    sel.append(it)
            kept.extend(sel)
        except Exception as e:
            log.warning("LLM 호출 실패(%s) → 해당 배치는 원본 유지", e)
            kept.extend(chunk)
        time.sleep(0.2)
    kept.sort(key=lambda x: (x.get("_llm_score",0), x.get("_score",0.0)), reverse=True)
    return kept

# --------------------------- 텔레그램 ---------------------------

def telegram_send_message(text: str, env: Dict[str, str]) -> bool:
    token = env.get("TELEGRAM_BOT_TOKEN","")
    chat_id = env.get("TELEGRAM_CHAT_ID","")
    if not token or not chat_id:
        log.error("TELEGRAM 환경 변수 누락")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    }
    r = requests.post(url, json=payload, timeout=20)
    try:
        r.raise_for_status()
        return True
    except Exception as e:
        log.error("텔레그램 전송 실패: %s / %s", e, r.text[:200])
        return False

def format_telegram_text(items: List[Dict[str, Any]], cfg: Dict[str, Any]) -> str:
    if not items:
        return "📭 신규 뉴스 없음"
    show_src = bool(cfg.get("SHOW_SOURCE_DOMAIN", True))
    lines = ["📌 <b>국내 PE 동향 관련 뉴스</b>"]
    for it in items:
        t = (it.get("title","") or "").strip()
        u = it.get("url","")
        src = domain_of(u)
        when = "-"
        try:
            pub = dt.datetime.strptime(it["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).astimezone(APP_TZ)
            when = pub.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
        suffix = f" — {src} ({when})" if show_src else f" ({when})"
        lines.append(f"• <a href=\"{u}\">{t}</a>{suffix}")
    return "\n".join(lines)

# --------------------------- 캐시 ---------------------------

def load_sent_cache() -> Dict[str, Any]:
    d = load_json(SENT_CACHE_PATH, {"sent_urls": []})
    d["sent_urls"] = list(dict.fromkeys(d.get("sent_urls", [])))  # unique
    return d

def update_sent_cache(sent_urls: List[str]) -> None:
    d = load_sent_cache()
    base = set(d.get("sent_urls", []))
    base.update(sent_urls)
    d["sent_urls"] = list(base)[-5000:]  # 최근 5000개만 유지
    save_json(SENT_CACHE_PATH, d)

def filter_already_sent(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    d = load_sent_cache()
    seen = set(d.get("sent_urls", []))
    out = []
    for it in items:
        u = it.get("url","")
        if u and u not in seen:
            out.append(it)
    return out

# --------------------------- 파이프라인 ---------------------------

def transmit_once() -> Dict[str, Any]:
    env = load_env()
    cfg = load_cfg()

    raw = collect_all(cfg, env)
    r1 = rule_filter(raw, cfg)
    ranked = rank_filtered(r1, cfg)
    ranked = llm_filter(ranked, cfg, env)  # LLM 재평가
    new_items = filter_already_sent(ranked)

    text = format_telegram_text(new_items, cfg)
    ok = telegram_send_message(text, env) if new_items else False

    if ok and new_items:
        update_sent_cache([it.get("url","") for it in new_items])

    return {
        "collected": len(raw),
        "rule_kept": len(r1),
        "ranked": len(ranked),
        "new": len(new_items),
        "sent": ok and bool(new_items)
    }

# --------------------------- 실행 ---------------------------

if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = transmit_once()
    print(json.dumps(res, ensure_ascii=False, indent=2))
