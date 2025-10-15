def llm_filter_items(items: List[dict], cfg: dict, env: dict) -> List[dict]:
    if not items or not bool(cfg.get("USE_LLM_FILTER", False)):
        return items

    api_key = env.get("OPENAI_API_KEY", "")
    if not api_key:
        log.warning("LLM 필터 활성화되어 있으나 OPENAI_API_KEY 미설정 → 규칙기반 결과 사용")
        return items

    model = cfg.get("LLM_MODEL", "gpt-4o-mini")
    base_th = float(cfg.get("LLM_CONF_THRESHOLD", 0.7))
    # 부정-선택형: irrelevant 확신이 매우 높을 때만 제외
    drop_th = min(1.0, base_th + 0.10)

    trusted = set(cfg.get("TRUSTED_SOURCES_FOR_FI", []))
    include = [w.lower() for w in (cfg.get("INCLUDE_TITLE_KEYWORDS", []) or [])]
    firms   = [w.lower() for w in (cfg.get("FIRM_WATCHLIST", []) or [])]

    def _has_kw_or_firm(t: str) -> bool:
        t = (t or "").lower()
        return any(k in t for k in include) or any(f in t for f in firms)

    out = []
    for it in items:
        src = domain_of(it.get("url",""))
        title = it.get("title","")

        # (바이패스1) 신뢰도메인 + (운용사/핵심키워드)면 LLM 생략하고 통과
        if (src in trusted) and _has_kw_or_firm(title):
            it["_llm_bypass"] = "trusted+kw"
            out.append(it)
            continue

        try:
            user_prompt = _llm_prompt_for_item(it, cfg)
            messages = [
                {"role": "system", "content":
                    "You are a professional news classifier for Private Equity (KR). "
                    "If unsure, prefer KEEP (relevant) to avoid false negatives. Return JSON only."
                },
                {"role": "user", "content": user_prompt},
            ]
            resp = _openai_chat(messages, api_key, model, max_tokens=int(cfg.get("LLM_MAX_TOKENS", 400)))
            meta = None
            try:
                meta = json.loads(resp.strip())
            except Exception:
                m = re.search(r"\{[\s\S]*\}$", resp.strip()); meta = json.loads(m.group(0)) if m else None

            # 기본은 통과(KEEP). 아래 조건에서만 DROP.
            drop = False
            if isinstance(meta, dict):
                cat = (meta.get("category","") or "").lower()
                conf = float(meta.get("confidence", 0.0))
                rel  = bool(meta.get("relevant", False))

                # (완화수용) 금융/산업 M&A라도 운용사/핵심키워드가 있으면 KEEP
                if _has_kw_or_firm(title):
                    rel = True

                # 부정-선택형: irrelevant 이면서 확신이 높을 때만 드롭
                if (rel is False) and (conf >= drop_th):
                    drop = True

                it["_llm"] = meta

            if not drop:
                out.append(it)
            else:
                it["_drop_reason"] = f"LLM irrelevant@{conf:.2f}"
                log.info("LLM drop: %s | %s", it.get("title",""), it.get('url',""))

        except Exception as e:
            log.warning("LLM 필터 실패(보류): %s", e)
            out.append(it)  # 실패 시 보수적으로 KEEP

    return out
