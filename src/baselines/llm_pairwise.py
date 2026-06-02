"""LLM pairwise matcher baseline (OpenSanctions-Pairs-style).

NOT run autonomously — this drives external paid APIs. Run it yourself when
you're ready to spend the API budget (~$10-30 for sample mode, ~$200-1000
for full run).

Default model: claude-opus-4-7 (placeholder; set --model as needed).
Default provider: anthropic. Set --provider openai to use OpenAI.

Prompt structure (single-pair classification with brief reasoning):

  System: You are matching legal entity records that may represent the same
          economic entity across a lifecycle event (M&A, rename, spinoff,
          ticker change). Answer ONLY 'YES' or 'NO'.
  User:   ENTITY A
            legal_name: {a_name}
            jurisdiction: {a_country}
            legal_form:   {a_form}
          ENTITY B
            legal_name: {b_name}
            jurisdiction: {b_country}
            legal_form:   {b_form}
          Are these the same underlying economic entity across a possible
          lifecycle event?

Score = log-prob of "YES" if available, else 1.0 for YES / 0.0 for NO.

Sampling: use --sample-frac 0.05 (5%) to get a cheap signal first; then
extrapolate before paying for a full run.
"""
import argparse
import json
import os
import sys
import time

import pandas as pd
import pyarrow.parquet as pq

PROMPT_SYSTEM = (
    "You score whether two legal entity records describe the same "
    "underlying economic entity across a possible lifecycle event "
    "(M&A, rename, spinoff, ticker change, jurisdiction migration). "
    "The two records may have different legal names, jurisdictions, "
    "or legal forms when they describe the same continuing entity "
    "(e.g., 'Twitter Inc.' -> 'X Corp.'). Conversely, identical or "
    "similar names may belong to legally distinct entities (e.g., a "
    "parent and its similarly-named subsidiary). "
    "Output a single integer 0-10 on the first line: 0 = certainly "
    "different entities, 5 = ambiguous, 10 = certainly the same "
    "continuing entity across a lifecycle event. You may add one "
    "short reasoning sentence on a second line."
)

USER_TEMPLATE = """ENTITY A
  legal_name:   {a_name}
  jurisdiction: {a_country}
  legal_form:   {a_form}

ENTITY B
  legal_name:   {b_name}
  jurisdiction: {b_country}
  legal_form:   {b_form}

Score (0-10):"""

def parse_response(text):
    """Return (score in [0,1], raw_text). Looks for an integer 0-10.

    Robust against thinking-model output that mixes scale labels ("0-10")
    with the actual score. Priority order:
    1. A line that is JUST a digit (e.g. "8" or "8.")
    2. A line starting with "Score:", "Rating:" etc. followed by a digit
    3. The first non-scale digit on the last line
    """
    if not text:
        return 0.0, ""
    import re
    text_stripped = text.strip()
    lines = [ln.strip().rstrip(".:!,;") for ln in text_stripped.splitlines()
              if ln.strip()]

    # Pass 1: line that is JUST a number (with optional trailing punct)
    for line in lines:
        if re.fullmatch(r"(10|[0-9])", line):
            return int(line) / 10.0, text_stripped

    # Pass 2: line starting with "Score" / "Rating" / "Answer" + a number
    for line in lines:
        m = re.match(r"(?i)^(?:final\s+)?(?:score|rating|answer)\s*"
                      r"(?:\(0-?10\))?\s*[:=]?\s*(10|[0-9])\b", line)
        if m:
            return int(m.group(1)) / 10.0, text_stripped

    # Pass 3: search last non-empty line for the first integer 0-10
    # that is NOT part of a "0-10" / "0 to 10" scale label
    for line in reversed(lines):
        # Strip "0-10" / "0 to 10" / "0/10" scale labels
        cleaned = re.sub(r"\b0\s*[-/to]+\s*10\b", " ", line, flags=re.I)
        m = re.search(r"\b(10|[0-9])\b", cleaned)
        if m:
            return int(m.group(1)) / 10.0, text_stripped
        up = line.upper()
        if "YES" in up.split():
            return 1.0, text_stripped
        if "NO" in up.split():
            return 0.0, text_stripped
    return 0.5, text_stripped

def call_anthropic(client, model, system, user):
    # Prompt-cache the system message: it's identical across all calls, so the
    # cache write happens once and every subsequent call pays ~10x less for the
    # cached read. Saves ~45% on a Haiku-class model over a 6K+ call run.
    msg = client.messages.create(
        model=model,
        max_tokens=100,
        system=[{"type": "text", "text": system,
                  "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text if msg.content else ""

def call_openai(client, model, system, user, max_tokens=100):
    msg = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    content = msg.choices[0].message.content or ""
    # Thinking models (gpt-oss, qwen3, etc.) put reasoning in a separate
    # field and may leave content empty if max_tokens cut off the
    # generation mid-think. Fall back to the reasoning trace if content
    # is empty — the parser will still pick up the trailing number.
    if not content:
        reasoning = getattr(msg.choices[0].message, "reasoning", "") or ""
        if reasoning:
            content = reasoning
    return content

def call_gemini(model_obj, system, user):
    # Gemini's free tier is generous (~250 RPD on 2.5 Flash, more on 2.0 Flash)
    # and the paid tier is the cheapest option per call (<= 0.4x Haiku cost).
    # System instruction is set when the model is constructed; we pass only
    # the user content here.
    resp = model_obj.generate_content(user)
    return resp.text or ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001",
                    help="Anthropic: claude-haiku-4-5-20251001 (cheap, recommended), "
                         "claude-sonnet-4-6, claude-opus-4-7. OpenAI: gpt-5, etc.")
    ap.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "openai", "gemini", "ollama"])
    ap.add_argument("--sample-frac", type=float, default=0.05,
                    help="fraction of pairs to score; 1.0 = full run. "
                         "Ignored when --test-anchors-per-split is set.")
    ap.add_argument("--test-anchors-per-split", type=int, default=0,
                    help="if >0, sample N TEST anchors from EACH of the three "
                         "split regimes (TIME/EVENT/JURIS) and score ALL their "
                         "candidate pairs. JURIS uses all 88 anchors if N>=88. "
                         "Designed for stratified Hits@1 evaluation with "
                         "bootstrap CIs.")
    ap.add_argument("--anchor-file", default="",
                    help="path to a newline-delimited file of anchor LEIs. "
                         "If set, restricts scoring to test pairs whose anchor "
                         "is in this list. Overrides --test-anchors-per-split.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-name", default="llm_pairwise")
    ap.add_argument("--qps", type=float, default=4.0,
                    help="(legacy, unused with concurrency>1) approx req/s cap")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="number of in-flight API requests")
    args = ap.parse_args()

    if args.provider == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError:
            print("pip install anthropic", file=sys.stderr); sys.exit(1)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("set ANTHROPIC_API_KEY", file=sys.stderr); sys.exit(1)
        client = Anthropic()
        call = lambda s, u: call_anthropic(client, args.model, s, u)
    elif args.provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            print("pip install openai", file=sys.stderr); sys.exit(1)
        if not os.environ.get("OPENAI_API_KEY"):
            print("set OPENAI_API_KEY", file=sys.stderr); sys.exit(1)
        client = OpenAI()
        call = lambda s, u: call_openai(client, args.model, s, u)
    elif args.provider == "ollama":
        # Ollama exposes an OpenAI-compatible endpoint at localhost:11434/v1.
        # No API key needed; reuse the OpenAI SDK with a dummy key. Bump
        # max_tokens to 1500 because thinking models (gpt-oss, qwen3) emit
        # a long reasoning trace before the final answer.
        try:
            from openai import OpenAI
        except ImportError:
            print("pip install openai", file=sys.stderr); sys.exit(1)
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        call = lambda s, u: call_openai(client, args.model, s, u, max_tokens=1500)
    else:  # gemini
        try:
            import google.generativeai as genai
        except ImportError:
            print("pip install google-generativeai", file=sys.stderr); sys.exit(1)
        if not os.environ.get("GOOGLE_API_KEY"):
            print("set GOOGLE_API_KEY (from aistudio.google.com)", file=sys.stderr)
            sys.exit(1)
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        # Gemini takes the system instruction at model-construction time, not
        # per-call. PROMPT_SYSTEM is constant, so set it once.
        gem_model = genai.GenerativeModel(
            args.model,
            system_instruction=PROMPT_SYSTEM,
            generation_config={"max_output_tokens": 100, "temperature": 0.0},
        )
        call = lambda s, u: call_gemini(gem_model, s, u)

    if args.anchor_file:
        with open(args.anchor_file) as f:
            anchor_list = {ln.strip() for ln in f if ln.strip()}
        pairs = pq.read_table(
            os.path.join(args.data, "pairs.parquet"),
            columns=["anchor_lei", "candidate_lei", "label", "source",
                      "split_time", "split_event", "split_jurisdiction"],
        ).to_pandas()
        is_test = ((pairs["split_time"] == "test")
                    | (pairs["split_event"] == "test")
                    | (pairs["split_jurisdiction"] == "test"))
        pairs = pairs[is_test & pairs["anchor_lei"].isin(anchor_list)].reset_index(drop=True)
        pairs = pairs[["anchor_lei", "candidate_lei", "label", "source"]]
        print(f"loaded {len(anchor_list):,} anchors from {args.anchor_file}", file=sys.stderr)
        print(f"total pairs to score: {len(pairs):,}", file=sys.stderr)
    elif args.test_anchors_per_split > 0:
        # Stratified sampling: N test anchors per split, score ALL their pairs.
        # This yields a proper Hits@1-evaluable subset per split.
        pairs = pq.read_table(
            os.path.join(args.data, "pairs.parquet"),
            columns=["anchor_lei", "candidate_lei", "label", "source",
                      "split_time", "split_event", "split_jurisdiction"],
        ).to_pandas()
        print(f"loaded {len(pairs):,} pairs", file=sys.stderr)
        import numpy as np
        rng = np.random.default_rng(args.seed)
        chosen_anchors = set()
        for s in ("split_time", "split_event", "split_jurisdiction"):
            test = pairs[pairs[s] == "test"]
            eligible = [a for a, sub in test.groupby("anchor_lei")
                          if sub["label"].sum() > 0 and len(sub) >= 2]
            if len(eligible) <= args.test_anchors_per_split:
                pick = eligible
            else:
                pick = list(rng.choice(eligible,
                                        size=args.test_anchors_per_split,
                                        replace=False))
            print(f"  {s}: {len(pick)} anchors sampled "
                  f"(from {len(eligible)} eligible)", file=sys.stderr)
            chosen_anchors.update(pick)
        pairs = pairs[pairs["anchor_lei"].isin(chosen_anchors)].reset_index(drop=True)
        # Drop the split columns again so the downstream loop sees the
        # standard schema.
        pairs = pairs[["anchor_lei", "candidate_lei", "label", "source"]]
        print(f"total pairs to score: {len(pairs):,} "
              f"(across {len(chosen_anchors):,} unique anchors)", file=sys.stderr)
    else:
        pairs = pq.read_table(
            os.path.join(args.data, "pairs.parquet"),
            columns=["anchor_lei", "candidate_lei", "label", "source"],
        ).to_pandas()
        print(f"loaded {len(pairs):,} pairs", file=sys.stderr)
        if 0 < args.sample_frac < 1.0:
            pairs = pairs.sample(frac=args.sample_frac, random_state=args.seed).reset_index(drop=True)
            print(f"sampled {len(pairs):,} pairs", file=sys.stderr)

    lei_set = set(pairs["anchor_lei"]) | set(pairs["candidate_lei"])
    ent = pq.read_table(
        os.path.join(args.data, "entities.parquet"),
        columns=["lei", "legal_name", "legal_addr_country", "legal_form_code"],
    ).to_pandas()
    ent = ent[ent["lei"].isin(lei_set)].drop_duplicates(subset=["lei"], keep="first")
    ent_map = ent.set_index("lei").to_dict(orient="index")

    out_dir = os.path.join(args.data, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.out_name}.parquet")
    log_path = os.path.join(args.data, f"{args.out_name}_log.jsonl")

    print(f"writing predictions -> {out_path}", file=sys.stderr)
    print(f"writing prompt/response log -> {log_path}", file=sys.stderr)
    print(f"~est cost: {len(pairs)} calls; budget accordingly", file=sys.stderr)

    # Concurrent dispatch — API latency (~1.5s/call) is the bottleneck,
    # not rate limits. Run --concurrency requests in flight at once.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    def score_one(idx, r):
        a = ent_map.get(r["anchor_lei"], {})
        b = ent_map.get(r["candidate_lei"], {})
        if not a or not b:
            return idx, r["anchor_lei"], r["candidate_lei"], 0.0, int(r["label"]), ""
        user = USER_TEMPLATE.format(
            a_name=a.get("legal_name", "") or "(unknown)",
            a_country=a.get("legal_addr_country", "") or "(unknown)",
            a_form=a.get("legal_form_code", "") or "(unknown)",
            b_name=b.get("legal_name", "") or "(unknown)",
            b_country=b.get("legal_addr_country", "") or "(unknown)",
            b_form=b.get("legal_form_code", "") or "(unknown)",
        )
        try:
            resp = call(PROMPT_SYSTEM, user)
            score, raw = parse_response(resp)
        except Exception as e:
            score, raw = 0.0, f"ERROR: {e}"
        return idx, r["anchor_lei"], r["candidate_lei"], score, int(r["label"]), raw

    rows = [None] * len(pairs)
    log_lock = threading.Lock()
    completed = 0
    t0 = time.time()
    with open(log_path, "w") as logf, \
         ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(score_one, i, r) for i, r in pairs.iterrows()]
        for fut in as_completed(futures):
            idx, anc, cand, score, label, raw = fut.result()
            rows[idx] = (anc, cand, score)
            with log_lock:
                json.dump({"i": int(idx), "pred": anc, "cand": cand,
                           "label": label, "score": score,
                           "raw": raw[:300]}, logf)
                logf.write("\n")
            completed += 1
            if completed % 100 == 0:
                rate = completed / (time.time() - t0)
                eta_s = (len(pairs) - completed) / max(rate, 0.01)
                print(f"  {completed:,}/{len(pairs):,} ({rate:.1f} req/s, "
                      f"eta {int(eta_s)}s)", file=sys.stderr)

    out = pd.DataFrame(rows, columns=["anchor_lei", "candidate_lei", "score"])
    out.to_parquet(out_path, compression="snappy", index=False)
    elapsed = time.time() - t0
    print(f"done -> {out_path} ({elapsed:.1f}s, "
          f"{completed/elapsed:.1f} req/s)", file=sys.stderr)

if __name__ == "__main__":
    main()
