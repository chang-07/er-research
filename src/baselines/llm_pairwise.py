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
    "You match legal entity records that may represent the same underlying "
    "economic entity across a lifecycle event (M&A, rename, spinoff, ticker "
    "change, jurisdiction migration). The two records may have different "
    "legal names, jurisdictions, or legal forms even when they describe the "
    "same continuing economic entity (e.g., 'Twitter Inc.' -> 'X Corp.'). "
    "Conversely, records with identical names may be legally distinct "
    "entities (e.g., a parent and its similarly-named subsidiary). "
    "Answer ONLY 'YES' or 'NO' on the first line. You may add one short "
    "reasoning sentence on a second line."
)

USER_TEMPLATE = """ENTITY A
  legal_name:   {a_name}
  jurisdiction: {a_country}
  legal_form:   {a_form}

ENTITY B
  legal_name:   {b_name}
  jurisdiction: {b_country}
  legal_form:   {b_form}

Are these the same underlying economic entity across a possible lifecycle event?"""

def parse_response(text):
    """Return (score in [0,1], raw_text). Score=1 for YES, 0 for NO."""
    if not text:
        return 0.0, ""
    line = text.strip().splitlines()[0].strip().upper().rstrip(".:!,;")
    if line.startswith("YES"):
        return 1.0, text.strip()
    if line.startswith("NO"):
        return 0.0, text.strip()
    return 0.5, text.strip()

def call_anthropic(client, model, system, user):
    msg = client.messages.create(
        model=model, max_tokens=100, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text if msg.content else ""

def call_openai(client, model, system, user):
    msg = client.chat.completions.create(
        model=model, max_tokens=100,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return msg.choices[0].message.content or ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--sample-frac", type=float, default=0.05,
                    help="fraction of pairs to score; 1.0 = full run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-name", default="llm_pairwise")
    ap.add_argument("--qps", type=float, default=4.0, help="approx requests/sec cap")
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
    else:
        try:
            from openai import OpenAI
        except ImportError:
            print("pip install openai", file=sys.stderr); sys.exit(1)
        if not os.environ.get("OPENAI_API_KEY"):
            print("set OPENAI_API_KEY", file=sys.stderr); sys.exit(1)
        client = OpenAI()
        call = lambda s, u: call_openai(client, args.model, s, u)

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
    ent = ent[ent["lei"].isin(lei_set)]
    ent_map = ent.set_index("lei").to_dict(orient="index")

    out_dir = os.path.join(args.data, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.out_name}.parquet")
    log_path = os.path.join(args.data, f"{args.out_name}_log.jsonl")

    print(f"writing predictions -> {out_path}", file=sys.stderr)
    print(f"writing prompt/response log -> {log_path}", file=sys.stderr)
    print(f"~est cost: {len(pairs)} calls; budget accordingly", file=sys.stderr)

    sleep_s = 1.0 / args.qps if args.qps > 0 else 0
    rows = []
    t0 = time.time()
    with open(log_path, "w") as logf:
        for i, r in pairs.iterrows():
            a = ent_map.get(r["anchor_lei"], {})
            b = ent_map.get(r["candidate_lei"], {})
            if not a or not b:
                rows.append((r["anchor_lei"], r["candidate_lei"], 0.0))
                continue
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
                print(f"  error at {i}: {e}", file=sys.stderr)
                score, raw = 0.0, f"ERROR: {e}"

            rows.append((r["anchor_lei"], r["candidate_lei"], score))
            json.dump({"i": int(i), "pred": r["anchor_lei"], "cand": r["candidate_lei"],
                       "label": int(r["label"]), "score": score,
                       "raw": raw[:300]}, logf)
            logf.write("\n")

            if (i + 1) % 50 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i+1:,}/{len(pairs):,} ({rate:.1f} req/s)", file=sys.stderr)
            if sleep_s:
                time.sleep(sleep_s)

    out = pd.DataFrame(rows, columns=["anchor_lei", "candidate_lei", "score"])
    out.to_parquet(out_path, compression="snappy", index=False)
    print(f"done -> {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
