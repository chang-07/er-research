"""Gemini Batch API runner — same pairwise protocol at 50% price, using
the batch quota pool (separate from the interactive RPM/RPD limits that
make a 26K-call interactive run impossible at low access tiers).

Writes into the SAME data/<out-name>_log.jsonl and predictions parquet
as llm_pairwise.py, so already-scored interactive results are never
re-bought and the eval pipeline is unchanged.

Design:
  * chunked submissions (canary first): the small first chunk validates
    auth, request shape, and thinking config for well under $1 before
    any large job is enqueued. Chunks submit sequentially — one active
    job at a time — to stay inside the enqueued-token quota.
  * idempotent: state lives in data/<out-name>_batch_state.json; every
    invocation fetches finished jobs, appends their results to the log,
    submits the next chunk if none is active, and reports. Re-running
    is always safe; nothing is ever double-submitted or double-billed.
  * --watch polls until everything is fetched (safe to Ctrl-C anytime).
  * money rails: cost estimate + confirmation gate before the first
    submission (--yes to skip; --dry-run plans without submitting).

Typical usage:
  python3 src/baselines/llm_pairwise_gemini_batch.py \
      --test-anchors-per-split 2100 --out-name llm_gemini_full --watch

Auth: GOOGLE_API_KEY or GEMINI_API_KEY env var (native endpoint —
new-format AQ. keys work here, unlike the OpenAI-compat endpoint).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

from llm_pairwise import (PROMPT_SYSTEM, build_user_prompt, load_entity_map,
                          load_log_status, parse_response, price_for,
                          rebuild_parquet_from_log, select_pairs)

TERMINAL_FAIL = ("FAILED", "CANCELLED", "EXPIRED")
TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504, 529}


class TransientAPIError(RuntimeError):
    """Quota/overload/network — wait and retry; never kill the watcher.
    (Batch submissions hit RESOURCE_EXHAUSTED when the enqueued-token
    quota hasn't freed up after the previous job; it clears on its own.)"""


def api(base, key, method, path, body=None):
    url = base.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "x-goog-api-key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:800]
        if e.code in TRANSIENT_HTTP:
            raise TransientAPIError(
                f"HTTP {e.code} on {method} {path}: {detail[:300]}") from None
        raise RuntimeError(f"HTTP {e.code} on {method} {path}: {detail}") from None
    except urllib.error.URLError as e:
        raise TransientAPIError(f"network error on {method} {path}: {e}") from None


def build_request(user_text, args):
    gc = {"maxOutputTokens": args.max_output_tokens}
    if args.thinking_level:
        gc["thinkingConfig"] = {"thinkingLevel": args.thinking_level}
    if args.temperature >= 0:
        gc["temperature"] = args.temperature
    return {
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "systemInstruction": {"parts": [{"text": PROMPT_SYSTEM}]},
        "generationConfig": gc,
    }


def extract_text(response_obj):
    """Concatenate non-thought text parts of the first candidate."""
    cands = response_obj.get("candidates") or []
    if not cands:
        return ""
    parts = (cands[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def op_status(op):
    """(done, failed, state_str, error_msg) from a long-running op."""
    meta = op.get("metadata") or {}
    state = str(meta.get("state") or op.get("state") or "")
    if op.get("error"):
        return True, True, state, str(op["error"])[:300]
    if op.get("done") or "SUCCEEDED" in state:
        return True, False, state, ""
    if any(t in state for t in TERMINAL_FAIL):
        return True, True, state, state
    return False, False, state or "PENDING", ""


def extract_inlined(op):
    """Locate the inlinedResponses array wherever the API nests it."""
    resp = op.get("response") or {}
    inl = resp.get("inlinedResponses")
    if isinstance(inl, dict):  # some shapes nest it once more
        inl = inl.get("inlinedResponses")
    return inl or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--test-anchors-per-split", type=int, default=0)
    ap.add_argument("--anchor-file", default="")
    ap.add_argument("--sample-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thinking-level", default="low",
                    choices=["", "low", "high"],
                    help="generationConfig.thinkingConfig.thinkingLevel; "
                         "'' omits the field entirely")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="-1 omits the parameter")
    ap.add_argument("--max-output-tokens", type=int, default=1500)
    ap.add_argument("--canary", type=int, default=500,
                    help="size of the first (validation) chunk")
    ap.add_argument("--chunk-size", type=int, default=13000,
                    help="pairs per subsequent batch job (stays inside "
                         "the preview-model enqueued-token quota)")
    ap.add_argument("--est-output-tokens", type=int, default=150)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--watch", action="store_true",
                    help="keep polling until all chunks are fetched")
    ap.add_argument("--poll-interval", type=int, default=300)
    ap.add_argument("--base-url",
                    default="https://generativelanguage.googleapis.com")
    args = ap.parse_args()

    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key and not args.dry_run:
        print("set GOOGLE_API_KEY (or GEMINI_API_KEY)", file=sys.stderr)
        sys.exit(1)

    log_path = os.path.join(args.data, f"{args.out_name}_log.jsonl")
    out_path = os.path.join(args.data, "predictions", f"{args.out_name}.parquet")
    state_path = os.path.join(args.data, f"{args.out_name}_batch_state.json")

    pairs = select_pairs(args)
    label_map = {(a, c): int(l) for a, c, l in
                 zip(pairs["anchor_lei"], pairs["candidate_lei"], pairs["label"])}
    sel_keys = list(dict.fromkeys(zip(pairs["anchor_lei"], pairs["candidate_lei"])))

    status = load_log_status(log_path)
    done = {k for k, ok in status.items() if ok}
    todo = [k for k in sel_keys if k not in done]
    print(f"selection: {len(sel_keys):,} unique pairs | already scored: "
          f"{len(done & set(sel_keys)):,} | todo: {len(todo):,}", file=sys.stderr)

    # ---- plan (or load) chunks -------------------------------------------
    if os.path.exists(state_path):
        state = json.load(open(state_path))
    else:
        chunks, i, pos = [], 0, 0
        while pos < len(todo):
            size = min(args.canary if i == 0 else args.chunk_size,
                       len(todo) - pos)
            kf = os.path.join(args.data, f"{args.out_name}_batch_chunk{i}.keys")
            with open(kf, "w") as f:
                for a, c in todo[pos:pos + size]:
                    f.write(f"{a}|{c}\n")
            chunks.append({"i": i, "n": size, "keys_file": kf,
                           "batch_name": "", "state": "UNSUBMITTED"})
            pos += size
            i += 1
        state = {"model": args.model, "chunks": chunks}
        json.dump(state, open(state_path, "w"), indent=1)
        print(f"planned {len(chunks)} chunks: "
              f"{[c['n'] for c in chunks]}", file=sys.stderr)

    # ---- cost gate --------------------------------------------------------
    p = price_for(args.model)
    if p:
        per_call = (255 * p[0] * 0.5 + args.est_output_tokens * p[1] * 0.5)
        est = len(todo) * per_call / 1e6
        print(f"est. batch cost for remaining {len(todo):,} pairs: "
              f"~${est:,.2f} USD (50% batch rate, ~{args.est_output_tokens} "
              f"out tok/call)", file=sys.stderr)
    if args.dry_run:
        print("[dry-run] nothing submitted.", file=sys.stderr)
        return
    unsubmitted = [c for c in state["chunks"] if c["state"] == "UNSUBMITTED"]
    if unsubmitted and not args.yes:
        if not sys.stdin.isatty():
            print("ABORT: pass --yes in non-interactive shells.", file=sys.stderr)
            sys.exit(1)
        if input("Proceed with batch submission? [y/N] ").strip().lower() \
                not in ("y", "yes"):
            sys.exit(1)

    ent_map = None  # loaded lazily only when a chunk needs submitting

    def save():
        json.dump(state, open(state_path, "w"), indent=1)

    def fetch_finished(chunk):
        op = api(args.base_url, key, "GET", f"/v1beta/{chunk['batch_name']}")
        finished, failed, st, err = op_status(op)
        if not finished:
            print(f"  chunk {chunk['i']}: {st} "
                  f"({chunk['n']:,} pairs)", file=sys.stderr)
            return False
        if failed:
            chunk["state"] = "FAILED"
            save()
            print(f"  chunk {chunk['i']} FAILED: {err}\n  fix the cause, "
                  f"delete its entry from {state_path} to resubmit.",
                  file=sys.stderr)
            sys.exit(2)
        items = extract_inlined(op)
        keys = [tuple(l.strip().split("|")) for l in open(chunk["keys_file"])
                if l.strip()]
        n_ok = n_err = 0
        with open(log_path, "a", buffering=1) as logf:
            for idx, item in enumerate(items):
                k = (item.get("metadata") or {}).get("key")
                key_t = tuple(k.split("|")) if k else (
                    keys[idx] if idx < len(keys) else None)
                if key_t is None:
                    continue
                if item.get("error"):
                    score, raw = None, f"ERROR: batch: {str(item['error'])[:200]}"
                    n_err += 1
                else:
                    text = extract_text(item.get("response") or {})
                    score, raw = parse_response(text)
                    n_ok += 1
                json.dump({"i": -1, "pred": key_t[0], "cand": key_t[1],
                           "label": label_map.get(key_t, -1),
                           "score": score, "raw": raw[:300]}, logf)
                logf.write("\n")
        chunk["state"] = "FETCHED"
        save()
        print(f"  chunk {chunk['i']} fetched: {n_ok:,} ok, {n_err:,} errored"
              f" (errored pairs can be re-planned later)", file=sys.stderr)
        return True

    def submit(chunk):
        nonlocal ent_map
        now_status = load_log_status(log_path)
        now_done = {k for k, ok in now_status.items() if ok}
        keys = [tuple(l.strip().split("|")) for l in open(chunk["keys_file"])
                if l.strip()]
        keys = [k for k in keys if k not in now_done]
        if not keys:
            chunk["state"] = "FETCHED"
            save()
            return
        if ent_map is None:
            leis = {x for k in keys for x in k}
            ent_map = load_entity_map(args.data, leis)
        reqs = []
        for a, c in keys:
            ea, eb = ent_map.get(a, {}), ent_map.get(c, {})
            reqs.append({"request": build_request(
                build_user_prompt(ea, eb), args),
                "metadata": {"key": f"{a}|{c}"}})
        body = {"batch": {
            "display_name": f"{args.out_name}-chunk{chunk['i']}",
            "input_config": {"requests": {"requests": reqs}}}}
        op = api(args.base_url, key, "POST",
                 f"/v1beta/models/{args.model}:batchGenerateContent", body)
        chunk["batch_name"] = op.get("name", "")
        chunk["state"] = "SUBMITTED"
        save()
        print(f"  chunk {chunk['i']} submitted ({len(reqs):,} pairs) -> "
              f"{chunk['batch_name']}", file=sys.stderr)

    while True:
        progressed = False
        active = None
        try:
            for chunk in state["chunks"]:
                if chunk["state"] == "SUBMITTED":
                    if fetch_finished(chunk):
                        progressed = True
                    else:
                        active = chunk
                    break
            if active is None:
                nxt = next((c for c in state["chunks"]
                            if c["state"] == "UNSUBMITTED"), None)
                if nxt is not None:
                    submit(nxt)
                    progressed = True
        except TransientAPIError as e:
            print(f"  transient API error ({str(e)[:140]}...) — "
                  + ("waiting before retry" if args.watch
                     else "re-run this command later to continue"),
                  file=sys.stderr)
            if not args.watch:
                return
            time.sleep(args.poll_interval)
            continue
        if all(c["state"] == "FETCHED" for c in state["chunks"]):
            n_rows, n_err, n_missing = rebuild_parquet_from_log(
                log_path, set(sel_keys), out_path)
            print(f"\nALL CHUNKS DONE. wrote {n_rows:,} scored pairs -> "
                  f"{out_path}", file=sys.stderr)
            if n_err:
                print(f"  {n_err:,} pairs errored — delete "
                      f"{state_path} and re-run to re-plan just those.",
                      file=sys.stderr)
            return
        if not args.watch:
            print("run this command again to poll / continue "
                  "(or add --watch).", file=sys.stderr)
            return
        if not progressed:
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
