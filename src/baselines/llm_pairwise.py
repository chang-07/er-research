"""LLM pairwise matcher baseline (OpenSanctions-Pairs-style).

NOT run autonomously — this drives external paid APIs. Run it yourself
when you're ready to spend the API budget.

Protocol (paper §5.1, "LLM baselines"): one API call per (anchor,
candidate) pair, byte-identical prompts across providers,
temperature=0 by default. The model scores the pair on an integer
0-10 rubric (parsed from the response, mapped to [0,1]). Running every
model through this same script is what makes the panel rows directly
comparable.

Money-safety rails:
  * --dry-run          count pairs + print a cost estimate, then exit.
                       Needs no API key.
  * cost gate          before dispatch: prints call count + estimated
                       dollars and asks for confirmation. Skip with
                       --yes; non-interactive runs REQUIRE --yes.
  * incremental log    every result is appended (and flushed) to
                       data/<out-name>_log.jsonl the moment it lands.
  * --resume           skip pairs already scored in the log (errored
                       pairs are re-attempted), then rebuild the final
                       parquet from the log. A run that died at 90%
                       costs ~10% to finish, not 100%.
  * retries + backoff  transient API errors are retried with
                       exponential backoff (--max-retries, default 4),
                       honoring provider Retry-After hints.
  * fail-fast          15 consecutive PERMANENT failures (bad key /
                       bad param / bad model) abort the run. Rate
                       limits, 5xx and network errors NEVER abort a
                       healthy run: they retry indefinitely with
                       adaptive (AIMD) pacing, so an overnight throttle
                       storm or provider outage pauses the run instead
                       of killing it. A fresh run that has never
                       succeeded keeps strict fail-fast, so an endpoint
                       typo still aborts in seconds.
  * errors excluded    pairs that still fail after retries are logged
                       with score=null and EXCLUDED from the output
                       parquet — never silently scored 0.0. The eval
                       scripts drop anchors with incomplete pools.

Typical usage:
  # scope + cost preview (no key needed)
  python3 src/baselines/llm_pairwise.py --dry-run --test-anchors-per-split 2100

  # tiny paid smoke test (~$0.05)
  python3 src/baselines/llm_pairwise.py --test-anchors-per-split 2 \
      --out-name llm_haiku_smoke

  # full eval-coverage run (~26.5K calls; all eligible test anchors)
  python3 src/baselines/llm_pairwise.py --test-anchors-per-split 2100 \
      --out-name llm_haiku_full

Sampling: --test-anchors-per-split N draws N eligible test anchors per
split regime (seeded; N >= the eligible count means full coverage) and
scores ALL their candidate pairs. --sample-frac is a legacy uniform
sample over all 125K pairs INCLUDING train rows — avoid for eval runs.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# ---------------------------------------------------------------------------
# Pricing ($/MTok list price: input, output, cached-read) — verified 2026-07.
# cached-read applies to the Anthropic prompt-cached system prefix (0.1x).
# Used ONLY for the pre-flight estimate; actual billing is the provider's.
# ---------------------------------------------------------------------------
PRICES = {
    "claude-haiku-4-5":  (1.00, 5.00, 0.10),
    "claude-sonnet-4-6": (3.00, 15.00, 0.30),
    "claude-opus-4-7":   (5.00, 25.00, 0.50),
    "gpt-4o-mini":       (0.15, 0.60, None),
    "gpt-4o":            (2.50, 10.00, None),
    "gemini-2.5-flash":  (0.30, 2.50, None),
    "gemini-2.5-pro":    (1.25, 10.00, None),
    # Preview id (verified 2026-07). Thinking tokens bill as output and are
    # NOT in this estimate — validate with a smoke run + console billing
    # check before any full run on a thinking-default model.
    "gemini-3.1-pro":    (2.00, 12.00, None),
    # Verified 2026-07 on api-docs.deepseek.com. Thinking is DEFAULT on
    # V4 — run with --thinking off for protocol-consistent (and cheap)
    # non-reasoning scoring; non-thinking mode supports temperature.
    "deepseek-v4-pro":   (0.435, 0.87, None),
    "deepseek-v4-flash": (0.14, 0.28, None),
}
# Per-call token shape measured on the June 2026 Haiku panel run:
# ~165-token system prefix, ~90 marginal user tokens; output measured at
# ~55 tokens mean / ~75 p95 (models write the score plus 1-2 sentences).
# We estimate with 60; max_tokens=100 puts a hard ceiling ~1.5x above.
TOK_SYSTEM, TOK_USER, TOK_OUT = 165, 90, 60


def price_for(model):
    for key, p in PRICES.items():
        if model.startswith(key):
            return p
    return None


def estimate_usd(n_calls, model, provider):
    """Estimated dollars for n_calls, or None if the model has no list
    price on file.

    NOTE: no prompt-cache discount is applied for Anthropic models even
    though call_anthropic sets cache_control — the minimum cacheable
    prefix is 4096 tokens on Haiku 4.5 / Opus 4.7 (2048 on Sonnet 4.6),
    and our ~165-token system prompt is far below it, so the marker is
    silently inert and the system prefix bills at full input price.
    (Verified empirically on the July 2026 Haiku full run: actual spend
    matched full-price input, not cache-read pricing.)"""
    p = price_for(model)
    if p is None:
        return None
    p_in, p_out, _p_cache = p
    return n_calls * ((TOK_SYSTEM + TOK_USER) * p_in
                       + estimate_usd.out_tokens * p_out) / 1_000_000.0

estimate_usd.out_tokens = TOK_OUT  # overridable via --est-output-tokens


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


# ---------------------------------------------------------------------------
# Provider call wrappers. temperature=None omits the parameter (for models
# that reject it); otherwise it is pinned explicitly — the paper's protocol
# states temperature=0, so the default must not silently fall back to the
# SDK default of 1.0.
# ---------------------------------------------------------------------------
def call_anthropic(client, model, system, user, temperature):
    # cache_control is set but INERT at this prompt size: the minimum
    # cacheable prefix (4096 tokens on Haiku 4.5/Opus 4.7, 2048 on
    # Sonnet 4.6) far exceeds our ~165-token system prompt, so no cache
    # is written or read and the system bills at full input price.
    # Kept in case the prompt is ever enriched past the threshold.
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    msg = client.messages.create(
        model=model,
        max_tokens=100,
        system=[{"type": "text", "text": system,
                  "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        **kwargs,
    )
    return msg.content[0].text if msg.content else ""


def call_openai(client, model, system, user, temperature, max_tokens=100,
                reasoning_effort=None, extra_body=None):
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning_effort:
        # Thinking models (Gemini 2.5/3.x Pro via the OpenAI-compat
        # endpoint, o-series) bill thinking as output tokens; "low" cuts
        # cost ~2.5x. Gemini Pro models reject "none" (thinking-only).
        kwargs["reasoning_effort"] = reasoning_effort
    if extra_body:
        # Provider-specific request fields, e.g. DeepSeek V4's
        # {"thinking": {"type": "disabled"}} (thinking is their default).
        kwargs["extra_body"] = extra_body
    msg = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        **kwargs,
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
    # System instruction and temperature are set at model construction.
    resp = model_obj.generate_content(user)
    return resp.text or ""


class RateLimiter:
    """Shared adaptive dispatch gate across worker threads (0 qps = off).

    AIMD, like TCP: each throttle signal (429/503/529) doubles the
    dispatch interval (capped at 60s); each success decays it back
    toward the configured rate. A fixed-interval gate overshoots
    per-minute quota windows precisely when the provider is already
    throttling — adapting to feedback does not."""

    def __init__(self, qps):
        self.base = (1.0 / qps) if qps and qps > 0 else 0.0
        self.interval = self.base
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self):
        with self.lock:
            iv = self.interval
            if not iv:
                return
            now = time.monotonic()
            t = max(self.next_t, now)
            self.next_t = t + iv
        if t > now:
            time.sleep(t - now)

    def on_throttle(self):
        with self.lock:
            self.interval = min(max(self.interval, self.base, 0.25) * 2.0, 60.0)

    def on_success(self):
        with self.lock:
            if self.interval > self.base:
                self.interval = max(self.base, self.interval * 0.98)


PERMANENT_TYPES = ("BadRequestError", "AuthenticationError", "NotFoundError",
                   "PermissionDeniedError", "UnprocessableEntityError")
TRANSIENT_TYPES = ("RateLimitError", "APIConnectionError", "APITimeoutError",
                   "InternalServerError")
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504, 529}


def _retry_after_s(e):
    headers = getattr(getattr(e, "response", None), "headers", None)
    if headers is not None:
        try:
            return float(headers.get("retry-after") or 0) or None
        except (TypeError, ValueError):
            pass
    return None


def call_with_retry(call, system, user, max_retries, limiter, abort_evt,
                     patient=lambda: False):
    """Error-class-aware retry. Every attempt passes the shared limiter.

    * Permanent 4xx (bad param/model/key): raise immediately — retrying
      can never succeed; fail-fast surfaces the real message.
    * Transient (rate limit, 5xx, network): once the run is healthy
      (patient() True, i.e. at least one success), retry INDEFINITELY
      with escalating sleeps and adaptive rate reduction — an overnight
      throttle storm or provider outage pauses the run, never kills it.
      On a fresh run that has never succeeded (endpoint typo, dead
      port), transient errors stay bounded so fail-fast still aborts.
    * Anything else: bounded by max_retries with jittered backoff.
    """
    attempts = 0
    delay = 1.0
    while True:
        if abort_evt.is_set():
            raise RuntimeError("aborted (fail-fast triggered)")
        limiter.wait()
        try:
            text = call(system, user)
            if not (text or "").strip():
                raise RuntimeError("empty response body")
            limiter.on_success()
            return text
        except Exception as e:
            name = type(e).__name__
            if name in PERMANENT_TYPES:
                raise
            status = getattr(e, "status_code", None)
            transient = name in TRANSIENT_TYPES or status in TRANSIENT_STATUS
            if transient and (name == "RateLimitError"
                               or status in (429, 503, 529)):
                limiter.on_throttle()
            retry_after = _retry_after_s(e)
            if transient and patient():
                time.sleep(min(retry_after or delay, 300.0))
                delay = min(delay * 2.0, 120.0)
                continue  # unlimited patience once the run is healthy
            attempts += 1
            if attempts > max_retries:
                raise
            sleep_s = retry_after or delay * (0.5 + random.random())
            time.sleep(min(sleep_s, 70.0))
            delay = min(delay * 2.0, 32.0)


def load_log_status(log_path):
    """Parse the JSONL log -> {(anchor, cand): ok_bool} (last entry wins).
    ok=True means validly scored (or MISSING_ENTITY, which will never
    succeed); ok=False means an errored attempt that --resume should retry.
    Tolerates a truncated final line from a crash. Also treats legacy
    error rows (score 0.0 with raw 'ERROR: ...') as retryable."""
    status = {}
    if not os.path.exists(log_path):
        return status
    with open(log_path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue  # truncated tail
            key = (e.get("pred"), e.get("cand"))
            raw = str(e.get("raw") or "")
            score = e.get("score")
            if raw.startswith("MISSING_ENTITY"):
                status[key] = True
            elif score is None or raw.startswith("ERROR"):
                status[key] = False
            else:
                status[key] = True
    return status


def select_pairs(args):
    """Pair selection shared by the interactive and batch runners.
    Deterministic for a given (flags, seed): --test-anchors-per-split with
    N >= the eligible count selects ALL eligible anchors, so reruns and
    resumes always target the identical pair set."""
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
        print(f"total pairs selected: {len(pairs):,}", file=sys.stderr)
        return pairs
    if args.test_anchors_per_split > 0:
        # Stratified sampling: N test anchors per split, score ALL their pairs.
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
        pairs = pairs[["anchor_lei", "candidate_lei", "label", "source"]]
        print(f"total pairs selected: {len(pairs):,} "
              f"(across {len(chosen_anchors):,} unique anchors)", file=sys.stderr)
        return pairs
    pairs = pq.read_table(
        os.path.join(args.data, "pairs.parquet"),
        columns=["anchor_lei", "candidate_lei", "label", "source"],
    ).to_pandas()
    print(f"loaded {len(pairs):,} pairs", file=sys.stderr)
    if 0 < args.sample_frac < 1.0:
        pairs = pairs.sample(frac=args.sample_frac, random_state=args.seed).reset_index(drop=True)
        print(f"sampled {len(pairs):,} pairs", file=sys.stderr)
    else:
        print("WARNING: full-pairs run includes TRAIN rows; eval only "
              "needs --test-anchors-per-split coverage.", file=sys.stderr)
    return pairs


def load_entity_map(data, lei_set):
    """lei -> {legal_name, legal_addr_country, legal_form_code} for the
    entities appearing in the selected pairs."""
    ent = pq.read_table(
        os.path.join(data, "entities.parquet"),
        columns=["lei", "legal_name", "legal_addr_country", "legal_form_code"],
    ).to_pandas()
    ent = ent[ent["lei"].isin(lei_set)].drop_duplicates(subset=["lei"], keep="first")
    return ent.set_index("lei").to_dict(orient="index")


def build_user_prompt(a, b, no_think=False):
    """Render the byte-identical user prompt for one (anchor, candidate)
    entity pair — shared by the interactive and batch runners."""
    user = USER_TEMPLATE.format(
        a_name=a.get("legal_name", "") or "(unknown)",
        a_country=a.get("legal_addr_country", "") or "(unknown)",
        a_form=a.get("legal_form_code", "") or "(unknown)",
        b_name=b.get("legal_name", "") or "(unknown)",
        b_country=b.get("legal_addr_country", "") or "(unknown)",
        b_form=b.get("legal_form_code", "") or "(unknown)",
    )
    return user + " /no_think" if no_think else user


def rebuild_parquet_from_log(log_path, sel_set, out_path):
    """Rebuild the predictions parquet from the JSONL log (single source
    of truth), restricted to the current selection, last entry wins.
    Returns (n_rows, n_api_err, n_missing)."""
    best = {}
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                key = (e.get("pred"), e.get("cand"))
                if key in sel_set:
                    best[key] = (e.get("score"), str(e.get("raw") or ""))
    rows = [(k[0], k[1], s) for k, (s, raw) in best.items()
            if s is not None and not raw.startswith("ERROR")]
    n_api_err = sum(1 for s, raw in best.values()
                    if (s is None and not raw.startswith("MISSING_ENTITY"))
                    or raw.startswith("ERROR"))
    n_missing = sum(1 for s, raw in best.values()
                    if raw.startswith("MISSING_ENTITY"))
    out = pd.DataFrame(rows, columns=["anchor_lei", "candidate_lei", "score"])
    out.to_parquet(out_path, compression="snappy", index=False)
    return len(out), n_api_err, n_missing


def main():
    # Shells launching this in the background (&, nohup, CI) often inherit
    # SIGINT as ignored, which would make `kill -INT <pid>` a silent no-op
    # and leave no graceful way to stop a paid run. Re-install the default
    # handler so interrupt -> KeyboardInterrupt -> flush log + resume hint.
    import signal
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except (ValueError, OSError):
        pass  # not the main thread / restricted env — harmless

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001",
                    help="Anthropic: claude-haiku-4-5-20251001 (cheap), "
                         "claude-sonnet-4-6, claude-opus-4-7. OpenAI: gpt-4o, "
                         "gpt-4o-mini. Gemini: gemini-2.5-flash.")
    ap.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "openai", "gemini", "ollama"])
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature, pinned explicitly (paper "
                         "protocol: 0). Pass -1 to omit the parameter for "
                         "models that reject it (claude-opus-4-7, gpt-5 "
                         "family return 400 'temperature is deprecated').")
    ap.add_argument("--sample-frac", type=float, default=0.05,
                    help="LEGACY: fraction of ALL pairs (train included); "
                         "1.0 = full 125K run. Ignored when "
                         "--test-anchors-per-split is set. Avoid for eval.")
    ap.add_argument("--test-anchors-per-split", type=int, default=0,
                    help="if >0, sample N eligible TEST anchors from EACH "
                         "split regime (TIME/EVENT/JURIS) and score ALL "
                         "their candidate pairs. N >= eligible count (e.g. "
                         "2100) = full eval coverage, ~26.5K pairs.")
    ap.add_argument("--anchor-file", default="",
                    help="newline-delimited anchor LEIs; restricts scoring "
                         "to test pairs with these anchors. Overrides "
                         "--test-anchors-per-split.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-name", default="llm_pairwise")
    ap.add_argument("--dry-run", action="store_true",
                    help="print pair counts + cost estimate and exit; "
                         "makes no API calls and needs no key.")
    ap.add_argument("--yes", action="store_true",
                    help="skip the interactive cost-confirmation gate.")
    ap.add_argument("--resume", action="store_true",
                    help="skip pairs already scored in the existing log "
                         "(errored pairs are re-attempted) and append.")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow clobbering an existing log without --resume.")
    ap.add_argument("--max-calls", type=int, default=0,
                    help="hard ceiling on API calls this run; abort before "
                         "dispatch if the selection exceeds it (0 = off).")
    ap.add_argument("--max-retries", type=int, default=4,
                    help="retries per pair on API errors (exp. backoff).")
    ap.add_argument("--qps", type=float, default=4.0,
                    help="client-side request-rate cap shared across all "
                         "workers (0 = off). Fresh/low-tier Anthropic orgs "
                         "(~50 RPM): try 0.7.")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="number of in-flight API requests")
    ap.add_argument("--base-url", default="",
                    help="override endpoint for --provider openai/ollama "
                         "(vLLM, LM Studio, mock). ollama default: "
                         "http://localhost:11434/v1")
    ap.add_argument("--max-tokens", type=int, default=100,
                    help="response token cap. Thinking models (Gemini Pro) "
                         "consume this on reasoning BEFORE the answer — "
                         "use 1500 so the score isn't truncated; unused "
                         "headroom is not billed.")
    ap.add_argument("--reasoning-effort", default="",
                    choices=["", "low", "medium", "high"],
                    help="openai-provider only: reasoning_effort for "
                         "thinking models (Gemini Pro via compat endpoint, "
                         "o-series). 'low' recommended for cost.")
    ap.add_argument("--thinking", default="",
                    choices=["", "on", "off"],
                    help="openai-provider only: explicit thinking toggle "
                         "for DeepSeek V4 (thinking is their DEFAULT — "
                         "pass 'off' for protocol-consistent non-reasoning "
                         "scoring; non-thinking mode supports temperature).")
    ap.add_argument("--est-output-tokens", type=int, default=0,
                    help="override the output-token assumption in the cost "
                         "gate (default 60). Use ~150 for Gemini Pro at "
                         "low reasoning effort (thinking bills as output).")
    ap.add_argument("--no-think", action="store_true",
                    help="for Qwen3 thinking models: append '/no_think' to "
                         "the user prompt (10x speedup; small quality cost).")
    args = ap.parse_args()

    temperature = None if args.temperature < 0 else args.temperature

    # ---- pair selection (no API key needed for any of this) -------------
    pairs = select_pairs(args)

    out_dir = os.path.join(args.data, "predictions")
    out_path = os.path.join(out_dir, f"{args.out_name}.parquet")
    log_path = os.path.join(args.data, f"{args.out_name}_log.jsonl")

    # ---- resume bookkeeping ---------------------------------------------
    sel_keys = list(zip(pairs["anchor_lei"], pairs["candidate_lei"]))
    done = set()
    if args.resume:
        status = load_log_status(log_path)
        done = {k for k, ok in status.items() if ok}
        n_err_prior = sum(1 for ok in status.values() if not ok)
        print(f"resume: {len(done):,} pairs already scored in log"
              + (f", {n_err_prior:,} errored (will retry)" if n_err_prior else ""),
              file=sys.stderr)
    todo_mask = [k not in done for k in sel_keys]
    todo = pairs[todo_mask].reset_index(drop=True)
    n_todo = len(todo)

    # ---- cost estimate + gates ------------------------------------------
    if args.est_output_tokens > 0:
        estimate_usd.out_tokens = args.est_output_tokens
    est = estimate_usd(n_todo, args.model, args.provider)
    if args.provider == "ollama":
        cost_str = "$0.00 (local)"
    elif est is None:
        cost_str = "UNKNOWN (no list price on file for this model)"
    else:
        cost_str = f"~${est:,.2f} USD (list price)"
    print(f"\nmodel: {args.model}  provider: {args.provider}  "
          f"temperature: {temperature}", file=sys.stderr)
    print(f"pairs: {len(pairs):,} selected, {len(done):,} done, "
          f"{n_todo:,} to score", file=sys.stderr)
    print(f"est. cost for this run: {cost_str} "
          f"(assumes ~{estimate_usd.out_tokens} output tok/call; thinking "
          f"models bill reasoning as output — see --est-output-tokens)",
          file=sys.stderr)
    print(f"predictions -> {out_path}\nlog -> {log_path}\n", file=sys.stderr)

    if args.dry_run:
        print("[dry-run] no API calls made.", file=sys.stderr)
        return

    if args.max_calls and n_todo > args.max_calls:
        print(f"ABORT: {n_todo:,} calls exceeds --max-calls "
              f"{args.max_calls:,}.", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(log_path) and not args.resume and not args.overwrite:
        print(f"ABORT: {log_path} already exists (a previous run's paid "
              "results). Pass --resume to continue it, or --overwrite to "
              "discard it.", file=sys.stderr)
        sys.exit(1)

    if n_todo > 0 and not args.yes:
        if not sys.stdin.isatty():
            print("ABORT: refusing to spend without confirmation in a "
                  "non-interactive shell. Re-run with --yes.", file=sys.stderr)
            sys.exit(1)
        answer = input(f"Proceed with {n_todo:,} API calls ({cost_str})? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("aborted.", file=sys.stderr)
            sys.exit(1)

    # ---- provider client (only now do we need a key) ---------------------
    call = None
    if n_todo > 0:
        if args.provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError:
                print("pip install anthropic", file=sys.stderr); sys.exit(1)
            if not os.environ.get("ANTHROPIC_API_KEY"):
                print("set ANTHROPIC_API_KEY", file=sys.stderr); sys.exit(1)
            client = Anthropic()
            call = lambda s, u: call_anthropic(client, args.model, s, u, temperature)
        elif args.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                print("pip install openai", file=sys.stderr); sys.exit(1)
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key and not args.base_url:
                print("set OPENAI_API_KEY", file=sys.stderr); sys.exit(1)
            client = OpenAI(base_url=args.base_url or None,
                            api_key=api_key or "local")
            extra = None
            if args.thinking:
                extra = {"thinking": {"type": "enabled" if args.thinking == "on"
                                               else "disabled"}}
            call = lambda s, u: call_openai(
                client, args.model, s, u, temperature,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort or None,
                extra_body=extra)
        elif args.provider == "ollama":
            # OpenAI-compatible endpoint; no key. max_tokens bumped because
            # thinking models emit a reasoning trace before the answer.
            try:
                from openai import OpenAI
            except ImportError:
                print("pip install openai", file=sys.stderr); sys.exit(1)
            client = OpenAI(base_url=args.base_url or "http://localhost:11434/v1",
                            api_key="ollama")
            call = lambda s, u: call_openai(client, args.model, s, u,
                                            temperature,
                                            max_tokens=max(1500, args.max_tokens))
        else:  # gemini
            try:
                import google.generativeai as genai
            except ImportError:
                print("pip install google-generativeai", file=sys.stderr); sys.exit(1)
            if not os.environ.get("GOOGLE_API_KEY"):
                print("set GOOGLE_API_KEY (from aistudio.google.com)", file=sys.stderr)
                sys.exit(1)
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            gen_cfg = {"max_output_tokens": 100}
            if temperature is not None:
                gen_cfg["temperature"] = temperature
            gem_model = genai.GenerativeModel(
                args.model,
                system_instruction=PROMPT_SYSTEM,
                generation_config=gen_cfg,
            )
            call = lambda s, u: call_gemini(gem_model, s, u)

    # ---- entity metadata --------------------------------------------------
    ent_map = load_entity_map(
        args.data, set(pairs["anchor_lei"]) | set(pairs["candidate_lei"]))
    os.makedirs(out_dir, exist_ok=True)

    # ---- concurrent dispatch ----------------------------------------------
    limiter = RateLimiter(args.qps)
    abort_evt = threading.Event()
    FAILFAST_N = 15
    state = {"consec_fail": 0, "ok": 0, "err": 0}
    state_lock = threading.Lock()
    log_lock = threading.Lock()

    def record(ok):
        with state_lock:
            if ok:
                state["ok"] += 1
                state["consec_fail"] = 0
            else:
                state["err"] += 1
                state["consec_fail"] += 1
                if state["consec_fail"] >= FAILFAST_N:
                    abort_evt.set()

    def score_one(idx, r):
        if abort_evt.is_set():
            return None
        a = ent_map.get(r["anchor_lei"], {})
        b = ent_map.get(r["candidate_lei"], {})
        if not a or not b:
            record(True)  # data gap, not an API failure
            return (idx, r["anchor_lei"], r["candidate_lei"], None,
                    int(r["label"]), "MISSING_ENTITY")
        user = build_user_prompt(a, b, args.no_think)
        try:
            text = call_with_retry(call, PROMPT_SYSTEM, user,
                                    args.max_retries, limiter, abort_evt,
                                    patient=lambda: state["ok"] > 0)
            score, raw = parse_response(text)
            record(True)
        except Exception as e:
            score, raw = None, f"ERROR: {type(e).__name__}: {e}"
            record(False)
        return (idx, r["anchor_lei"], r["candidate_lei"], score,
                int(r["label"]), raw)

    interrupted = False
    t0 = time.time()
    if n_todo > 0:
        completed = 0
        log_mode = "a" if args.resume else "w"
        with open(log_path, log_mode, buffering=1) as logf:
            pool = ThreadPoolExecutor(max_workers=max(1, args.concurrency))
            futures = [pool.submit(score_one, i, r) for i, r in todo.iterrows()]
            try:
                for fut in as_completed(futures):
                    res = fut.result()
                    if res is None:
                        continue  # worker saw the abort flag
                    idx, anc, cand, score, label, raw = res
                    with log_lock:
                        json.dump({"i": int(idx), "pred": anc, "cand": cand,
                                   "label": label, "score": score,
                                   "raw": raw[:300]}, logf)
                        logf.write("\n")
                    completed += 1
                    if completed % 100 == 0:
                        rate = completed / max(time.time() - t0, 1e-9)
                        eta_s = (n_todo - completed) / max(rate, 0.01)
                        print(f"  {completed:,}/{n_todo:,} "
                              f"(ok {state['ok']:,} / err {state['err']:,}, "
                              f"{rate:.1f} req/s, eta {int(eta_s)}s)",
                              file=sys.stderr)
                pool.shutdown(wait=True)
            except KeyboardInterrupt:
                interrupted = True
                print("\ninterrupted — cancelling in-flight requests; "
                      "completed results are already in the log.",
                      file=sys.stderr)
                pool.shutdown(wait=False, cancel_futures=True)

    # ---- rebuild the parquet from the log (single source of truth) --------
    n_rows, n_api_err, n_missing = rebuild_parquet_from_log(
        log_path, set(sel_keys), out_path)

    elapsed = time.time() - t0
    print(f"\nwrote {n_rows:,} scored pairs -> {out_path} "
          f"({elapsed:.1f}s)", file=sys.stderr)
    if n_missing:
        print(f"  {n_missing:,} pairs skipped: entity metadata missing",
              file=sys.stderr)
    if n_api_err:
        print(f"  {n_api_err:,} pairs FAILED after retries (excluded from "
              f"parquet) — re-run with --resume to retry them.",
              file=sys.stderr)
    if abort_evt.is_set():
        print(f"FAIL-FAST: {FAILFAST_N} consecutive failures — check the "
              "API key, credit balance, and endpoint, then re-run with "
              "--resume.", file=sys.stderr)
        sys.exit(2)
    if interrupted:
        print("interrupted — re-run the same command with --resume to "
              "finish the remaining pairs.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
