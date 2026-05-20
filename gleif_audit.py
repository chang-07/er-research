"""
Audit GLEIF Level 1 LEI-CDF for lifecycle event volume.
Tells us whether the lifecycle-aware ER paper is viable.
"""
import sys
import time
from collections import Counter, defaultdict
from lxml import etree

PATH = "/Users/chang/Personal/research/20260519-gleif-concatenated-file-lei2.xml"
NS = "http://www.gleif.org/data/schema/leidata/2016"

def tag(name):
    return f"{{{NS}}}{name}"

def main():
    total = 0
    with_successor = 0
    with_events = 0
    with_completed_event = 0

    event_type_counts = Counter()
    event_type_status_counts = defaultdict(Counter)
    event_year_counts = defaultdict(Counter)  # year -> {type: count}
    successor_examples = []

    t0 = time.time()
    last_report = t0

    context = etree.iterparse(PATH, tag=tag("LEIRecord"), events=("end",))

    for _, rec in context:
        total += 1
        entity = rec.find(tag("Entity"))
        if entity is not None:
            succ = entity.find(tag("SuccessorEntity"))
            if succ is not None:
                with_successor += 1
                if len(successor_examples) < 5:
                    succ_lei = succ.findtext(tag("SuccessorLEI"))
                    name = entity.findtext(tag("LegalName"))
                    successor_examples.append((name, succ_lei))

            events = entity.find(tag("LegalEntityEvents"))
            if events is not None:
                evlist = events.findall(tag("LegalEntityEvent"))
                if evlist:
                    with_events += 1
                    has_completed = False
                    for ev in evlist:
                        etype = ev.findtext(tag("LegalEntityEventType")) or "UNKNOWN"
                        status = ev.get("event_status") or "UNKNOWN"
                        event_type_counts[etype] += 1
                        event_type_status_counts[etype][status] += 1
                        if status == "COMPLETED":
                            has_completed = True
                        eff = ev.findtext(tag("LegalEntityEventEffectiveDate"))
                        if eff and len(eff) >= 4:
                            try:
                                year = int(eff[:4])
                                event_year_counts[year][etype] += 1
                            except ValueError:
                                pass
                    if has_completed:
                        with_completed_event += 1

        # crucial: free memory
        rec.clear()
        # also clear preceding siblings to avoid memory growth
        while rec.getprevious() is not None:
            del rec.getparent()[0]

        if time.time() - last_report > 30:
            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            print(f"  [{elapsed:6.1f}s] {total:>10,} records "
                  f"({rate:>7,.0f}/s) | succ={with_successor:,} events={with_events:,}",
                  flush=True)
            last_report = time.time()

    elapsed = time.time() - t0
    print()
    print("=" * 72)
    print(f"PARSE COMPLETE in {elapsed:.1f}s ({total/elapsed:,.0f} records/s)")
    print("=" * 72)
    print()
    print(f"Total entities                            : {total:>12,}")
    print(f"Entities with SuccessorEntity link        : {with_successor:>12,}  "
          f"({100*with_successor/total:.3f}%)")
    print(f"Entities with any LegalEntityEvents       : {with_events:>12,}  "
          f"({100*with_events/total:.3f}%)")
    print(f"Entities with >=1 COMPLETED event         : {with_completed_event:>12,}  "
          f"({100*with_completed_event/total:.3f}%)")
    print()
    total_events = sum(event_type_counts.values())
    print(f"Total LegalEntityEvent records            : {total_events:>12,}")
    print()
    print("Event types (all statuses):")
    print(f"  {'TYPE':40s} {'COUNT':>10s}  {'% events':>9s}")
    for etype, count in event_type_counts.most_common():
        pct = 100 * count / total_events if total_events else 0
        print(f"  {etype:40s} {count:>10,}  {pct:>8.2f}%")
    print()
    print("Status breakdown for top event types:")
    for etype, _ in event_type_counts.most_common(8):
        statuses = event_type_status_counts[etype]
        parts = ", ".join(f"{s}={c:,}" for s, c in statuses.most_common())
        print(f"  {etype:40s} {parts}")
    print()
    print("Year histogram (effective-date year x event type, last 12 years):")
    current_year = 2026
    years = sorted([y for y in event_year_counts if 2014 <= y <= current_year])
    top_types = [t for t, _ in event_type_counts.most_common(5)]
    header = "  year  " + "  ".join(f"{t[:14]:>14s}" for t in top_types) + f"  {'TOTAL':>10s}"
    print(header)
    for y in years:
        row = event_year_counts[y]
        parts = "  ".join(f"{row.get(t, 0):>14,}" for t in top_types)
        total_y = sum(row.values())
        print(f"  {y}  {parts}  {total_y:>10,}")
    print()
    print("Sample SuccessorEntity links (entity -> successor LEI):")
    for name, succ_lei in successor_examples:
        print(f"  '{name}' -> {succ_lei}")
    print()
    print("=" * 72)
    print("VIABILITY VERDICT (target: >20K labeled lifecycle events for paper)")
    print("=" * 72)
    high_value_types = {"MERGERS_AND_ACQUISITIONS", "CHANGE_LEGAL_NAME",
                        "DEMERGER", "SPINOFF", "DISSOLUTION",
                        "ABSORPTION", "BREAKUP", "CHANGE_LEGAL_FORM"}
    high_value_count = sum(event_type_counts[t] for t in high_value_types if t in event_type_counts)
    print(f"High-signal event count (M&A + rename + spinoff + ...) : {high_value_count:,}")
    print(f"Of which COMPLETED (usable as ground truth)            : "
          f"~{sum(event_type_status_counts[t].get('COMPLETED', 0) for t in high_value_types):,}")
    print()
    if high_value_count >= 20000:
        print(">> GO. Sufficient lifecycle ground truth for a benchmark + method paper.")
    elif high_value_count >= 5000:
        print(">> CAUTION. Enough for a method paper, but the benchmark will be small.")
        print("   Consider supplementing with EDGAR 8-K Item 5.03 + Wikidata.")
    else:
        print(">> NO-GO from GLEIF alone. Need to lean on EDGAR + Wikidata as primary.")

if __name__ == "__main__":
    main()
