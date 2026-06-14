"""Ad-hoc driver: judge a random sample of N papers through the full pipeline.

Criteria 1&2 → local Gemma (32k batch server); criterion 3 → paid Nemotron via
OpenRouter. Reproducible via --seed. Not part of the test suite.

  python3 -u run_sample.py [--n 10] [--seed 0] [--workers 6]
"""

import argparse
import glob
import logging
import os
import random
import re

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("sample")


def load_env(path=".env"):
    for line in open(path):
        m = re.match(r'\s*(?:export\s+)?([A-Z_]+)\s*=\s*["\']?([^"\'\n]+)', line)
        if m:
            os.environ.setdefault(m.group(1), m.group(2).strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--csv", default="biological_codes.csv")
    ap.add_argument("--pdf-dir", default="pdfs")
    ap.add_argument("--checkpoint", default="sample_verdicts.jsonl")
    args = ap.parse_args()

    load_env()
    import criteria_judge as cj

    papers = list(cj.iter_papers(args.csv, args.pdf_dir))
    log.info("corpus: %d papers with PDFs on disk", len(papers))
    rng = random.Random(args.seed)
    sample = rng.sample(papers, min(args.n, len(papers)))
    log.info("sampled %d papers (seed=%d)", len(sample), args.seed)
    for p in sample:
        log.info("  code %s | %s | %s", p["code_number"],
                 p["code_name"][:40], os.path.basename(p["pdf_path"]))

    lc = cj.local_complete_factory()
    orc = cj.openrouter_complete_factory()

    def judge_fn(paper):
        return cj.judge_paper(paper, lc, orc)

    # fresh run each invocation for a clean sample read-out
    if os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
    records = cj.run_batch(sample, judge_fn, args.checkpoint,
                           max_workers=args.workers)

    print("\n=== PER-PAPER VERDICTS ===")
    for r in sorted(records, key=lambda r: int(r["code_number"])):
        c = r["criteria"]
        qual = cj.paper_qualifies(c)
        def fmt(k):
            v = c.get(k, {})
            tag = v.get("verdict", "?")
            if v.get("grounding_failed"):
                tag += "*"
            return f"{k}={tag}"
        print(f"[{'QUALIFIES' if qual else '         '}] code {r['code_number']:>4} "
              f"{fmt('two_worlds')} {fmt('adaptors')} {fmt('arbitrariness')}  "
              f"| {os.path.basename(r['pdf_path'])}")

    codes = cj.aggregate(records)
    print("\n=== PER-CODE ROLLUP ===")
    for code in sorted(codes, key=int):
        e = codes[code]
        print(f"code {code:>4}: {e['supported']}/{e['total']} papers qualify  "
              f"| {e['code_name'][:50]}")
    print(f"\n{sum(cj.paper_qualifies(r['criteria']) for r in records)}/"
          f"{len(records)} papers meet all three criteria")


if __name__ == "__main__":
    main()
