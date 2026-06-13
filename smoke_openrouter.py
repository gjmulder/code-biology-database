"""Live smoke test for the OpenRouter Nemotron criterion-3 judge.

Verbose by design: logs the paper, request size, HTTP attempt, latency, and the
parsed/grounded verdict so the end-to-end path is visible. Not part of the
offline suite. Run: python3 -u smoke_openrouter.py [pdf_path]
"""

import glob
import logging
import os
import re
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("smoke")


def load_env(path=".env"):
    for line in open(path):
        m = re.match(r'\s*(?:export\s+)?([A-Z_]+)\s*=\s*["\']?([^"\'\n]+)', line)
        if m:
            os.environ.setdefault(m.group(1), m.group(2).strip())


def main():
    load_env()
    has_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    log.info("OPENROUTER_API_KEY loaded: %s", has_key)

    import pdf_text
    import criteria_judge as cj

    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("pdfs/*.pdf"))[0]
    text = pdf_text.extract_text(path)
    log.info("paper=%s chars=%d est_tokens=%d", os.path.basename(path), len(text),
             pdf_text.estimate_tokens(text))

    # Wrap the OpenRouter complete callable to log request/response sizes + timing.
    base = cj.openrouter_complete_factory()

    def logged_complete(system, user, response_format=None):
        log.info("-> POST %s | system=%dch user=%dch json_mode=%s",
                 cj.OPENROUTER_MODEL, len(system), len(user), response_format is not None)
        t0 = time.time()
        out = base(system, user, response_format=response_format)
        log.info("<- reply in %.1fs | %d chars | head=%r", time.time() - t0, len(out), out[:120])
        return out

    log.info("calling Nemotron for criterion 3 (arbitrariness)...")
    verdicts = cj.judge_criteria(text, logged_complete, cj.CRITERION_3)
    v = verdicts["arbitrariness"]
    log.info("VERDICT verdict=%s confidence=%s grounding_failed=%s",
             v["verdict"], v["confidence"], v.get("grounding_failed", False))
    log.info("quote[:160]=%r", v["evidence_quote"][:160])
    log.info("reasoning[:200]=%r", v["reasoning"][:200])
    log.info("SMOKE TEST OK — auth, JSON parse, grounding gate all ran.")


if __name__ == "__main__":
    main()
