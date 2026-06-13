# CLAUDE.md

## Project Context
This project processes Code Biology data from `biological_codes.csv` (derived from `Biological_Code_List_20260531.pdf`).
- **Expected Code Categories:** 435
- **Expected References:** 2299

## Current Status
The extraction pipeline is implemented and passing its test suite.

- **`extract_csv.py`** — parses the PDF directly with `pdfplumber` and emits
  `biological_codes.csv` with columns `Code Number, Code Name, Paper Name, URL`.
  Each citation in the source is a hyperlink whose anchor text is the full
  reference, so hyperlink runs (not text splitting) are the extraction anchor.
- **`test_extract.py`** — pytest suite (18 tests) covering code coverage,
  contiguity, known per-code reference counts, URL validity, cross-page
  continuation, and cross-row hyperlink-bleed regressions. Run with `pytest`.
- **Output:** 2290 references across all **435** codes (within tolerance of the
  quoted 2299). One reference (code 352, SeqCode) has citation text but no
  hyperlink, so it carries an empty URL.

### Key implementation notes
- A code's citation list can spill across pages; the "current" code is carried
  over page boundaries since continuation pages lack a left-column number.
- A hyperlink rectangle sits a few points *above* its row's number digit, so row
  bands are shifted up by `ANCHOR_SLACK` to stop a row stealing the next row's
  first link.
- Data-integrity logging: a `(YYYY)` year heuristic cross-checks the text
  citation count against the URL count and logs a WARNING on mismatch.

## AI Goals & Responsibilities
- **Primary Task:** Parse the CSV to process the codes and their associated citations.
- **Specific Extraction:** For every code, parse the references to isolate the **paper name** and map it directly to its corresponding **hyperlink/URL**.
- **Output:** Generate a clean, structured format (e.g., JSON or cleaned CSV) mapping `Code -> Paper Name -> URL`.

## Rules for AI Agents
1. **Data Integrity:** Extract exactly what is in the CSV columns. Do not hallucinate references or URLs.
2. **Data Parsing:** Handle string splitting carefully, as multiple citations and URLs are bundled in single cells.
3. **Libraries:** Default to Python (`pandas`, `re`) for string manipulation and data extraction. 
4. **Error Handling:** Log any code categories where the number of parsed paper names does not match the number of parsed URLs.

## Code development rules
1. **Testing:** For any new functionality or changes to existing functionality always write or expand code using TDD. Write a failing test first, then the feature or change
2. **Language** Always write in pythonic readable python and prefer numpy for data management
3. **Logging** Always use paython logging and choose DEBUG, INFO, levels suitably depending on criticality and informationality of the code
