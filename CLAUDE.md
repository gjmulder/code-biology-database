# AGENTS.md

## Project Context
This project processes Code Biology data from `biological_codes.csv` (derived from `Biological_Code_List_20260531.pdf`).
- **Expected Code Categories:** 435
- **Expected References:** 2299

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
