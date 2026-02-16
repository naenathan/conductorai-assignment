PDF Largest Number Finder — Implementation Plan
Objective
Build a Python CLI tool that takes a PDF file path as input and outputs:
1. The largest raw number found in the document
2. The largest adjusted number found, where adjustment means applying natural language multipliers (e.g., "in millions") to raw values
Constraints
* Python 3.10+
* No external API calls. Everything runs locally.
* Must complete in under 60 seconds on a 115-page PDF.
* Dependencies must be installable via pip.
Recommended Dependencies
* pdfplumber — primary text and table extraction
* PyMuPDF (fitz) — fallback extraction, image detection
* pytesseract + pdf2image — OCR for image-based pages (optional, include if time allows)
* re — regex for number and qualifier extraction
Architecture
The solution should be a single Python file (e.g., find_largest.py) with a clear pipeline:
PDF → Page-by-page text extraction → Number extraction → Qualifier detection → Adjustment → Output max 
Step 1: Text Extraction
* Use pdfplumber to extract text from each page.
* Preserve page boundaries (track which page each piece of text came from).
* For each page, extract both raw text (page.extract_text()) and attempt table extraction (page.extract_tables()).
* If a page yields no text (possible scanned image), flag it. Optionally fall back to OCR via pytesseract.
Step 2: Number Extraction
Write a function that takes a string and returns all candidate numbers with their position/context.
Number formats to handle:
* Integers: 1234, 1,234,567
* Decimals: 3.15, 1,234.56
* Parenthesized negatives (financial): (3.15) → treat as -3.15
* Numbers with inline multiplier suffixes: $1.2M, 3.5B, 100K, $2.5 million, 1.2 billion
* Currency prefixed: $1,234.56
Regex pattern (starting point):
# Matches numbers with optional currency prefix, commas, decimals, and optional suffix pattern = r'[\$]?\(?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?(?:\s*(?:million|billion|thousand|trillion|[MBKTmkbt]))?' 
Refine this to capture groups for: the numeric value, whether it's negative (parenthesized), and any inline multiplier.
Numbers to EXCLUDE:
* Dates: years like 2023, 2024, and full dates like 12/31/2023. Heuristic: 4-digit numbers between 1900-2100 appearing near date-like context (months, slashes). Be conservative — if it looks like a year, exclude it.
* Page numbers (typically small integers appearing alone at top/bottom of page text)
* Footnote references (superscript-style small integers)
* Phone numbers (patterns like XXX-XXX-XXXX)
* Zip codes (5-digit numbers near addresses)
* Percentages: numbers immediately followed by % — these should NOT receive multiplier adjustments but can still compete as raw values
Step 3: Qualifier/Multiplier Detection
For each page, detect if there is a multiplier context that applies to numbers on that page.
Multiplier keywords and their values:
MULTIPLIERS = {     'thousand': 1_000,     'thousands': 1_000,     'million': 1_000_000,     'millions': 1_000_000,     'billion': 1_000_000_000,     'billions': 1_000_000_000,     'trillion': 1_000_000_000_000,     'trillions': 1_000_000_000_000,     'K': 1_000,     'k': 1_000,     'M': 1_000_000,     'm': 1_000_000,     'B': 1_000_000_000,     'b': 1_000_000_000,     'T': 1_000_000_000_000, } 
Detection approach (layered, highest priority first):
1. Inline qualifiers — the multiplier is adjacent to the number (e.g., $1.2 million, 3.5B). These are captured during number extraction. Apply directly to that number only.
2. Table/section qualifiers — search for patterns like (in millions), ($ in thousands), (in billions of dollars) on the same page, above the number's position. Regex:
qualifier_pattern = r'\(?\$?\s*in\s+(thousands|millions|billions|trillions)\s*(?:of\s+dollars)?\s*\)?' 
Apply this multiplier to all numbers on the page that do NOT already have an inline qualifier, UNLESS the number is a percentage.
3. Page header/footer qualifiers — check the first and last few lines of each page's text for multiplier language. Lower confidence; use as fallback.
Resolution rule: Inline qualifier always wins. If no inline qualifier, use the nearest table/section qualifier found on the same page. Do not carry qualifiers across pages unless you find a document-level statement (rare; ignore for v1).
Step 4: Aggregation and Output
Maintain two running maximums:
* max_raw: the largest parsed numeric value before any multiplier adjustment
* max_adjusted: the largest value after applying the appropriate multiplier
For each candidate number:
1. Parse its raw numeric value
2. Determine its multiplier (inline > page-level > 1x default)
3. Compute adjusted = raw * multiplier
4. Update max_raw and max_adjusted
Output format:
Largest raw number: 1,234,567.89 (found on page 42) Largest adjusted number: 3,150,000,000.00 (raw: 3.15, multiplier: billions, found on page 78) 
Include the page number and context (surrounding text snippet) for both, so the result can be manually verified.
Project Structure
pdf-largest-number/ ├── find_largest.py      # Main script ├── requirements.txt     # pdfplumber, PyMuPDF, etc. └── README.md            # Setup and usage instructions 
Keep it to a single Python file unless it gets unwieldy. A few well-named functions are better than a class hierarchy for a project this size.
Edge Cases to Handle
* Tables where some columns are percentages and others are dollar amounts (don't multiply percentages)
* Negative numbers in parentheses — these won't be the largest, but parse them correctly
* Numbers that span formatting (e.g., split across table cells or lines) — pdfplumber's table extraction should help here
* Empty pages or pages with only images
* Commas as thousands separators vs. European decimal commas — assume US formatting (commas = thousands) unless the document indicates otherwise
README Template
# PDF Largest Number Finder  Parses a PDF document to find the largest numerical value, both as a raw number and adjusted for natural language context (e.g., "in millions").  ## Setup # Install uv if you don't have it curl -LsSf https://astral.sh/uv/install.sh | sh  # Create venv and install dependencies uv venv source .venv/bin/activate uv pip install -r requirements.txt  ## Usage python find_largest.py <path_to_pdf>  ## Approach [Brief description of the extraction pipeline, qualifier detection logic, and known limitations]