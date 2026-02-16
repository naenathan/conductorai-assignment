# PDF Largest Number Finder

Parses a PDF document to find the largest numerical value, both as a raw number and adjusted for natural language context (e.g., "in millions").

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```bash
python find_largest.py <path_to_pdf>
```

### Example Output

```
Largest raw number: 1,234,567 (found on page 42)
  Context: ...total assets of $1,234,567 as reported in...

Largest adjusted number: 3,150,000,000 (raw: 3.15, multiplier: billions, found on page 78)
  Context: ...revenue reached $3.15 billion during the...

Completed in 2.34s
```

## Approach

The tool uses a four-stage pipeline:

1. **Text Extraction** — Uses `pdfplumber` to extract text and tables page-by-page. Falls back to `PyMuPDF` for pages that yield no text (e.g., scanned images without OCR).

2. **Number Extraction** — Regex-based extraction handles integers, decimals, comma-separated values, parenthesized negatives (financial convention), currency prefixes, and inline multiplier suffixes (`$1.2M`, `3.5 billion`). Filters out dates/years, page numbers, phone numbers, zip codes, and footnote references.

3. **Qualifier Detection** — Detects page-level multiplier context like "(in millions)" or "($ in thousands)" and applies it to numbers on that page. Inline multipliers always take priority over page-level qualifiers. Percentages are never multiplied.

4. **Aggregation** — Tracks two running maximums (raw and adjusted) and reports both with page number and surrounding context for manual verification.

## Known Limitations

- Assumes US number formatting (commas as thousands separators).
- Does not perform OCR on image-only pages (would require `pytesseract` + `pdf2image`).
- Page-level qualifiers are not carried across pages.
- Year detection uses heuristics and may occasionally misclassify a number near 1900-2100.
