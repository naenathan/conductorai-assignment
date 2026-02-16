#!/usr/bin/env python3
"""
PDF Largest Number Finder

Parses a PDF document to find the largest numerical value, both as a raw number
and adjusted for natural language context (e.g., "in millions").

Usage: python find_largest.py <path_to_pdf>
"""

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INLINE_MULTIPLIERS = {
    "thousand": 1_000,
    "thousands": 1_000,
    "million": 1_000_000,
    "millions": 1_000_000,
    "billion": 1_000_000_000,
    "billions": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "trillions": 1_000_000_000_000,
}

# Pattern for inline numbers with optional currency, commas, decimals,
# parenthesized negatives, and optional inline multiplier suffix.
#
# Groups:
#   1 - open paren (negative indicator)
#   2 - the numeric part (digits, commas, optional decimal)
#   3 - close paren
#   4 - percent sign (if present, skip multiplier adjustment)
#   5 - inline multiplier word
NUMBER_PATTERN = re.compile(
    r"(?<!\d[/\-])"                           # not preceded by digit+slash/dash (date guard)
    r"(?<!\d\.)"                              # not preceded by digit+dot (version guard)
    r"\$?\s*"                                 # optional currency prefix
    r"(\()?"                                  # group 1: open paren (negative)
    r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"  # group 2: number
    r"(\))?"                                  # group 3: close paren
    r"(?:"
    r"\s*(%)"                                 # group 4: percent sign
    r"|"
    r"\s+(million|millions|billion|billions|thousand|thousands|trillion|trillions)"  # group 5: word multiplier (requires space)
    r"(?![a-zA-Z])"
    r")?"
)

# Section/table qualifier: "(in millions)", "($ in thousands)", etc.
QUALIFIER_PATTERN = re.compile(
    r"\(?\s*(\$|dollars?)?\s*(?:amounts\s+)?in\s+(thousands|millions|billions|trillions)"
    r"(?:\s+of\s+dollars?)?\s*\)?",
    re.IGNORECASE,
)

QUALIFIER_WORD_TO_MULT = {
    "thousands": 1_000,
    "millions": 1_000_000,
    "billions": 1_000_000_000,
    "trillions": 1_000_000_000_000,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CandidateNumber:
    raw_value: float
    adjusted_value: float
    multiplier_label: str = "1x"  # e.g. "millions", "1x", "B"
    page: int = 0
    context: str = ""  # surrounding text snippet


# ---------------------------------------------------------------------------
# Text Extraction
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: str) -> list[tuple[int, str, list[str]]]:
    """
    Extract text and tables from each page of a PDF.

    Returns a list of (page_number, body_text, table_texts) tuples.
    page_number is 1-indexed. table_texts is a list of text strings,
    one per table found on the page.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            table_texts = []
            try:
                raw_tables = page.extract_tables() or []
            except Exception:
                raw_tables = []
            for table in raw_tables:
                rows = []
                for row in table:
                    if row:
                        rows.append(" ".join(cell or "" for cell in row))
                if rows:
                    table_texts.append("\n".join(rows))
            pages.append((i, text, table_texts))
    return pages


def try_fitz_fallback(pdf_path: str, page_numbers: list[int]) -> dict[int, str]:
    """
    Fallback extraction using PyMuPDF for pages that yielded no text.
    Returns {page_number: text}.
    """
    results = {}
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for pn in page_numbers:
            idx = pn - 1
            if 0 <= idx < len(doc):
                results[pn] = doc[idx].get_text() or ""
        doc.close()
    except ImportError:
        pass
    return results


# ---------------------------------------------------------------------------
# Number Extraction
# ---------------------------------------------------------------------------

def get_context_snippet(text: str, start: int, end: int, window: int = 60) -> str:
    """Extract a snippet of text around the match for display."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    snippet = text[ctx_start:ctx_end].replace("\n", " ").strip()
    if ctx_start > 0:
        snippet = "..." + snippet
    if ctx_end < len(text):
        snippet = snippet + "..."
    return snippet


def resolve_inline_multiplier(suffix: str) -> tuple[float, str]:
    """Convert an inline multiplier suffix to (multiplier_value, label)."""
    if not suffix:
        return 1.0, ""
    mult = INLINE_MULTIPLIERS.get(suffix)
    if mult:
        return float(mult), suffix
    return 1.0, ""


def extract_numbers(
    text: str,
    page_num: int,
    page_multiplier: float = 1.0,
    page_mult_label: str = "",
    is_dollar_scoped: bool = False,
    qualifiers: list[tuple[int, float, str, bool]] | None = None,
) -> list[CandidateNumber]:
    """
    Extract all candidate numbers from a text segment.

    page_multiplier and page_mult_label are fallback qualifiers if qualifiers list is not provided.
    is_dollar_scoped: if True, only apply the multiplier to numbers with decimals.
    qualifiers: list of (position, multiplier, label, is_dollar_scoped) for positional lookup.
    """
    candidates = []

    for match in NUMBER_PATTERN.finditer(text):
        open_paren = match.group(1)
        num_str = match.group(2)
        close_paren = match.group(3)
        percent = match.group(4)
        inline_mult_str = match.group(5)

        # Parse the numeric value
        cleaned = num_str.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue

        # Handle parenthesized negatives
        if open_paren and close_paren:
            value = -value

        # Determine multiplier
        is_percent = bool(percent)

        if inline_mult_str:
            mult_val, mult_label = resolve_inline_multiplier(inline_mult_str)
        elif is_percent:
            # Percentages don't get page-level multiplier
            mult_val, mult_label = 1.0, ""
        else:
            # Use positional or fallback qualifier
            if qualifiers:
                mult_val, mult_label, qual_dollar_scoped = get_nearest_qualifier(
                    match.start(2), qualifiers
                )
            else:
                mult_val = page_multiplier
                mult_label = page_mult_label
                qual_dollar_scoped = is_dollar_scoped

            # If qualifier is dollar-scoped and this is a whole integer, don't apply multiplier
            if qual_dollar_scoped and "." not in num_str:
                mult_val, mult_label = 1.0, ""

        adjusted = value * mult_val
        context = get_context_snippet(text, match.start(), match.end())

        candidates.append(CandidateNumber(
            raw_value=value,
            adjusted_value=adjusted,
            multiplier_label=mult_label or "1x",
            page=page_num,
            context=context,
        ))

    return candidates


# ---------------------------------------------------------------------------
# Qualifier Detection
# ---------------------------------------------------------------------------

def detect_all_qualifiers(text: str) -> list[tuple[int, float, str, bool]]:
    """
    Find all qualifiers in text and return (position, multiplier, label, is_dollar_scoped).
    Position is the start index of the qualifier match.
    """
    qualifiers = []
    for match in QUALIFIER_PATTERN.finditer(text):
        dollar_indicator = match.group(1)
        word = match.group(2).lower()
        mult = QUALIFIER_WORD_TO_MULT.get(word, 1.0)
        is_dollar_scoped = bool(dollar_indicator) or "dollar" in match.group(0).lower()
        qualifiers.append((match.start(), float(mult), word, is_dollar_scoped))
    return qualifiers


def get_nearest_qualifier(
    match_start: int, qualifiers: list[tuple[int, float, str, bool]]
) -> tuple[float, str, bool]:
    """
    Find the nearest qualifier that appears before match_start.
    Returns (multiplier, label, is_dollar_scoped).
    """
    # Filter to qualifiers that appear before the number
    preceding = [q for q in qualifiers if q[0] < match_start]
    if not preceding:
        return 1.0, "", False
    # Take the closest one (highest position)
    _, mult, label, is_dollar_scoped = max(preceding, key=lambda q: q[0])
    return mult, label, is_dollar_scoped


def detect_page_qualifier(text: str) -> tuple[float, str, bool]:
    """
    Detect a section/table qualifier, e.g. "(in millions)" or "(Dollars in Millions)".

    Returns (multiplier_value, label_string, is_dollar_scoped).
    is_dollar_scoped is True when the qualifier explicitly references dollars,
    meaning the multiplier should only apply to dollar amounts (numbers with decimals).

    This is a convenience wrapper that returns the first qualifier found.
    """
    qualifiers = detect_all_qualifiers(text)
    if qualifiers:
        _, mult, label, is_dollar_scoped = qualifiers[0]
        return mult, label, is_dollar_scoped
    return 1.0, "", False


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def find_largest_numbers(pdf_path: str) -> tuple[CandidateNumber | None, CandidateNumber | None]:
    """
    Main pipeline: extract → parse → qualify → find max.

    Returns (max_raw_candidate, max_adjusted_candidate).
    """
    pages = extract_pages(pdf_path)

    # Identify empty pages for fallback
    empty_pages = [pn for pn, text, _ in pages if not text.strip()]
    if empty_pages:
        fallback = try_fitz_fallback(pdf_path, empty_pages)
        pages = [
            (pn, fallback.get(pn, text) if not text.strip() else text, table_texts)
            for pn, text, table_texts in pages
        ]

    all_candidates = []

    for page_num, body_text, table_texts in pages:
        # Process each table separately with its own qualifier
        for table_text in table_texts:
            if not table_text.strip():
                continue
            t_mult, t_label, t_dollar = detect_page_qualifier(table_text)
            all_candidates.extend(
                extract_numbers(table_text, page_num, t_mult, t_label, t_dollar)
            )

        # Process body text with positional qualifiers to handle multiple sections
        if body_text.strip():
            qualifiers = detect_all_qualifiers(body_text)
            all_candidates.extend(
                extract_numbers(body_text, page_num, qualifiers=qualifiers)
            )

    if not all_candidates:
        return None, None

    max_raw = max(all_candidates, key=lambda c: c.raw_value)
    max_adjusted = max(all_candidates, key=lambda c: c.adjusted_value)

    return max_raw, max_adjusted


def format_number(value: float) -> str:
    """Format a number with commas and up to 2 decimal places."""
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def main():
    if len(sys.argv) != 2:
        print("Usage: python find_largest.py <path_to_pdf>", file=sys.stderr)
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    if not pdf_path.lower().endswith(".pdf"):
        print("Warning: File does not have .pdf extension", file=sys.stderr)

    start_time = time.time()

    max_raw, max_adjusted = find_largest_numbers(pdf_path)

    elapsed = time.time() - start_time

    if max_raw is None:
        print("No numbers found in the document.")
        sys.exit(0)

    print(f"Largest raw number: {format_number(max_raw.raw_value)} (found on page {max_raw.page})")
    print(f"  Context: {max_raw.context}")
    print()
    print(
        f"Largest adjusted number: {format_number(max_adjusted.adjusted_value)} "
        f"(raw: {format_number(max_adjusted.raw_value)}, "
        f"multiplier: {max_adjusted.multiplier_label}, "
        f"found on page {max_adjusted.page})"
    )
    print(f"  Context: {max_adjusted.context}")
    print()
    print(f"Completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
