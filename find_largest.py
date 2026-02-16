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
from dataclasses import dataclass
from pathlib import Path

import pdfplumber


# Single source of truth for multiplier words and their values
MULTIPLIERS = {
    "thousand": 1_000,
    "thousands": 1_000,
    "million": 1_000_000,
    "millions": 1_000_000,
    "billion": 1_000_000_000,
    "billions": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "trillions": 1_000_000_000_000,
}

_inline_mult_words = "|".join(MULTIPLIERS.keys())
_qualifier_mult_words = "|".join(w for w in MULTIPLIERS.keys() if w.endswith("s"))

# Pattern for inline numbers with optional currency, commas, decimals,
# parenthesized negatives, and optional inline multiplier suffix.
NUMBER_PATTERN = re.compile(
    r"(\()?"                                  # group 1: open paren (negative)
    r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"  # group 2: number
    r"(\))?"                                  # group 3: close paren
    r"(?:"
    r"\s*(%)"                                 # group 4: percent sign
    r"|"
    rf"\s+({_inline_mult_words})"            # group 5: word multiplier (requires space)
    r"(?![a-zA-Z])"
    r")?"
)

QUALIFIER_PATTERN = re.compile(
    rf"\(?\s*(\$|dollars?)?\s*(?:amounts\s+)?in\s+({_qualifier_mult_words})"
    r"(?:\s+of\s+dollars?)?\s*\)?",
    re.IGNORECASE,
)


@dataclass
class CandidateNumber:
    raw_value: float
    adjusted_value: float
    multiplier_label: str = "1x"  # e.g. "millions", "1x", "B"
    page: int = 0
    context: str = ""  # surrounding text snippet


# Financial keywords that indicate a row contains monetary values
FINANCIAL_KEYWORDS = {
    'revenue', 'cost', 'budget', 'appropriation', 'expense', 'spending',
    'income', 'profit', 'loss', 'price', 'fee', 'payment', 'debt',
    'investment', 'funding', 'allocation', 'obligation', 'outlay',
    'dollar', '$', 'sales', 'earnings', 'capital', 'liability', 'asset',
    'expenditure', 'receipt', 'surplus', 'deficit', 'balance'
}


@dataclass
class TableRow:
    """Represents a table row with spatial and content information."""
    text: str  # Full row text
    label: str  # Row label (first cell)
    values: list[str]  # Data values (remaining cells)
    indent_level: float  # X-coordinate of leftmost text (for indentation detection)
    is_financial: bool = False  # Whether this row contains financial data


def extract_pages(pdf_path: str) -> list[tuple[int, str, list[list[TableRow]]]]:
    """
    Extract text and tables from each page of a PDF.

    Returns a list of (page_number, body_text, tables) tuples.
    page_number is 1-indexed. tables is a list of TableRow lists.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = []
            try:
                # Use more aggressive table detection settings
                table_settings = {
                    "vertical_strategy": "text",  # Detect columns by text alignment
                    "horizontal_strategy": "text",  # Detect rows by text alignment
                    "intersection_tolerance": 3,
                }
                raw_tables = page.extract_tables(table_settings) or []
            except Exception:
                # Fallback to default if settings fail
                try:
                    raw_tables = page.extract_tables() or []
                except Exception:
                    raw_tables = []

            for table in raw_tables:
                if not table:
                    continue

                # Extract table with spatial information
                table_rows = []
                for row in table:
                    if not row or not any(cell and cell.strip() for cell in row):
                        continue

                    # First cell is typically the row label
                    label = (row[0] or "").strip()
                    values = [cell.strip() if cell else "" for cell in row[1:]]

                    # Get indent level by analyzing the label text position
                    # We'll use a simple heuristic: count leading spaces
                    indent_level = len(label) - len(label.lstrip())

                    table_rows.append(TableRow(
                        text=" ".join(cell or "" for cell in row),
                        label=label,
                        values=values,
                        indent_level=indent_level,
                        is_financial=False  # Will be determined later
                    ))

                if table_rows:
                    # Determine which rows are financial based on hierarchy
                    mark_financial_rows(table_rows)
                    tables.append(table_rows)

            pages.append((i, text, tables))
    return pages


def has_financial_keyword(text: str) -> bool:
    """Check if text contains any financial keywords."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in FINANCIAL_KEYWORDS)


def mark_financial_rows(rows: list[TableRow]) -> None:
    """
    Mark rows as financial based on indentation hierarchy.

    Rules:
    1. If a row has financial keywords, it's financial
    2. If a row is indented under a financial row, it's financial
    3. Non-indented rows without financial keywords reset the context
    """
    if not rows:
        return

    # Track the current financial context with indent level
    financial_context_stack = []  # List of (indent_level, is_financial) tuples

    for row in rows:
        # Check if this row has explicit financial keywords
        has_fin_keyword = has_financial_keyword(row.label)

        # Pop context stack if we've outdented (returned to a shallower level)
        while financial_context_stack and financial_context_stack[-1][0] >= row.indent_level:
            financial_context_stack.pop()

        # Determine if this row is financial
        if has_fin_keyword:
            # Explicitly financial
            row.is_financial = True
            financial_context_stack.append((row.indent_level, True))
        elif financial_context_stack and financial_context_stack[-1][1]:
            # Indented under a financial row - inherit financial context
            row.is_financial = True
            financial_context_stack.append((row.indent_level, True))
        else:
            # Not financial
            row.is_financial = False
            financial_context_stack.append((row.indent_level, False))


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


def get_context_snippet(text: str, start: int, end: int, window: int = 60) -> str:
    """
    Extract a snippet of text around the match for display.

    Args:
        text: The full text to extract from
        start: Start position of the match
        end: End position of the match
        window: Number of characters to include before and after the match (default: 60)

    Returns:
        A context snippet with ellipsis indicators if text was truncated.
    """
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
    mult = MULTIPLIERS.get(suffix)
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

    Args:
        text: The text to search for numbers
        page_num: Page number for tracking where the number was found
        page_multiplier: Fallback multiplier if qualifiers list is not provided
        page_mult_label: Fallback label if qualifiers list is not provided
        is_dollar_scoped: If True, only apply the multiplier to numbers with decimals
        qualifiers: List of tuples for positional lookup. Each tuple contains:
            - position (int): Start index of the qualifier in the text
            - multiplier (float): The multiplier value (e.g., 1000000 for "millions")
            - label (str): The label string (e.g., "millions")
            - is_dollar_scoped (bool): Whether this qualifier only applies to dollar amounts

    Returns:
        List of CandidateNumber objects found in the text.
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


def detect_all_qualifiers(text: str) -> list[tuple[int, float, str, bool]]:
    """
    Find all qualifiers in text and return (position, multiplier, label, is_dollar_scoped).
    Position is the start index of the qualifier match.
    """
    qualifiers = []
    for match in QUALIFIER_PATTERN.finditer(text):
        dollar_indicator = match.group(1)
        word = match.group(2).lower()
        mult = MULTIPLIERS.get(word, 1.0)
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

    for page_num, body_text, tables in pages:
        # Process each table using spatial hierarchy
        for table_rows in tables:
            # Detect table-level qualifier
            table_text = "\n".join(row.text for row in table_rows)
            t_mult, t_label, t_dollar = detect_page_qualifier(table_text)

            # Process each row
            for row in table_rows:
                # Only apply multiplier if row is marked as financial
                if row.is_financial and t_dollar:
                    # Financial row - apply multiplier
                    row_candidates = extract_numbers(
                        row.text, page_num, t_mult, t_label, is_dollar_scoped=False
                    )
                elif not t_dollar:
                    # Non-dollar-scoped qualifier (like plain "in millions") - apply to all
                    row_candidates = extract_numbers(
                        row.text, page_num, t_mult, t_label, is_dollar_scoped=False
                    )
                else:
                    # Dollar-scoped but not financial row - no multiplier
                    row_candidates = extract_numbers(
                        row.text, page_num, 1.0, "", is_dollar_scoped=False
                    )
                all_candidates.extend(row_candidates)

        # Process body text with positional qualifiers
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
