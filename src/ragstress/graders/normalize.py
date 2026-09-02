"""Value normalization for deterministic evaluation (SPEC §8.1)."""

import re
from datetime import datetime
from typing import Literal

MONTH_NAMES = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

DISALLOWED_PREFIXES_REGEX = re.compile(
    r"^(about|approx|approximately|around|nearly|~)\s*", re.IGNORECASE
)
K_NOTATION_REGEX = re.compile(r"^\$?\s*[\d.]+\s*k\b", re.IGNORECASE)


def normalize_value(
    raw_value: str | None,
    value_type: Literal["int", "money", "date", "duration", "enum"],
    allowed_enums: list[str] | set[str] | None = None,
) -> str | None:
    """Normalize a raw string value into its canonical form.

    Returns None if the value cannot be parsed or violates acceptance rules.
    """
    if raw_value is None:
        return None

    cleaned = raw_value.strip()
    if not cleaned:
        return None

    # Reject approximate / estimated prefix qualifiers
    if DISALLOWED_PREFIXES_REGEX.search(cleaned):
        return None

    # Reject shorthand "k" notation (e.g. 1.2k, 5k)
    if K_NOTATION_REGEX.search(cleaned):
        return None

    if value_type == "int":
        return _normalize_int(cleaned)
    elif value_type == "money":
        return _normalize_money(cleaned)
    elif value_type == "date":
        return _normalize_date(cleaned)
    elif value_type == "duration":
        return _normalize_duration(cleaned)
    elif value_type == "enum":
        return _normalize_enum(cleaned, allowed_enums)
    return None


def _normalize_int(val: str) -> str | None:
    # Match integer optionally formatted with commas, followed by optional unit words like requests/s, people, floors, etc.
    # Note: reject floats like 1200.5
    m = re.match(r"^([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\s+[a-zA-Z°/_-]+)*$", val)
    if not m:
        return None
    num_str = m.group(1).replace(",", "")
    try:
        n = int(num_str)
        return str(n)
    except ValueError:
        return None


def _normalize_money(val: str) -> str | None:
    # Match: $1,500, 1500 USD, 1,500.00, $1500.00 USD
    # Reject: $1.5k, ~$1,500
    cleaned = val
    if cleaned.startswith("$"):
        cleaned = cleaned[1:].strip()
    if cleaned.upper().endswith("USD"):
        cleaned = cleaned[:-3].strip()

    m = re.match(r"^([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.([0-9]{1,2}))?$", cleaned)
    if not m:
        return None
    whole_str = m.group(1).replace(",", "")
    dec_str = m.group(2)
    if dec_str is None:
        dec = "00"
    elif len(dec_str) == 1:
        dec = dec_str + "0"
    else:
        dec = dec_str

    try:
        n = int(whole_str)
        return f"{n}.{dec}"
    except ValueError:
        return None


def _normalize_date(val: str) -> str | None:
    # Accepted:
    # 2024-03-01 (ISO)
    # March 1, 2024
    # 1 Mar 2024 / 1 March 2024
    # Rejected:
    # 03/01/2024 (ambiguous day/month)
    cleaned = val.strip().rstrip(".")

    # ISO format YYYY-MM-DD
    m_iso = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", cleaned)
    if m_iso:
        y, m, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        try:
            dt = datetime(y, m, d)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    # Ambiguous slash formats (MM/DD/YYYY or DD/MM/YYYY) are rejected per SPEC §8.1
    if "/" in cleaned:
        return None

    # "Month Day, Year" or "Month Day Year"
    m_mdy = re.match(r"^([a-zA-Z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})$", cleaned)
    if m_mdy:
        mon_str, day_str, yr_str = m_mdy.groups()
        mon = MONTH_NAMES.get(mon_str.lower())
        if not mon:
            return None
        try:
            dt = datetime(int(yr_str), mon, int(day_str))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    # "Day Month Year"
    m_dmy = re.match(r"^(\d{1,2})(?:st|nd|rd|th)?\s+([a-zA-Z]+),?\s+(\d{4})$", cleaned)
    if m_dmy:
        day_str, mon_str, yr_str = m_dmy.groups()
        mon = MONTH_NAMES.get(mon_str.lower())
        if not mon:
            return None
        try:
            dt = datetime(int(yr_str), mon, int(day_str))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    return None


def _normalize_duration(val: str) -> str | None:
    # Accepted: "30 days", "1 month", "6 months", "1 day", etc.
    # Rejected: "half a year", "0.5 years", "2 weeks" (if unit does not match)
    # Canonical: "<n> days" or "<n> months" (no cross-unit conversion)
    m = re.match(r"^(\d+)\s+(day|days|month|months)$", val.strip(), re.IGNORECASE)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    if "day" in unit:
        return f"{n} days"
    elif "month" in unit:
        return f"{n} months"
    return None


def _normalize_enum(val: str, allowed_enums: list[str] | set[str] | None) -> str | None:
    cleaned = val.strip().lower()
    if allowed_enums is not None:
        valid_lower = {e.lower() for e in allowed_enums}
        if cleaned in valid_lower:
            return cleaned
        return None
    return cleaned
