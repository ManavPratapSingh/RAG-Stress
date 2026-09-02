import pytest

from ragstress.graders.normalize import normalize_value


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1200", "1200"),
        ("1,200", "1200"),
        ("1200 requests/s", "1200"),
        ("50,000", "50000"),
        ("75 °C", "75"),
        ("40 floors", "40"),
        ("1.2k", None),
        ("about 1200", None),
        ("~1200", None),
        ("approx 500", None),
        ("1200.5", None),
    ],
)
def test_normalize_int(raw: str, expected: str | None) -> None:
    assert normalize_value(raw, "int") == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("$1,500", "1500.00"),
        ("1500 USD", "1500.00"),
        ("1,500.00", "1500.00"),
        ("$1,500.00 USD", "1500.00"),
        ("50.00", "50.00"),
        ("20000.00", "20000.00"),
        ("1.5k", None),
        ("~$1,500", None),
        ("about $1500", None),
    ],
)
def test_normalize_money(raw: str, expected: str | None) -> None:
    assert normalize_value(raw, "money") == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2024-03-01", "2024-03-01"),
        ("March 1, 2024", "2024-03-01"),
        ("1 Mar 2024", "2024-03-01"),
        ("15 October 2023", "2023-10-15"),
        ("03/01/2024", None),  # Ambiguous slash formats rejected per SPEC §8.1
        ("about March 2024", None),
    ],
)
def test_normalize_date(raw: str, expected: str | None) -> None:
    assert normalize_value(raw, "date") == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("30 days", "30 days"),
        ("1 day", "1 days"),
        ("6 months", "6 months"),
        ("1 month", "1 months"),
        ("24 months", "24 months"),
        ("half a year", None),
        ("0.5 years", None),
        ("about 30 days", None),
    ],
)
def test_normalize_duration(raw: str, expected: str | None) -> None:
    assert normalize_value(raw, "duration") == expected


@pytest.mark.parametrize(
    "raw,allowed,expected",
    [
        ("low", ["low", "medium", "high", "restricted"], "low"),
        ("LOW", ["low", "medium", "high", "restricted"], "low"),
        ("Restricted", ["low", "medium", "high", "restricted"], "restricted"),
        ("secure", ["low", "medium", "high", "restricted"], None),  # Synonym rejected
    ],
)
def test_normalize_enum(raw: str, allowed: list[str], expected: str | None) -> None:
    assert normalize_value(raw, "enum", allowed_enums=allowed) == expected
