"""Golden snapshot test for report generation (SPEC §11.4, FR-REP-4)."""

from pathlib import Path

from ragstress.report.generator import generate_report


def test_golden_report_snapshot():
    """Verify that fixture run generates byte-identical report matching committed golden snapshot."""
    golden_dir = Path(__file__).resolve().parents[1] / "golden"
    fixture_run = golden_dir / "fixture_run"
    golden_md_file = golden_dir / "report.md"

    assert golden_md_file.is_file(), f"Missing golden snapshot file at {golden_md_file}"

    generated_md = generate_report(fixture_run)
    expected_md = golden_md_file.read_text(encoding="utf-8")

    assert generated_md == expected_md, (
        "Report output differed from committed tests/golden/report.md snapshot."
    )
