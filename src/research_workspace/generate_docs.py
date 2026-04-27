"""Generate REPORT.md and README.md from completed experiment outputs."""

from __future__ import annotations

from pathlib import Path

from .reporting import generate_report_markdown


def main() -> None:
    docs = generate_report_markdown()
    Path("REPORT.md").write_text(docs["report"])
    Path("README.md").write_text(docs["readme"])


if __name__ == "__main__":
    main()
