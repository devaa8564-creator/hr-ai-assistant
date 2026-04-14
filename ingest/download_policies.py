"""
Download HR policy documents from HuggingFace and save as PDFs
into the ./policies/ folder, ready for ChromaDB ingestion.

Dataset used: syncora/hr-policies-qa-dataset
  - Multi-turn HR policy Q&A conversations
  - Covers: leave, sick leave, benefits, compliance, recruitment, etc.

Usage:
    python ingest/download_policies.py

After running, execute:
    python ingest/ingest_pdfs.py
"""

import os
import sys
import re
from pathlib import Path
from collections import defaultdict

# ── Deps ──────────────────────────────────────────────────────────────────────
try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Run: pip install datasets")

try:
    from fpdf import FPDF
except ImportError:
    sys.exit("Run: pip install fpdf2")

POLICIES_DIR = Path(os.getenv("POLICIES_DIR", "./policies"))
POLICIES_DIR.mkdir(parents=True, exist_ok=True)

# ── Topic keywords → PDF filename ─────────────────────────────────────────────
TOPIC_MAP = {
    "annual_leave_policy": [
        "annual leave", "holiday", "vacation", "pto", "paid time off",
        "leave entitlement", "days off", "leave balance", "leave remaining",
    ],
    "sick_leave_policy": [
        "sick leave", "sick day", "medical leave", "illness", "unwell",
        "medical certificate", "doctor", "health leave",
    ],
    "maternity_paternity_policy": [
        "maternity", "paternity", "parental leave", "adoption leave",
        "family leave", "baby", "child",
    ],
    "remote_work_policy": [
        "remote work", "work from home", "wfh", "hybrid", "flexible working",
        "telecommute",
    ],
    "performance_policy": [
        "performance review", "appraisal", "kpi", "goal setting",
        "probation", "pip", "performance improvement",
    ],
    "recruitment_policy": [
        "recruitment", "hiring", "interview", "onboarding", "job offer",
        "background check", "reference check",
    ],
    "general_hr_policy": [],   # catch-all
}


def classify_text(text: str) -> str:
    """Return the best-matching topic key for a piece of text."""
    lower = text.lower()
    for topic, keywords in TOPIC_MAP.items():
        if topic == "general_hr_policy":
            continue
        if any(kw in lower for kw in keywords):
            return topic
    return "general_hr_policy"


def extract_conversation_text(row: dict) -> str:
    """
    Pull readable text out of one dataset row.
    The dataset uses a 'conversations' or 'messages' field with
    system/user/assistant turns.
    """
    lines = []

    # Try common field names
    for field in ("conversations", "messages", "conversation"):
        if field in row and row[field]:
            for turn in row[field]:
                role = turn.get("role", turn.get("from", "")).strip().title()
                content = turn.get("content", turn.get("value", "")).strip()
                if content:
                    lines.append(f"{role}: {content}")
            return "\n\n".join(lines)

    # Fallback: join all string values
    for v in row.values():
        if isinstance(v, str) and len(v) > 50:
            lines.append(v)
    return "\n\n".join(lines)


def sanitize(text: str) -> str:
    """Remove characters not supported by Latin-1 / Helvetica."""
    # Replace common Unicode punctuation with ASCII equivalents
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00b7": "*",
        "\u2022": "*", "\u00a0": " ",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Drop any remaining non-latin-1 chars
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def make_pdf(topic: str, sections: list[str], output_path: Path):
    """Write a list of text sections to a nicely formatted PDF."""
    title = sanitize(topic.replace("_", " ").title())

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 30, 100)
    pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    # Subtitle
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(
        0, 8,
        "Company HR Policy Document - Sourced from HuggingFace (syncora/hr-policies-qa-dataset)",
        new_x="LMARGIN", new_y="NEXT", align="C",
    )
    pdf.ln(8)

    # Divider
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # Body sections
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(30, 30, 30)

    for i, section in enumerate(sections, 1):
        # Section header
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(50, 50, 150)
        pdf.cell(0, 8, f"Section {i}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        # Section body
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)

        # Clean and wrap text
        clean = sanitize(re.sub(r"\n{3,}", "\n\n", section.strip()))
        pdf.multi_cell(0, 6, clean)
        pdf.ln(5)

        # Section divider
        pdf.set_draw_color(220, 220, 220)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

    pdf.output(str(output_path))
    print(f"  Saved: {output_path.name}  ({len(sections)} sections, {output_path.stat().st_size // 1024} KB)")


def download_and_convert():
    print("=" * 60)
    print("HR Policy PDF Downloader")
    print("Dataset: syncora/hr-policies-qa-dataset (HuggingFace)")
    print("=" * 60)

    print("\nLoading dataset from HuggingFace...")
    try:
        ds = load_dataset("syncora/hr-policies-qa-dataset", split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Check your internet connection and try again.")
        sys.exit(1)

    print(f"Loaded {len(ds)} records.")
    print(f"Fields: {ds.column_names}\n")

    # Classify each row into a topic bucket
    buckets: dict[str, list[str]] = defaultdict(list)

    print("Classifying records by HR topic...")
    for row in ds:
        text = extract_conversation_text(row)
        if not text.strip():
            continue
        topic = classify_text(text)
        buckets[topic].append(text)

    # Summary
    for topic, items in sorted(buckets.items()):
        print(f"  {topic}: {len(items)} records")

    # Generate one PDF per topic (max 60 sections each to keep file size reasonable)
    print(f"\nGenerating PDFs in {POLICIES_DIR}/...")
    MAX_SECTIONS = 60

    generated = []
    for topic, sections in buckets.items():
        if not sections:
            continue
        out_path = POLICIES_DIR / f"{topic}.pdf"
        make_pdf(topic, sections[:MAX_SECTIONS], out_path)
        generated.append(out_path)

    print(f"\nDone! {len(generated)} policy PDFs written to {POLICIES_DIR}/")
    print("\nNext step — ingest into ChromaDB:")
    print("  python ingest/ingest_pdfs.py")


if __name__ == "__main__":
    download_and_convert()
