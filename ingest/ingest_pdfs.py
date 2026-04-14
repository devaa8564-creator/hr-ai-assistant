"""
Step 2 — Ingest company policy PDFs into ChromaDB.

Usage:
    python ingest/ingest_pdfs.py

Place your PDF files in the ./policies/ folder before running.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import chromadb
import pdfplumber
from sentence_transformers import SentenceTransformer

POLICIES_DIR = Path(os.getenv("POLICIES_DIR", "./policies"))
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "hr_policy")
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between chunks


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ingest_all_pdfs():
    """Parse all PDFs in the policies directory and store in ChromaDB."""
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # Delete and recreate collection for a clean ingest
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    pdf_files = list(POLICIES_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {POLICIES_DIR}. Add policy PDFs there and re-run.")
        # Add a sample policy document so the system works out of the box
        _ingest_sample_policy(collection, model)
        return

    total_chunks = 0
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)

        embeddings = model.encode(chunks).tolist()
        ids = [f"{pdf_path.stem}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": pdf_path.name, "chunk": i} for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)
        print(f"  >> {len(chunks)} chunks ingested")

    print(f"\nDone! Total chunks stored: {total_chunks}")


def _ingest_sample_policy(collection, model):
    """Insert a built-in sample policy so the demo works without real PDFs."""
    sample_policy = """
    Annual Leave Policy

    Full-time employees are entitled to 25 days of paid annual leave per calendar year.
    Part-time employees receive leave on a pro-rata basis proportional to their working hours.

    Leave entitlement accrues from the first day of employment.
    Employees may carry over up to 5 unused days to the following year, subject to manager approval.

    Sick Leave Policy

    Employees are entitled to 10 days of paid sick leave per year.
    A medical certificate is required for absences exceeding 3 consecutive days.

    Public Holidays

    In addition to annual leave, employees receive all public bank holidays as paid days off.
    If a public holiday falls on a weekend, the following Monday is given instead.

    Leave Request Process

    All leave requests must be submitted at least 2 weeks in advance via the HR system.
    Emergency leave may be approved retrospectively at manager discretion.
    """

    chunks = chunk_text(sample_policy)
    embeddings = model.encode(chunks).tolist()
    ids = [f"sample_policy_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "sample_policy.txt", "chunk": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    print("Inserted built-in sample policy (25 days full-time, 10 days sick leave, etc.)")
    print("Replace with your real PDFs in ./policies/ and re-run to use actual documents.")


if __name__ == "__main__":
    ingest_all_pdfs()
