# AI-Powered HR Assistant

Ask HR questions in plain English and get instant, accurate answers powered by local AI.

100% Open Source | Runs on your own machine | No external API costs

---

## What It Does

**Without this system:** Employee emails HR and waits hours for a manual reply.

**With this system:** Ask in plain English, get an accurate answer in seconds.

**Example:**

    Employee:  "How many holiday days do I have left? My ID is EMP001"
    Assistant: "Hi John! You have 6 days remaining. Your entitlement is 25 days
                and you have taken 17 so far this year, with 2 days pending approval."

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Ollama + Llama 3 (runs locally) |
| RAG Framework | LangChain + LangGraph |
| HR Database | PostgreSQL |
| Vector Store | ChromaDB |
| PDF Processing | pdfplumber + PyMuPDF |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Chat UI | Streamlit |
| Policy Data | HuggingFace (syncora/hr-policies-qa-dataset) |

---

## Architecture

    Employee question
          |
          v
    RAG Orchestrator (Python)
       |                  |
       v                  v
    PostgreSQL          ChromaDB
    (days taken,        (policy text,
     employee info)      entitlements)
       |                  |
       +--------+---------+
                |
                v
          Ollama / Llama 3
          (generates answer)
                |
                v
          Streamlit UI

---

## Project Structure

    hr-ai-assistant/
    |-- docker-compose.yml         # PostgreSQL + ChromaDB containers
    |-- requirements.txt           # Python dependencies
    |-- .env.example               # Config template
    |-- db/
    |   |-- init.sql               # Schema + seed data
    |-- ingest/
    |   |-- download_policies.py   # Download HR policies from HuggingFace
    |   |-- ingest_pdfs.py         # Chunk + embed PDFs into ChromaDB
    |-- mcp_server/
    |   |-- server.py              # Tools: get_leave_info + search_policy
    |-- rag/
    |   |-- orchestrator.py        # Pipeline: fetch data -> prompt -> Ollama
    |-- ui/
    |   |-- app.py                 # Streamlit chat interface
    |-- policies/                  # HR policy PDF files
    |-- GUIDE.md                   # Full low-level build guide

---

## Prerequisites

- Docker Desktop
- Python 3.11+
- Ollama (ollama.com/download)

---

## Quick Start

### 1. Clone the repo

    git clone https://github.com/devaa8564-creator/hr-ai-assistant.git
    cd hr-ai-assistant

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Copy environment config

    cp .env.example .env

### 4. Start the databases

    docker-compose up -d

### 5. Download HR policy documents

    python ingest/download_policies.py

### 6. Ingest PDFs into ChromaDB

    python ingest/ingest_pdfs.py

### 7. Pull the Ollama model

    ollama pull llama3

### 8. Launch the chat UI

    streamlit run ui/app.py

Open your browser at: http://localhost:8501

---

## Sample Employees (for testing)

| Employee ID | Name | Department | Days Taken |
|-------------|------|-----------|-----------|
| EMP001 | John Smith | Engineering | 17 |
| EMP002 | Sara Jones | Marketing | 8 |
| EMP003 | Ahmed Hassan | Finance | 22 |
| EMP004 | Lisa Chen | HR | 5 |

---

## Try These Questions

- "How many holiday days do I have left? My ID is EMP001"
- "What is the sick leave policy?"
- "Can I carry over unused leave to next year?"
- "What is the remote working policy?"
- "How many days has Ahmed Hassan taken? EMP003"

---

## Policy Documents

HR policy PDFs are automatically downloaded from the HuggingFace dataset
syncora/hr-policies-qa-dataset and classified into:

- annual_leave_policy.pdf
- sick_leave_policy.pdf
- remote_work_policy.pdf
- recruitment_policy.pdf
- performance_policy.pdf
- general_hr_policy.pdf

You can replace these with your own company PDF files in the ./policies/ folder.

---

## Configuration

Edit .env to customise:

    DATABASE_URL=postgresql://hruser:hrpassword@localhost:5432/hrdb
    CHROMA_HOST=localhost
    CHROMA_PORT=8000
    CHROMA_COLLECTION=hr_policy
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_MODEL=llama3
    POLICIES_DIR=./policies

To use a different LLM model: change OLLAMA_MODEL and run ollama pull <model-name>.
Models that work well: llama3.1, mistral, qwen2.5

---

## Full Documentation

See GUIDE.md for the complete low-level build guide including:
- Detailed architecture and data flow
- All build stages with code explanations
- Issues encountered and exact fixes applied
- Quick-start cheatsheet

---

## License

MIT License - free to use, modify and distribute.
