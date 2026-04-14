"""
Step 3 — MCP Server with two tools:
  - get_leave_info(employee_id)  → queries PostgreSQL
  - search_policy(query)         → queries ChromaDB

Run:
    python mcp_server/server.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

import psycopg2
import chromadb
from sentence_transformers import SentenceTransformer
from fastmcp import FastMCP

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://hruser:hrpassword@localhost:5432/hrdb")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "hr_policy")

mcp = FastMCP("hr-assistant-server")
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _get_db_conn():
    return psycopg2.connect(DATABASE_URL)


@mcp.tool()
def get_leave_info(employee_id: str) -> dict:
    """
    Retrieve leave information for an employee from the HR database.
    Returns days taken, days pending approval, and employee details.
    """
    try:
        conn = _get_db_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.name, e.department, e.employment_type,
                   l.year, l.days_taken, l.days_pending
            FROM employees e
            LEFT JOIN employee_leave l
                ON e.employee_id = l.employee_id
                AND l.year = EXTRACT(YEAR FROM CURRENT_DATE)
            WHERE e.employee_id = %s
            """,
            (employee_id.upper(),),
        )
        row = cur.fetchone()
        conn.close()

        if not row:
            return {"error": f"No employee found with ID {employee_id}"}

        name, department, emp_type, year, days_taken, days_pending = row
        return {
            "employee_id": employee_id.upper(),
            "name": name,
            "department": department,
            "employment_type": emp_type,
            "year": year,
            "days_taken": days_taken or 0,
            "days_pending": days_pending or 0,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def search_policy(query: str) -> str:
    """
    Search the company HR policy documents for information relevant to the query.
    Returns the most relevant policy text chunks.
    """
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = client.get_collection(COLLECTION_NAME)

        model = _get_embed_model()
        query_embedding = model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3,
        )

        docs = results.get("documents", [[]])[0]
        if not docs:
            return "No relevant policy information found."

        return "\n\n---\n\n".join(docs)
    except Exception as e:
        return f"Policy search error: {e}"


@mcp.tool()
def list_employees() -> list:
    """List all employees in the HR system (for debugging / admin use)."""
    try:
        conn = _get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT employee_id, name, department FROM employees ORDER BY name")
        rows = cur.fetchall()
        conn.close()
        return [{"employee_id": r[0], "name": r[1], "department": r[2]} for r in rows]
    except Exception as e:
        return [{"error": str(e)}]


if __name__ == "__main__":
    print("Starting HR MCP Server...")
    mcp.run()
