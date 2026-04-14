"""
MCP Client — connects to the running MCP server via SSE transport
and calls tools over the MCP protocol (not direct Python imports).

The MCP protocol flow per tool call:
  1. Client sends  -> {"method": "tools/call", "params": {"name": ..., "arguments": ...}}
  2. Server receives -> executes the tool function (queries DB or ChromaDB)
  3. Server responds -> {"content": [{"type": "text", "text": "<result>"}]}
  4. Client returns -> parsed result
"""

import os
import json
import asyncio
from dotenv import load_dotenv
load_dotenv()

from fastmcp import Client

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/sse")


def _run_async(coro):
    """
    Run an async coroutine from sync code.
    Handles the case where an event loop is already running (e.g. inside Streamlit).
    """
    try:
        loop = asyncio.get_running_loop()
        # Already inside a running loop (Streamlit / Jupyter)
        # Run in a separate thread with its own event loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=120)
    except RuntimeError:
        # No running loop — safe to call asyncio.run() directly
        return asyncio.run(coro)


async def _list_tools_async() -> list:
    """List all tools registered on the MCP server."""
    async with Client(MCP_SERVER_URL) as client:
        tools = await client.list_tools()
        return [{"name": t.name, "description": t.description} for t in tools]


async def _call_tool_async(tool_name: str, arguments: dict):
    """Call a single tool on the MCP server and return the raw content."""
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(tool_name, arguments)
        return result


def list_tools() -> list:
    """Synchronous wrapper — list all MCP tools."""
    return _run_async(_list_tools_async())


def call_tool(tool_name: str, arguments: dict):
    """
    Synchronous wrapper — call an MCP tool and return the parsed result.

    MCP returns a list of Content objects. We extract the text and parse JSON where possible.
    """
    raw = _run_async(_call_tool_async(tool_name, arguments))

    # raw is a list of TextContent / other Content types
    if not raw:
        return None

    text = raw[0].text if hasattr(raw[0], "text") else str(raw[0])

    # Try to parse as JSON (dicts/lists); return raw string if not JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def ping() -> bool:
    """Check if the MCP server is reachable."""
    try:
        tools = list_tools()
        return len(tools) > 0
    except Exception:
        return False
