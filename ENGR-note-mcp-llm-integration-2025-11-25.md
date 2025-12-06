# Engineering Notes: MCP + Local LLM Integration

**Date:** 2025-11-25
**Project:** llama-dockerize
**Topic:** Integrating local llama.cpp with MCP tool calling

---

## Overview

This document captures lessons learned while building a fully local agentic loop: a local LLM (llama.cpp on Apple Silicon Metal) calling tools via the Model Context Protocol (MCP).

## Architecture

```
┌─────────────────┐     HTTP POST      ┌──────────────────┐
│  test_mcp_loop  │ ←───────────────→  │  llama.cpp       │
│  (Python)       │   localhost:9000   │  (Metal GPU)     │
└────────┬────────┘                    └──────────────────┘
         │
         │ stdio (JSON-RPC 2.0)
         ↓
┌─────────────────┐     HTTPS          ┌──────────────────┐
│  OpenFDA MCP    │ ←───────────────→  │  api.fda.gov     │
│  Server         │                    │  (FDA Open Data) │
└─────────────────┘                    └──────────────────┘
```

---

## Lesson 1: MCP stdio/stdout Complexity

**Problem:** The MCP protocol uses stdin/stdout for JSON-RPC communication. Any stray output to stdout corrupts the message stream.

**Symptoms:**
- Pydantic validation errors on `JSONRPCMessage`
- Mysterious parsing failures
- "Invalid JSON" errors even when JSON looks correct

**Root Causes:**
1. Python logging defaulting to stdout instead of stderr
2. Print statements for debugging
3. Library warnings/deprecation notices

**Solution:**
```python
# Force all logging to stderr
import sys
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,  # CRITICAL: not stdout!
)

# When spawning MCP server, redirect stderr to /dev/null
server_params = StdioServerParameters(
    command="sh",
    args=["-c", "uv run -q python -m module 2>/dev/null"],
    ...
)
```

**Key Insight:** MCP server stdout is SACRED. Only valid JSON-RPC messages should ever be written there.

---

## Lesson 2: JSON-RPC 2.0 Notifications vs Requests

**Problem:** Pydantic validation errors with `id: None` in responses.

**Root Cause:** JSON-RPC 2.0 distinguishes between:
- **Requests**: Have an `id` field, expect a response
- **Notifications**: Have NO `id` field, MUST NOT receive a response

The MCP client sends `notifications/initialized` after initialization. If the server responds to this, it violates the spec.

**Solution:**
```python
# Check if this is a notification
message_id = message.get("id")
is_notification = message_id is None

if is_notification:
    # DO NOT respond - per JSON-RPC 2.0 spec
    logger.debug(f"Received notification: {message.get('method')}")
    continue

# Only respond to requests (those with id)
response = await handle_mcp_request(server, message)
print(json.dumps(response), flush=True)
```

---

## Lesson 3: Pagination Parameter Confusion

**Problem:** MCP tool calls timing out after 60 seconds.

**Symptoms:**
- Direct curl to FDA API: < 1 second
- MCP tool call: 60+ seconds timeout

**Root Cause:** Confusing parameter names in the OpenFDA client:
- `limit` = **page size** (records per API call)
- `max_records` = **total records** to fetch across all pages

With `limit=1` and `max_records=1000` (default), the client made **1000 individual API calls** (1 record per page × 1000 pages).

**Solution:**
```python
# For quick tests, use sensible defaults
result = await session.call_tool("search_devices", arguments={
    "query": "product_code:QIH",
    "limit": 10,       # Records per API call (page size)
    "max_records": 5   # Total records to fetch (stops early)
})
```

**Key Insight:** When wrapping APIs, parameter naming matters. Consider `page_size` and `total_limit` for clarity.

---

## Lesson 4: Async HTTP Client Lifecycle

**Problem:** LLM calls failing on iteration 2+ of agentic loop.

**Symptoms:**
- First LLM call works
- Second call returns empty/None
- No clear error message

**Root Cause:** HTTP client connection pool issues when reusing clients across async contexts.

**Solution:** Create fresh client per request:
```python
async def call_local_llm(messages: list[dict], ...) -> str | None:
    # Create new client each time - avoids connection pool issues
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(LOCAL_LLM_URL, json=payload)
        ...
```

**Also fixed:** In the base API client, reset `_closed` flag after close:
```python
async def close(self) -> None:
    if self._client is not None and not self._closed:
        await self._client.aclose()
        self._closed = True
        self._client = None  # Allow recreation
    self._closed = False  # Reset so we can reopen
```

---

## Lesson 5: llama.cpp Tool Calling Requires --jinja

**Problem:** Native OpenAI-style tool calling fails with llama.cpp.

**Error:**
```
"tools param requires --jinja flag"
```

**Context:** llama.cpp can support native tool calling, but requires:
1. The `--jinja` flag when starting the server
2. A model with proper chat template supporting tools

**Workaround:** Prompt-based tool calling. Include tool definitions in the system prompt and instruct the LLM to output JSON:

```python
system_prompt = f"""You are a helpful assistant with access to FDA data tools.

AVAILABLE TOOLS:
- search_devices(query: string, endpoint: string, max_records: integer): Search FDA device databases

TOOL CALLING FORMAT:
When you need to use a tool, respond with ONLY a JSON object:
{{"tool": "tool_name", "args": {{"param1": "value1"}}}}

After receiving results, provide a helpful summary.
"""
```

Then parse the response:
```python
if assistant_content.strip().startswith("{"):
    parsed = json.loads(assistant_content.strip())
    if "tool" in parsed and "args" in parsed:
        # Execute tool via MCP
        ...
```

---

## Lesson 6: Avoid Vendor-Specific SDK Naming

**Problem:** Using `AsyncOpenAI` client is misleading when calling LOCAL infrastructure.

**Context:** llama.cpp provides an OpenAI-compatible API, so the `openai` SDK works. But:
- The code reads like it's calling OpenAI cloud
- Future maintainers might be confused
- The whole point is LOCAL inference with no cloud dependencies

**Solution:** Use direct `httpx` calls with clear naming:
```python
LOCAL_LLM_URL = "http://localhost:9000/v1/chat/completions"
LOCAL_LLM_MODEL = "kb-llm"

async def call_local_llm(messages: list[dict], ...) -> str | None:
    """
    Call the local llama.cpp server (OpenAI-compatible API format).
    This is a direct HTTP call to YOUR local server - no cloud dependencies!
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(LOCAL_LLM_URL, json=payload)
        ...
```

**Key Insight:** Code should be self-documenting. Naming matters.

---

## Debugging Tips

### 1. Test MCP Server Independently
Use MCP Inspector to verify the server works before integrating with LLM:
```bash
npx @anthropic/mcp-inspector
```

### 2. Test LLM Independently
Use curl to verify the LLM server responds:
```bash
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "kb-llm", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'
```

### 3. Test FDA API Independently
Verify the upstream API works:
```bash
curl "https://api.fda.gov/device/510k.json?search=product_code:QIH&limit=1"
```

### 4. Add Timing to Identify Bottlenecks
```python
import time
start = time.time()
# ... operation ...
print(f"Operation took {time.time() - start:.2f}s")
```

---

## OpenFDA Query Syntax Reference

For future reference when crafting queries:

| Pattern | Example | Description |
|---------|---------|-------------|
| Field search | `product_code:QIH` | Exact field match |
| Date range | `decision_date:[20200101 TO 20231231]` | Date range (inclusive) |
| Open-ended date | `decision_date:[20200101 TO *]` | From date to present |
| Boolean AND | `product_code:QIH AND device_class:2` | Both conditions |
| Boolean OR | `device_name:pump OR device_name:infusion` | Either condition |
| Wildcard | `device_name:pace*` | Prefix matching |

---

## Files Modified

1. `test_mcp_loop.py` - Main test script
2. `README.md` - Added MCP integration documentation
3. `ashmatics-tools/.../openfda/__main__.py` - Fixed notification handling
4. `ashmatics-tools/.../external_apis/base.py` - Fixed client reset

---

## Summary

Building a local LLM + MCP tool calling system requires attention to:

1. **Protocol correctness** - JSON-RPC 2.0 has specific rules
2. **Stream isolation** - stdout is for messages only
3. **Parameter semantics** - naming matters for clarity
4. **Client lifecycle** - async HTTP clients need careful management
5. **Graceful degradation** - prompt-based fallbacks when native support unavailable
6. **Clear naming** - code should reflect actual infrastructure

The result is a fully local agentic system with zero cloud dependencies.
