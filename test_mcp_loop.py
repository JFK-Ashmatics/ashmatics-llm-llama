"""
Test script for LLM + MCP tool calling loop.

This demonstrates a local LLM (llama.cpp on Metal) calling MCP tools (OpenFDA).
Uses the OpenAI-compatible API format but calls LOCAL infrastructure only.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

# Check if mcp is installed
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("Please run 'uv sync' to install 'mcp' dependency.")
    sys.exit(1)

# =============================================================================
# LOCAL LLM Configuration (llama.cpp server)
# =============================================================================
# This is YOUR local llama.cpp server running on Metal - NOT OpenAI cloud!
LOCAL_LLM_URL = "http://localhost:9000/v1/chat/completions"
LOCAL_LLM_MODEL = "kb-llm"  # Model alias configured in llama.cpp

# =============================================================================
# MCP Server Configuration
# =============================================================================
MCP_REPO_PATH = Path("/Users/kalafuj/GitHub/AsherInformatics/ashmatics-tools").resolve()


async def call_local_llm(messages: list[dict], max_tokens: int = 1024, temperature: float = 0.1) -> str | None:
    """
    Call the local llama.cpp server (OpenAI-compatible API format).

    This is a direct HTTP call to YOUR local server - no cloud dependencies!

    Args:
        messages: Chat messages in OpenAI format
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)

    Returns:
        Assistant's response content, or None on error
    """
    payload = {
        "model": LOCAL_LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(LOCAL_LLM_URL, json=payload)
        response.raise_for_status()
        data = response.json()

        if data.get("choices"):
            return data["choices"][0]["message"]["content"]
        return None


async def main():
    # 1. Setup MCP Client (OpenFDA)
    print(f"🔌 Connecting to MCP Server at: {MCP_REPO_PATH}")
    
    # Use sh -c to redirect stderr to /dev/null to isolate stdout
    server_params = StdioServerParameters(
        command="sh",
        args=["-c", "uv run -q python -m src.ashmatics_tools.mcp_servers.openfda 2>/dev/null"],
        cwd=str(MCP_REPO_PATH),
        env={
            **os.environ, 
            "FDA_API_KEY": os.getenv("FDA_API_KEY", ""),
            "PYTHONUNBUFFERED": "1"
        } 
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 3. List Tools
            tools_result = await session.list_tools()
            mcp_tools = tools_result.tools
            print(f"🛠️  Found {len(mcp_tools)} tools: {[t.name for t in mcp_tools]}")

            # 4. Convert to OpenAI Format
            openai_tools = []
            for tool in mcp_tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })

            # 5. Manual Tool Call Test (Skip LLM for now)
            print("\n🧪 Testing Manual Tool Call: search_devices(query='product_code:QIH')")
            try:
                # Use a known fast query
                # NOTE: 'limit' is page size (records per API call), 'max_records' is total limit
                result = await session.call_tool("search_devices", arguments={
                    "query": "product_code:QIH",
                    "limit": 10,       # Records per API call (page size)
                    "max_records": 5   # Total records to fetch (stops pagination early)
                })
                tool_output = result.content[0].text
                print(f"✅ Tool Call Successful!")
                print(f"📄 Result Preview: {tool_output[:200]}...")
            except Exception as e:
                print(f"❌ Tool Call Failed: {e}")
                raise

            # 6. Full Chat Loop with LLM + MCP Tools (Prompt-based approach)
            # Note: llama.cpp requires --jinja flag for native tool calling.
            # This approach uses prompt engineering to achieve tool calling.
            print("\n" + "="*60)
            print("🤖 Starting LLM Chat Loop with FDA Tools")
            print("="*60)

            # Build tool descriptions for the prompt
            tool_descriptions = []
            for tool in mcp_tools:
                params = tool.inputSchema.get("properties", {})
                param_str = ", ".join([f"{k}: {v.get('type', 'any')}" for k, v in params.items()])
                tool_descriptions.append(f"- {tool.name}({param_str}): {tool.description}")

            tools_text = "\n".join(tool_descriptions)

            # System prompt with tool calling format
            system_prompt = f"""You are a helpful assistant with access to FDA data tools.

AVAILABLE TOOLS:
{tools_text}

OPENFDA QUERY SYNTAX:
- Field search: field_name:value (e.g., product_code:QIH)
- Common 510(k) fields: product_code, device_name, applicant, decision_date, k_number
- Date ranges: decision_date:[20200101 TO 20231231]
- Boolean: field1:value1 AND field2:value2

TOOL CALLING FORMAT:
When you need to use a tool, respond with ONLY a JSON object:
{{"tool": "tool_name", "args": {{"param1": "value1"}}}}

RULES:
1. Always include "max_records" in args (3-10 for quick results)
2. Use endpoint "device_510k" for 510(k) clearances
3. After receiving results, provide a helpful summary
4. If no tool needed, respond normally without JSON

EXAMPLE:
{{"tool": "search_devices", "args": {{"query": "product_code:QIH", "endpoint": "device_510k", "max_records": 3}}}}
"""

            user_query = "Find 510(k) clearances for product code QIH from 2020 onwards. Limit to 3 results."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]

            print(f"\n👤 User: {user_query}")

            # Agentic loop
            max_iterations = 5
            iteration = 0
            had_errors = False

            while iteration < max_iterations:
                iteration += 1
                print(f"\n--- Iteration {iteration} ---")

                # Call local LLM (llama.cpp on Metal)
                try:
                    assistant_content = await call_local_llm(messages)
                except Exception as e:
                    print(f"❌ Local LLM Error: {e}")
                    import traceback
                    traceback.print_exc()
                    had_errors = True
                    break

                # Handle potential None responses
                if not assistant_content:
                    print("❌ Local LLM returned no content")
                    had_errors = True
                    break
                print(f"🤖 LLM Response: {assistant_content[:300]}{'...' if len(assistant_content) > 300 else ''}")

                # Try to parse tool call from response
                tool_call = None
                try:
                    # Try to parse the entire response as JSON first
                    stripped = assistant_content.strip()
                    if stripped.startswith("{"):
                        parsed = json.loads(stripped)
                        if "tool" in parsed and "args" in parsed:
                            tool_call = parsed
                except json.JSONDecodeError:
                    # Try to find JSON embedded in text
                    try:
                        import re
                        # Match JSON with nested braces for args
                        json_match = re.search(r'\{[^{}]*"tool"[^{}]*"args"\s*:\s*\{[^{}]*\}[^{}]*\}', assistant_content)
                        if json_match:
                            parsed = json.loads(json_match.group())
                            if "tool" in parsed:
                                tool_call = parsed
                    except (json.JSONDecodeError, AttributeError):
                        pass

                if tool_call:
                    tool_name = tool_call["tool"]
                    tool_args = tool_call.get("args", {})

                    # Add safety limits
                    if "max_records" not in tool_args:
                        tool_args["max_records"] = 10
                    if "limit" not in tool_args:
                        tool_args["limit"] = 100

                    print(f"🔧 Detected tool call: {tool_name}({tool_args})")

                    # Execute via MCP
                    try:
                        mcp_result = await session.call_tool(tool_name, arguments=tool_args)
                        tool_output = mcp_result.content[0].text

                        # Parse and summarize the output
                        try:
                            result_data = json.loads(tool_output)
                            result_count = result_data.get("count", 0)
                            print(f"✅ Tool returned {result_count} results")
                        except:
                            print(f"✅ Tool completed")

                        # Truncate for context window (llama.cpp may have limited context)
                        if len(tool_output) > 2000:
                            tool_output = tool_output[:2000] + "\n... (truncated)"

                    except Exception as e:
                        tool_output = f"Error executing tool: {str(e)}"
                        print(f"❌ Tool error: {e}")
                        had_errors = True

                    # Add to conversation
                    messages.append({"role": "assistant", "content": assistant_content})
                    messages.append({"role": "user", "content": f"Tool result for {tool_name}:\n{tool_output}\n\nPlease summarize these results for me."})

                else:
                    # No tool call detected - this is the final answer
                    print(f"\n✨ Final Answer:\n{assistant_content}")
                    break

            if iteration >= max_iterations:
                print(f"\n⚠️ Reached max iterations ({max_iterations})")

            print("\n" + "="*60)
            if had_errors:
                print("⚠️ Chat loop completed with errors")
            else:
                print("✅ Chat loop completed successfully!")
            print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
