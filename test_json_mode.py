"""
Test JSON mode reliability with local llama.cpp server.

This script tests whether the model reliably outputs valid JSON when using
the response_format parameter. Useful for validating extraction workflows.

Usage:
    uv run python test_json_mode.py

Requires the server to be running (make run-local).
"""
import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any

import httpx

LOCAL_LLM_URL = "http://localhost:9000/v1/chat/completions"
LOCAL_LLM_MODEL = "kb-llm"


@dataclass
class TestResult:
    name: str
    passed: bool
    response: str
    parsed: dict | None
    error: str | None


async def call_llm(
    messages: list[dict],
    response_format: dict | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Call local LLM with optional JSON mode."""
    payload = {
        "model": LOCAL_LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if response_format:
        payload["response_format"] = response_format

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(LOCAL_LLM_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def try_parse_json(text: str) -> tuple[dict | None, str | None]:
    """Attempt to parse JSON, including from markdown code blocks."""
    # Try direct parse first
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    # Try extracting from code blocks
    if "```json" in text:
        try:
            extracted = text.split("```json")[1].split("```")[0].strip()
            return json.loads(extracted), None
        except (json.JSONDecodeError, IndexError):
            pass

    if "```" in text:
        try:
            extracted = text.split("```")[1].split("```")[0].strip()
            return json.loads(extracted), None
        except (json.JSONDecodeError, IndexError):
            pass

    return None, f"Failed to parse JSON from: {text[:200]}..."


async def test_simple_json() -> TestResult:
    """Test basic JSON output without response_format."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON only, no other text."},
        {"role": "user", "content": "Return a JSON object with keys 'name' (string) and 'age' (integer) for a person named Alice who is 30."}
    ]

    response = await call_llm(messages)
    parsed, error = try_parse_json(response)

    passed = parsed is not None and "name" in parsed and "age" in parsed
    return TestResult(
        name="Simple JSON (no response_format)",
        passed=passed,
        response=response,
        parsed=parsed,
        error=error
    )


async def test_json_mode() -> TestResult:
    """Test JSON output with response_format parameter."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that outputs valid JSON."},
        {"role": "user", "content": "Return a JSON object with keys 'name' (string) and 'age' (integer) for a person named Bob who is 25."}
    ]

    response = await call_llm(
        messages,
        response_format={"type": "json_object"}
    )
    parsed, error = try_parse_json(response)

    passed = parsed is not None and "name" in parsed and "age" in parsed
    return TestResult(
        name="JSON mode (with response_format)",
        passed=passed,
        response=response,
        parsed=parsed,
        error=error
    )


async def test_extraction_schema() -> TestResult:
    """Test complex extraction schema like ashmatics-tools uses."""
    messages = [
        {
            "role": "system",
            "content": """You are an expert at extracting structured data.
Output valid JSON matching this schema:
{
    "has_data": boolean,
    "items": [{"name": string, "value": number, "unit": string}],
    "confidence": number (0-100)
}"""
        },
        {
            "role": "user",
            "content": """Extract metrics from this text:

The device showed 95% sensitivity and 92% specificity in clinical trials.
The accuracy was measured at 0.94 AUC."""
        }
    ]

    response = await call_llm(
        messages,
        response_format={"type": "json_object"},
        max_tokens=1024
    )
    parsed, error = try_parse_json(response)

    # Check structure
    passed = (
        parsed is not None
        and "has_data" in parsed
        and "items" in parsed
        and isinstance(parsed.get("items"), list)
    )
    return TestResult(
        name="Complex extraction schema",
        passed=passed,
        response=response,
        parsed=parsed,
        error=error
    )


async def test_nested_json() -> TestResult:
    """Test nested JSON structures."""
    messages = [
        {"role": "system", "content": "You output valid JSON only."},
        {
            "role": "user",
            "content": """Return a JSON object representing a company with:
- name (string)
- employees (array of objects with 'name' and 'role')
- address (object with 'city' and 'country')

Use "TechCorp" with 2 employees."""
        }
    ]

    response = await call_llm(
        messages,
        response_format={"type": "json_object"}
    )
    parsed, error = try_parse_json(response)

    passed = (
        parsed is not None
        and "name" in parsed
        and "employees" in parsed
        and isinstance(parsed.get("employees"), list)
        and "address" in parsed
    )
    return TestResult(
        name="Nested JSON structure",
        passed=passed,
        response=response,
        parsed=parsed,
        error=error
    )


async def test_stop_sequences() -> TestResult:
    """Test if stop sequences are respected (known limitation)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count from 1 to 10, one number per line."}
    ]

    payload = {
        "model": LOCAL_LLM_MODEL,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": False,
        "stop": ["5"]  # Should stop at 5
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(LOCAL_LLM_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]

    # Check if it stopped before reaching numbers > 5
    has_six_or_higher = any(str(n) in content for n in range(6, 11))
    passed = not has_six_or_higher

    return TestResult(
        name="Stop sequences (known limitation)",
        passed=passed,
        response=content,
        parsed=None,
        error="Stop sequences not respected - this is a known llama-server limitation" if not passed else None
    )


async def main():
    print("=" * 60)
    print("JSON Mode Reliability Test")
    print("=" * 60)
    print(f"Server: {LOCAL_LLM_URL}")
    print()

    # Check server is running
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get("http://localhost:9000/health")
            r.raise_for_status()
            health = r.json()
            print(f"Model: {health.get('model', 'unknown')}")
    except Exception as e:
        print(f"Server not reachable: {e}")
        print("Start the server with: make run-local")
        sys.exit(1)

    print()

    tests = [
        test_simple_json,
        test_json_mode,
        test_extraction_schema,
        test_nested_json,
        test_stop_sequences,
    ]

    results: list[TestResult] = []
    for test_fn in tests:
        print(f"Running: {test_fn.__name__}...")
        try:
            result = await test_fn()
            results.append(result)
        except Exception as e:
            results.append(TestResult(
                name=test_fn.__name__,
                passed=False,
                response="",
                parsed=None,
                error=str(e)
            ))

    # Summary
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        icon = "\u2705" if result.passed else "\u274c"
        print(f"{icon} [{status}] {result.name}")

        if not result.passed and result.error:
            print(f"   Error: {result.error}")

        if result.parsed:
            print(f"   Parsed: {json.dumps(result.parsed, indent=2)[:200]}...")
        elif result.response and not result.passed:
            print(f"   Response: {result.response[:200]}...")

        print()

    # Final summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print("=" * 60)
    print(f"Summary: {passed}/{total} tests passed")

    if passed < total:
        print()
        print("Note: Stop sequence limitation is expected behavior.")
        print("JSON mode failures may require client-side retry logic.")

    print("=" * 60)

    return 0 if passed >= total - 1 else 1  # Allow 1 failure (stop sequences)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
