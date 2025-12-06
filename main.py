import os
import json
from typing import AsyncGenerator, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

import logging
import time

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("llama-wrapper")

LLAMA_PORT = int(os.getenv("LLAMA_PORT", "11434"))
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "kb-llm")

LLAMA_BASE_URL = f"http://127.0.0.1:{LLAMA_PORT}/v1"

app = FastAPI(title="llama.cpp FastAPI Wrapper", version="0.2.0")


# ---------- Schemas ----------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = MODEL_ALIAS
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = True
    tools: Optional[List[dict]] = None
    tool_choice: Optional[str] = None

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    stop: Optional[List[str]] = None
    # For RAG-like prompts, you can pass context/system fields:
    system: Optional[str] = "You are a helpful assistant."
    # streaming is always True for this endpoint, but kept for future flexibility
    stream: bool = True


class EmbedRequest(BaseModel):
    inputs: List[str]


class EmbeddingsRequest(BaseModel):
    """OpenAI-compatible embeddings request schema."""
    input: str | List[str]
    model: Optional[str] = MODEL_ALIAS
    encoding_format: Optional[str] = "float"  # 'float' or 'base64'


# ---------- Utils ----------
async def _ensure_llama_ready():
    """
    Verify llama-server is reachable; raise HTTPException if not.
    """
    async with httpx.AsyncClient(timeout=3) as client:
        try:
            r = await client.get(f"{LLAMA_BASE_URL}/models")
            r.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"llama-server not ready: {e}")


# ---------- Endpoints ----------
@app.get("/health")
async def health():
    await _ensure_llama_ready()
    return {"status": "ok", "llama_server": LLAMA_BASE_URL, "model": MODEL_ALIAS}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Streams tokens from llama-server (OpenAI-compatible /v1/chat/completions).
    Returns text/event-stream-like chunked text via StreamingResponse.
    """
    await _ensure_llama_ready()

    payload = {
        "model": MODEL_ALIAS,
        "stream": True,  # streaming
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
        "stop": req.stop or [],
        "messages": [
            {"role": "system", "content": req.system or ""},
            {"role": "user", "content": req.prompt},
        ],
    }

    async def event_stream() -> AsyncGenerator[bytes, None]:
        # Note: /v1/chat/completions with stream=True typically yields "data: {...}\n\n"
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{LLAMA_BASE_URL}/chat/completions", json=payload) as r:
                if r.status_code >= 400:
                    # Read body (if any) to include in the error
                    try:
                        err = await r.aread()
                    except Exception:
                        err = b""
                    detail = err.decode("utf-8", errors="ignore")
                    raise HTTPException(status_code=r.status_code, detail=detail or "llama-server error")

                async for line in r.aiter_lines():
                    if not line:
                        continue
                    # Typical format: "data: {JSON}\r\n"
                    if line.startswith("data: "):
                        data = line[len("data: "):]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                            # OpenAI chat streaming payload has choices[].delta.content
                            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                # yield raw text chunks (you can also wrap in JSON)
                                yield delta.encode("utf-8")
                        except json.JSONDecodeError:
                            # Ignore malformed lines
                            continue

    # Content-Type is text/plain (chunked). Change to "text/event-stream" if you want SSE semantics.
    return StreamingResponse(event_stream(), media_type="text/plain")


@app.post("/embed")
async def embed(req: EmbedRequest):
    """
    Calls llama-server /v1/embeddings (non-streaming).
    Legacy endpoint - prefer /v1/embeddings for OpenAI compatibility.
    """
    await _ensure_llama_ready()

    payload = {"model": MODEL_ALIAS, "input": req.inputs}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{LLAMA_BASE_URL}/embeddings", json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        # Normalize embeddings response
        vectors = [item.get("embedding", []) for item in data.get("data", [])]
        return {"embeddings": vectors, "count": len(vectors)}


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingsRequest):
    """
    OpenAI-compatible embeddings endpoint.

    Request:
        {
            "input": "text" or ["text1", "text2", ...],
            "model": "model-name" (optional),
            "encoding_format": "float" (optional)
        }

    Response:
        {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [...], "index": 0},
                ...
            ],
            "model": "model-name",
            "usage": {"prompt_tokens": N, "total_tokens": N}
        }
    """
    await _ensure_llama_ready()

    start_time = time.time()

    # Normalize input to list
    input_texts = req.input if isinstance(req.input, list) else [req.input]
    logger.info(f"Embeddings request: {len(input_texts)} inputs")

    payload = {
        "model": MODEL_ALIAS,
        "input": input_texts,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{LLAMA_BASE_URL}/embeddings", json=payload)

        if r.status_code >= 400:
            logger.error(f"Embeddings error: {r.status_code} - {r.text}")
            raise HTTPException(status_code=r.status_code, detail=r.text)

        data = r.json()
        duration = time.time() - start_time
        logger.info(f"Embeddings completed in {duration:.2f}s")

        # llama-server should return OpenAI-compatible format already,
        # but ensure we have proper structure
        if "data" not in data:
            # Fallback: construct response from raw embeddings
            embeddings_list = data.get("embeddings", data.get("embedding", []))
            if not isinstance(embeddings_list[0], list):
                embeddings_list = [embeddings_list]

            data = {
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": emb, "index": i}
                    for i, emb in enumerate(embeddings_list)
                ],
                "model": MODEL_ALIAS,
                "usage": {"prompt_tokens": 0, "total_tokens": 0}
            }

        # Ensure model field is set
        data["model"] = data.get("model", MODEL_ALIAS)

        return data


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    Standard OpenAI-compatible chat completion endpoint.
    Proxies directly to llama-server.
    """
    await _ensure_llama_ready()
    
    start_time = time.time()
    logger.info(f"Chat request: {len(req.messages)} messages")

    # Forward the request payload almost as-is
    payload = req.dict(exclude_none=True)
    # Ensure model name matches what llama-server expects (or use alias)
    payload["model"] = MODEL_ALIAS

    async def event_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", f"{LLAMA_BASE_URL}/chat/completions", json=payload) as r:
                    if r.status_code >= 400:
                        err_body = await r.aread()
                        logger.error(f"Upstream error: {r.status_code} - {err_body}")
                        raise HTTPException(status_code=r.status_code, detail=err_body.decode())

                    async for line in r.aiter_lines():
                        if line:
                            yield f"{line}\n\n"
            except Exception as e:
                logger.error(f"Stream error: {e}")
                raise e
            finally:
                duration = time.time() - start_time
                logger.info(f"Request finished in {duration:.2f}s")

    if req.stream:
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        # Non-streaming fallback
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{LLAMA_BASE_URL}/chat/completions", json=payload)
            return r.json()


@app.get("/v1/models")
async def list_models():
    """
    List available models. Proxies to llama-server.
    """
    await _ensure_llama_ready()
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{LLAMA_BASE_URL}/models")
        return r.json()


@app.get("/v1")
async def v1_root():
    return {"message": "Llama Wrapper API v1", "docs": "/docs"}


# ---------- UI ----------
@app.get("/", response_class=HTMLResponse)
async def ui():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Llama Docker Wrapper</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
        .chat-container { height: 65vh; overflow-y: auto; border: 1px solid #333; padding: 20px; margin-bottom: 20px; background: #252526; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .message { margin: 15px 0; padding: 12px 16px; border-radius: 8px; line-height: 1.5; }
        .user { background: #0e639c; color: white; margin-left: 20%; border-bottom-right-radius: 2px; }
        .assistant { background: #3c3c3c; margin-right: 20%; border-bottom-left-radius: 2px; }
        .controls { display: flex; gap: 10px; }
        input[type="text"] { flex-grow: 1; padding: 12px; background: #3c3c3c; border: 1px solid #555; color: white; border-radius: 4px; font-size: 16px; outline: none; }
        input[type="text"]:focus { border-color: #0e639c; }
        button { padding: 12px 24px; background: #0e639c; color: white; border: none; cursor: pointer; border-radius: 4px; font-size: 16px; font-weight: 600; transition: background 0.2s; }
        button:hover { background: #1177bb; }
        pre { white-space: pre-wrap; margin: 0; font-family: inherit; }
        .status { font-size: 12px; color: #888; margin-bottom: 10px; text-align: right; }
    </style>
</head>
<body>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <h1 style="margin: 0;">🐳 Llama Local</h1>
        <div class="status">Model: """ + MODEL_ALIAS + """</div>
    </div>
    <div id="chat" class="chat-container"></div>
    <div class="controls">
        <input type="text" id="prompt" placeholder="Type a message..." autocomplete="off">
        <button onclick="send()">Send</button>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const promptInput = document.getElementById('prompt');
        let history = [];

        promptInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') send();
        });

        async function send() {
            const text = promptInput.value.trim();
            if (!text) return;
            
            addMessage('user', text);
            promptInput.value = '';
            history.push({role: 'user', content: text});

            const assistantDiv = addMessage('assistant', '...');
            let fullResponse = "";

            try {
                const response = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        messages: history,
                        stream: true
                    })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                assistantDiv.innerHTML = ''; 

                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                            try {
                                const data = JSON.parse(line.slice(6));
                                const content = data.choices[0].delta.content || "";
                                fullResponse += content;
                                assistantDiv.innerHTML = `<pre>${fullResponse}</pre>`;
                                chat.scrollTop = chat.scrollHeight;
                            } catch (e) { console.error(e); }
                        }
                    }
                }
                history.push({role: 'assistant', content: fullResponse});
            } catch (e) {
                assistantDiv.textContent = "Error: " + e.message;
            }
        }

        function addMessage(role, text) {
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = `<pre>${text}</pre>`;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }
    </script>
</body>
</html>
"""

