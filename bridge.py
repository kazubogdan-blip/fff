import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx

app = FastAPI()
LLAMA_URL = "http://localhost:8080/v1/chat/completions"

@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    messages = []
    if "system" in body:
        sys = body["system"]
        if isinstance(sys, list):
            sys = "\n".join([s.get("text", "") for s in sys])
        if sys:
            messages.append({"role": "system", "content": sys})
    for m in body.get("messages", []):
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            content = "".join([c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"])
        elif not isinstance(content, str):
            content = str(content)
        messages.append({"role": role, "content": content})
    is_stream = body.get("stream", False)
    openai_payload = {
        "model": "qwen-27b",
        "messages": messages,
        "max_tokens": body.get("max_tokens", 4096),
        "temperature": body.get("temperature", 0.7),
        "stream": is_stream
    }
    client = httpx.AsyncClient(timeout=120.0)
    if is_stream:
        async def event_generator():
            async with client.stream("POST", LLAMA_URL, json=openai_payload) as response:
                yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'content': [], 'model': 'claude-3-5-sonnet-20241022', 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 10, 'output_tokens': 1}}})}\n\n"
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta}})}\n\n"
                        except Exception:
                            pass
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 15}}})}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        async with client as c:
            resp = await c.post(LLAMA_URL, json=openai_payload)
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": content}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 20}
            }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434)