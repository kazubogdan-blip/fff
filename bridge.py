cat << 'EOF' > bridge.py
import json
from fastapi import FastAPI, Request
import httpx

app = FastAPI()
LLAMA_URL = "http://localhost:8080/v1/chat/completions"

def convert_tools(anthropic_tools):
    if not anthropic_tools:
        return None
    openai_tools = []
    for t in anthropic_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t.get("name"),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}})
            }
        })
    return openai_tools

def convert_messages(anthropic_messages):
    openai_messages = []
    for m in anthropic_messages:
        role = m.get("role")
        content = m.get("content")
        
        if role == "user":
            if isinstance(content, str):
                openai_messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        b_type = block.get("type")
                        if b_type == "text":
                            text_parts.append(block.get("text", ""))
                        elif b_type == "tool_result":
                            tool_result_content = block.get("content", "")
                            if isinstance(tool_result_content, list):
                                tool_result_content = "".join([c.get("text", "") for c in tool_result_content if isinstance(c, dict)])
                            # OpenAI așteaptă rolul "tool" pentru rezultatele uneltelor
                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id"),
                                "content": str(tool_result_content)
                            })
                if text_parts:
                    openai_messages.append({"role": "user", "content": "\n".join(text_parts)})
                    
        elif role == "assistant":
            if isinstance(content, str):
                openai_messages.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text_parts = []
                tool_calls = []
                for block in content:
                    if isinstance(block, dict):
                        b_type = block.get("type")
                        if b_type == "text":
                            text_parts.append(block.get("text", ""))
                        elif b_type == "tool_use":
                            tool_calls.append({
                                "id": block.get("id", "call_1"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name"),
                                    "arguments": json.dumps(block.get("input", {}))
                                }
                            })
                msg_obj = {"role": "assistant"}
                msg_obj["content"] = "\n".join(text_parts) if text_parts else None
                if tool_calls:
                    msg_obj["tool_calls"] = tool_calls
                openai_messages.append(msg_obj)
    return openai_messages

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
            
    messages.extend(convert_messages(body.get("messages", [])))
    
    openai_payload = {
        "model": "qwen",
        "messages": messages,
        "max_tokens": min(body.get("max_tokens", 8192), 8192),
        "temperature": body.get("temperature", 0.7),
        "stream": False
    }
    
    tools = convert_tools(body.get("tools"))
    if tools:
        openai_payload["tools"] = tools

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(LLAMA_URL, json=openai_payload)
        data = resp.json()
        
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls")
        
        response_content = []
        stop_reason = "end_turn"
        
        if content:
            response_content.append({"type": "text", "text": content})
            
        if tool_calls:
            stop_reason = "tool_use"
            for tc in tool_calls:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args_dict = json.loads(args_str)
                except Exception:
                    args_dict = {}
                    
                response_content.append({
                    "type": "tool_use",
                    "id": tc.get("id", "call_1"),
                    "name": func.get("name"),
                    "input": args_dict
                })
                
        if not response_content:
            response_content.append({"type": "text", "text": "Done."})

        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": response_content,
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 20}
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434)
EOF
