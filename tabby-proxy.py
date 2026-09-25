import os
import re
import json
import uuid
import logging
import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tabby_proxy")

app = FastAPI(title="TabbyAPI Schema Guard Middleware")

TABBY_API_URL = os.getenv("TABBY_API_URL", "http://127.0.0.1:5000")

# Regex patterns for DSML and Qwen tool calling formats
DSML_BLOCK_REGEX = re.compile(
    r"<\s*[\|｜]\s*DSML\s*[\|｜]\s*tool_call\s*>(.*?)(?:</\s*[\|｜]\s*DSML\s*[\|｜]\s*tool_call\s*>|$)",
    re.DOTALL
)
DSML_INVOKE_REGEX = re.compile(
    r"<\s*[\|｜]\s*DSML\s*[\|｜]\s*invoke\s+name=[\"']([^\"']+)[\"']\s*>(.*?)(?:</\s*[\|｜]\s*DSML\s*[\|｜]\s*invoke\s*>|$)",
    re.DOTALL
)
DSML_PARAM_REGEX = re.compile(
    r"<\s*[\|｜]\s*DSML\s*[\|｜]\s*parameter\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</\s*[\|｜]\s*DSML\s*[\|｜]\s*parameter\s*>",
    re.DOTALL
)
QWEN_TAG_REGEX = re.compile(
    r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)",
    re.DOTALL
)


def patch_tools_schema(data: dict) -> dict:
    """Recursively reconstructs and ensures valid fallback descriptions for TabbyAPI validation."""
    if not isinstance(data, dict):
        return data

    patched = {}
    for key, value in data.items():
        if key == "tools" and isinstance(value, list):
            new_tools = []
            for tool in value:
                if isinstance(tool, dict) and tool.get("type") == "function" and "function" in tool:
                    func = dict(tool["function"])

                    if not func.get("description"):
                        func["description"] = f"Executes {func.get('name', 'action')} routine."

                    if "parameters" not in func or func["parameters"] is None:
                        func["parameters"] = {"type": "object", "properties": {}}

                    if "parameters" in func and isinstance(func["parameters"], dict):
                        params = dict(func["parameters"])
                        if "properties" in params and isinstance(params["properties"], dict):
                            props = {}
                            for p_k, p_v in params["properties"].items():
                                if isinstance(p_v, dict):
                                    prop_val = dict(p_v)
                                    if not prop_val.get("description"):
                                        prop_val["description"] = f"Argument parameter: {p_k}"
                                    props[p_k] = prop_val
                                else:
                                    props[p_k] = p_v
                            params["properties"] = props
                        func["parameters"] = params

                    tool["function"] = func
                new_tools.append(tool)
            patched["tools"] = new_tools
        else:
            patched[key] = value

    return patched


def inject_tools_into_messages(messages: list, tools: list) -> list:
    """Injects tool descriptions and tool-call format instructions into the system prompt."""
    if not tools:
        return messages

    tool_lines = []
    for t in tools:
        f = t.get("function", {})
        name = f.get("name")
        if not name:
            continue
        desc = f.get("description", "")
        params = json.dumps(f.get("parameters", {}))
        tool_lines.append(f"- {name}: {desc}\n  Parameters: {params}")

    tool_text = "\n".join(tool_lines)
    instruction_text = (
        "## Available Tools\n"
        "You have access to the following tools:\n"
        f"{tool_text}\n\n"
        "## Tool Calling Instructions\n"
        "When you need to call a tool, you MUST output a tool call block in the following format:\n"
        "<tool_call>\n"
        '{"name": "tool_name", "arguments": {"param1": "value1"}}\n'
        "</tool_call>\n"
        "- Only call tools that are listed above.\n"
        "- Always provide valid JSON for the arguments.\n"
        "- Do not output bracketed pseudo-calls like [tool_call: ...].\n"
        "- You may explain your thoughts before the <tool_call> block.\n"
    )

    out_messages = []
    sys_found = False
    for m in messages:
        m_copy = dict(m)
        if m_copy.get("role") == "system" and not sys_found:
            sys_found = True
            existing_content = m_copy.get("content") or ""
            if "## Available Tools" not in existing_content:
                m_copy["content"] = f"{existing_content}\n\n{instruction_text}".strip()
        out_messages.append(m_copy)

    if not sys_found:
        out_messages.insert(0, {"role": "system", "content": instruction_text})

    return out_messages


def normalize_history_messages(messages: list) -> list:
    """Ensures content is never null and transforms role: 'tool' messages so TabbyAPI does not drop them."""
    out_messages = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if content is None:
            content = ""

        if role == "assistant":
            # If assistant message had tool_calls, make sure they are reflected in content
            tcs = m.get("tool_calls")
            if tcs and isinstance(tcs, list):
                tc_blocks = []
                for tc in tcs:
                    func = tc.get("function", {})
                    name = func.get("name")
                    args = func.get("arguments", "{}")
                    if name and f'"name": "{name}"' not in content and f"<{name}>" not in content:
                        tc_blocks.append(f"\n<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>")
                if tc_blocks:
                    content = (content + "".join(tc_blocks)).strip()
            out_messages.append({"role": "assistant", "content": content})

        elif role == "tool":
            # TabbyAPI ignores role: tool! Convert to role: user
            tool_id = m.get("tool_call_id", "")
            out_messages.append({
                "role": "user",
                "content": f"[Tool Output {tool_id}]:\n{content}"
            })
        else:
            out_messages.append({"role": role, "content": content})

    # Merge consecutive user messages (e.g. parallel tool outputs)
    merged = []
    for m in out_messages:
        if merged and merged[-1]["role"] == "user" and m["role"] == "user":
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(m)

    return merged


def parse_bracket_tool_call(raw: str):
    """Parses bracketed pseudo-syntax [tool_call: ...] emitted by models mimicking few-shot examples."""
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            d = json.loads(raw)
            return d.get("name"), d.get("arguments", {})
        except Exception:
            pass

    parts = raw.split(None, 1)
    if not parts:
        return None, {}
    name = parts[0].strip()
    if len(parts) > 1 and parts[1].strip().startswith("{"):
        try:
            args = json.loads(parts[1].strip())
            return name, args
        except Exception:
            pass

    args = {}
    rest = parts[1].strip() if len(parts) > 1 else ""
    quotes = re.findall(r"['\"]([^'\"]+)['\"]", rest)

    if name in ("read_file", "read"):
        name = "read_file"
        if quotes:
            args["file_path"] = quotes[0]
        m_lim = re.search(r"limit\s+(\d+)", rest)
        if m_lim:
            args["limit"] = int(m_lim.group(1))
        m_off = re.search(r"offset\s+(\d+)", rest)
        if m_off:
            args["offset"] = int(m_off.group(1))
    elif name in ("glob",):
        if quotes:
            args["pattern"] = quotes[0]
        elif rest:
            args["pattern"] = rest.replace("for", "").strip()
    elif name in ("grep_search", "grep"):
        name = "grep_search"
        m_pat = re.search(r"pattern\s+['\"]([^'\"]+)['\"]", rest)
        if m_pat:
            args["pattern"] = m_pat.group(1)
        elif quotes:
            args["pattern"] = quotes[0]
        m_path = re.search(r"path\s+['\"]([^'\"]+)['\"]", rest)
        if m_path:
            args["path"] = m_path.group(1)
        elif len(quotes) > 1:
            args["path"] = quotes[1]
    elif name in ("run_shell_command", "shell"):
        name = "run_shell_command"
        if quotes:
            args["command"] = quotes[0]
        if "is_background: true" in rest.lower():
            args["is_background"] = True
    elif name in ("write_file", "write"):
        name = "write_file"
        if quotes:
            args["file_path"] = quotes[0]
        if len(quotes) > 1:
            args["content"] = quotes[1]
    elif name in ("edit",):
        if quotes:
            args["file_path"] = quotes[0]
    else:
        if quotes:
            args["arg"] = quotes[0]

    return name, args


def extract_tool_calls(content: str):
    """Parses XML, DSML, markdown, or bracketed markup into standard OpenAI tool_calls objects."""
    if not content:
        return None, content

    tool_calls = []
    first_start = len(content)

    # 1. <tool_call>...</tool_call>
    qwen_matches = list(QWEN_TAG_REGEX.finditer(content))
    if qwen_matches:
        for match in qwen_matches:
            first_start = min(first_start, match.start())
            raw_payload = match.group(1).strip()

            # Check JSON
            try:
                call_json = json.loads(raw_payload)
                name = call_json.get("name")
                args = call_json.get("arguments", {})
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args_str
                    }
                })
                continue
            except Exception:
                pass

            # Check XML function format <function=NAME>...
            fn_m = re.search(r"<function=([^>]+)>(.*?)(?:</function>|$)", raw_payload, re.DOTALL)
            if fn_m:
                fn_name = fn_m.group(1).strip()
                params = {}
                for pk, pv in re.findall(r"<parameter=([^>]+)>\s*(.*?)\s*(?:</parameter>|$)", fn_m.group(2), re.DOTALL):
                    val = pv.strip()
                    try:
                        params[pk.strip()] = json.loads(val)
                    except Exception:
                        params[pk.strip()] = val
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": json.dumps(params)
                    }
                })
                continue

    # 2. Markdown tool code blocks
    md_matches = list(re.finditer(r"```(?:tool_call|json)?\s*(\{\s*['\"]name['\"]\s*:.*?)\s*```", content, re.DOTALL))
    if md_matches:
        for match in md_matches:
            first_start = min(first_start, match.start())
            raw_payload = match.group(1).strip()
            try:
                call_json = json.loads(raw_payload)
                name = call_json.get("name")
                args = call_json.get("arguments", {})
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args_str
                    }
                })
            except Exception:
                pass

    # 3. DSML formatting (< | DSML | tool_call>...</ | DSML | tool_call>)
    dsml_matches = list(DSML_BLOCK_REGEX.finditer(content))
    if dsml_matches:
        for match in dsml_matches:
            first_start = min(first_start, match.start())
            block = match.group(1)
            for func_name, invoke_body in DSML_INVOKE_REGEX.findall(block):
                params = {}
                for param_name, param_val in DSML_PARAM_REGEX.findall(invoke_body):
                    val_str = param_val.strip()
                    try:
                        params[param_name] = json.loads(val_str)
                    except Exception:
                        params[param_name] = val_str

                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": func_name.strip(),
                        "arguments": json.dumps(params)
                    }
                })

    # 4. Bracketed formatting [tool_call: ...]
    bracket_matches = list(re.finditer(r"\[tool_call:\s*(.*?)\]", content, re.DOTALL))
    if bracket_matches:
        for match in bracket_matches:
            first_start = min(first_start, match.start())
            raw = match.group(1).strip()
            name, args = parse_bracket_tool_call(raw)
            if name:
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args) if isinstance(args, dict) else str(args)
                    }
                })

    if tool_calls:
        clean_content = content[:first_start].strip() or None
        return tool_calls, clean_content

    return None, content


async def sse_event_stream(resp_json: dict):
    """Converts a parsed tool-completion JSON object into OpenAI-compliant SSE chunks."""
    resp_id = resp_json.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}")
    model = resp_json.get("model", "qwen")
    created = resp_json.get("created", 1700000000)

    for choice in resp_json.get("choices", []):
        idx = choice.get("index", 0)
        message = choice.get("message", {})
        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        tool_calls = message.get("tool_calls")
        finish_reason = choice.get("finish_reason", "stop")

        # Initial role chunk
        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': idx, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

        # Reasoning chunk (if reasoning_content exists)
        if reasoning_content:
            yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': idx, 'delta': {'reasoning_content': reasoning_content}, 'finish_reason': None}]})}\n\n"

        # Content chunk (if any pre-tool thought exists)
        if content:
            yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': idx, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"

        # Tool calls chunks
        if tool_calls:
            for tc_idx, tc in enumerate(tool_calls):
                yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': idx, 'delta': {'tool_calls': [{'index': tc_idx, 'id': tc['id'], 'type': 'function', 'function': {'name': tc['function']['name'], 'arguments': tc['function']['arguments']}}]}, 'finish_reason': None}]})}\n\n"

        # Finish reason chunk
        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': idx, 'delta': {}, 'finish_reason': finish_reason}]})}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request):
    try:
        body = await request.json()
    except Exception:
        return Response(content='{"error": "Invalid JSON raw input body"}', status_code=400, media_type="application/json")

    patched_body = patch_tools_schema(body)

    is_streaming = patched_body.get("stream", False)
    tools = patched_body.get("tools", [])
    has_tools = bool(tools)

    # Preprocess messages for TabbyAPI compatibility
    messages = patched_body.get("messages", [])
    if messages:
        messages = normalize_history_messages(messages)
        if has_tools:
            messages = inject_tools_into_messages(messages, tools)
        patched_body["messages"] = messages

    # Force downstream stop sequences so TabbyAPI halts at the tool boundary
    stop_tokens = [
        "</tool_call>",
        "</ | DSML | tool_call>",
        "</|DSML|tool_call>",
        "<｜tool calls end｜>",
        "<｜tool call end｜>",
        "<|im_end|>"
    ]
    existing_stop = patched_body.get("stop")
    if not existing_stop:
        patched_body["stop"] = stop_tokens
    elif isinstance(existing_stop, str):
        patched_body["stop"] = [existing_stop] + stop_tokens
    elif isinstance(existing_stop, list):
        patched_body["stop"] = existing_stop + [s for s in stop_tokens if s not in existing_stop]

    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    # If streaming with no tools, pass through stream directly
    if not has_tools and is_streaming:
        async def downstream_stream():
            async with httpx.AsyncClient() as stream_client:
                async with stream_client.stream(
                    "POST",
                    f"{TABBY_API_URL}/v1/chat/completions",
                    json=patched_body,
                    headers=headers,
                    timeout=120.0
                ) as down_resp:
                    async for chunk in down_resp.aiter_raw():
                        yield chunk

        return StreamingResponse(downstream_stream(), media_type="text/event-stream")

    # If tools are active, disable downstream streaming so we can intercept and parse tool calls before client sees them
    if has_tools and is_streaming:
        patched_body["stream"] = False

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{TABBY_API_URL}/v1/chat/completions",
                json=patched_body,
                headers=headers,
                timeout=180.0
            )
        except httpx.ConnectError:
            return Response(
                content='{"error": "Could not connect to TabbyAPI downstream server."}',
                status_code=502,
                media_type="application/json"
            )

        if response.status_code != 200:
            return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))

        try:
            resp_data = response.json()
        except Exception:
            return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))

        # Parse message content for tools
        for choice in resp_data.get("choices", []):
            msg = choice.get("message", {})
            content = msg.get("content")

            # Extract any embedded <think>...</think> tags if reasoning_content is empty
            if content and "<think>" in content:
                think_match = re.search(r"<think>(.*?)(?:</think>|$)", content, re.DOTALL)
                if think_match:
                    if not msg.get("reasoning_content"):
                        msg["reasoning_content"] = think_match.group(1).strip()
                    content = re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL).strip()
                    if not content:
                        content = None
                    msg["content"] = content

            if content:
                extracted_calls, cleaned_content = extract_tool_calls(content)
                if extracted_calls:
                    logger.info(f"Intercepted and parsed {len(extracted_calls)} tool call(s): {[c['function']['name'] for c in extracted_calls]}")
                    msg["tool_calls"] = extracted_calls
                    msg["content"] = cleaned_content
                    choice["finish_reason"] = "tool_calls"

        # If the client sent stream=True, convert the parsed result to valid SSE chunks
        if has_tools and is_streaming:
            return StreamingResponse(
                sse_event_stream(resp_data),
                media_type="text/event-stream"
            )

        return Response(
            content=json.dumps(resp_data),
            status_code=200,
            media_type="application/json"
        )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all_proxy(request: Request, path: str):
    async with httpx.AsyncClient() as client:
        url = f"{TABBY_API_URL}/{path}"
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
        req_content = await request.body()

        try:
            response = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=req_content,
                params=request.query_params,
                timeout=60.0
            )
            return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))
        except httpx.ConnectError:
            return Response(content='{"error": "Downstream proxy connection failed"}', status_code=502, media_type="application/json")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)
