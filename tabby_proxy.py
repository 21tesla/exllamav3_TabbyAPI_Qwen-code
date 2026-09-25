import os
import re
import json
import uuid
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple
import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tabby_proxy")

app = FastAPI(title="TabbyAPI Schema Guard & Tool Calling Middleware")

TABBY_API_URL = os.getenv("TABBY_API_URL", "http://127.0.0.1:5000")
# Upstream auth normally rides along on the client's own Authorization/x-api-key
# header; this is only a fallback for clients that send none.
TABBY_API_KEY = os.getenv("TABBY_API_KEY", "")

# ---------------------------------------------------------------------------
# DSML tool-call parsing
#
# DeepSeek-V4-Flash emits its native tool syntax *inside* a <tool_call> wrapper,
# and this checkpoint of the model does so in several mangled dialects rather
# than only the documented one:
#
#   <tool_call>
#   <|DSML|tool name="read_file">          <- name attribute on the open tag
#   <parameter name="file_path">p</parameter>
#   </|DSML|tool>
#   </tool_call>
#
#   <tool_call>
#   <|DSML| name="read_file">              <- no tag name, only the attribute
#   <parameter name="file_path">p</parameter>
#   </|DSML|>                              <- bare closing tags
#   </|DSML|>
#   </tool_call>
#
#   <tool_call>
#   <\uff5cDSML\uff5c name="read_file">      <- the bars arrive *escaped*
#   ...
#
#   <tool_call>
#   <|DSML|read_file>                      <- function name used as the tag name
#   ...
#
# A single tag scanner covers all of them: walk the <...> tags, strip the DSML
# markers so the tag body reads as a tag name, and track nested call/parameter
# frames. Tags it does not recognise are ignored, leaving the JSON, bracketed
# and fenced extractors to try them.
# ---------------------------------------------------------------------------

_DSML_BARS = re.compile(r"[|\uff5c]")
_DSML_WORD = re.compile(r"DSML", re.IGNORECASE)
_DSML_ESCAPED_BAR = re.compile(r"(?i)\\uff5c")
_DSML_ESCAPED_UNDERSCORE = re.compile(r"(?i)\\u2581")
_TAG_RE = re.compile(r"<[^<>]{0,200}>")
_ATTR_RE = re.compile(r"([A-Za-z_]\w*)\s*=\s*[\"\x27]([^\"\x27]*)[\"\x27]")
_CALL_TAGS = {"tool_call", "tool_calls", "tool", "invoke", "function", "function_call"}
_PARAM_TAGS = {"parameter", "param"}
_CALL_NAME_KEYS = {"name", "tool", "tool_name", "function"}


def unescape_dsml(text: str) -> str:
    """Resolve literal \\uff5c / \\u2581 escapes into the characters they denote."""
    if "\\u" not in text:
        return text
    text = _DSML_ESCAPED_BAR.sub("\uff5c", text)
    return _DSML_ESCAPED_UNDERSCORE.sub("\u2581", text)


def split_dsml_tag(tag: str) -> Tuple[str, bool, Dict[str, str], bool]:
    """Classify one angle-bracket tag -> (keyword, is_closing, attrs, was_dsml)."""
    body = unescape_dsml(tag[1:-1].strip())
    is_closing = body.startswith("/")
    if is_closing:
        body = body[1:].strip()
    attrs = {k.lower(): v for k, v in _ATTR_RE.findall(body)}
    was_dsml = bool(_DSML_WORD.search(body))
    body = _DSML_WORD.sub(" ", _DSML_BARS.sub(" ", body)).strip()
    match = re.match(r"[A-Za-z_][\w.\-]*", body)
    keyword = match.group(0).lower() if match else ""
    return keyword, is_closing, attrs, was_dsml


def classify_dsml_tag(keyword: str, is_closing: bool, attrs: Dict[str, str], was_dsml: bool) -> str:
    """Map a tag to one of: call, param, close, other."""
    if is_closing:
        return "close"
    if keyword in _CALL_TAGS:
        return "call"
    if keyword in _PARAM_TAGS:
        return "param"
    if was_dsml and (attrs.get("name") or keyword):
        return "call"
    return "other"


def _close_dsml_frame(stack: List[dict], keyword: str, calls: List[dict]) -> None:
    """Pop the frame a closing tag refers to and fold its content into the parent."""
    idx = len(stack) - 1
    while idx >= 0:
        frame = stack[idx]
        if not keyword:
            break
        if frame["kind"] == "call" and keyword in _CALL_TAGS:
            break
        if frame["kind"] == "param" and keyword in _PARAM_TAGS:
            break
        idx -= 1
    else:
        return

    frame = stack.pop(idx)
    if frame["kind"] == "param":
        value = "".join(frame["buf"]).strip()
        if value:
            try:
                value = json.loads(value)
            except Exception:
                pass
        for parent in reversed(stack):
            if parent["kind"] != "call":
                continue
            if frame["name"] in _CALL_NAME_KEYS and not parent["name"]:
                parent["name"] = value if isinstance(value, str) else json.dumps(value)
            else:
                parent["args"][frame["name"]] = value
            break
    elif frame["name"]:
        calls.append({"name": frame["name"], "arguments": frame["args"]})


def parse_dsml_tool_calls(content: str) -> Tuple[List[dict], Optional[int]]:
    """Extract DSML tool calls; also return the offset where the syntax begins."""
    calls: List[dict] = []
    stack: List[dict] = []
    first: Optional[int] = None
    cursor = 0

    for match in _TAG_RE.finditer(content):
        if stack and stack[-1]["kind"] == "param":
            stack[-1]["buf"].append(content[cursor:match.start()])
        cursor = match.end()

        keyword, is_closing, attrs, was_dsml = split_dsml_tag(match.group(0))
        kind = classify_dsml_tag(keyword, is_closing, attrs, was_dsml)
        if kind == "call":
            name = attrs.get("name") or (keyword if keyword not in _CALL_TAGS else None)
            stack.append({"kind": "call", "name": name, "args": {}})
            if first is None:
                first = match.start()
        elif kind == "param":
            stack.append({"kind": "param", "name": attrs.get("name") or "arg", "buf": []})
            if first is None:
                first = match.start()
        elif kind == "close":
            _close_dsml_frame(stack, keyword, calls)

    if stack and stack[-1]["kind"] == "param":
        stack[-1]["buf"].append(content[cursor:])
    while stack:
        _close_dsml_frame(stack, "", calls)

    return calls, first


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
        "- If you need to call multiple tools, you can output multiple <tool_call> blocks or a JSON array of tool calls.\n"
        "- You may explain your thoughts or insights before the <tool_call> block.\n"
        "- Immediately after outputting the tool call block(s), end your turn with </tool_call>. Do not hallucinate tool outputs.\n"
    )

    out_messages = []
    sys_found = False
    for m in messages:
        m_copy = dict(m)
        if m_copy.get("role") == "system" and not sys_found:
            sys_found = True
            existing_content = m_copy.get("content") or ""
            if "## Available Tools" in existing_content:
                parts = re.split(r"## Available Tools.*", existing_content, flags=re.DOTALL)
                existing_content = parts[0].strip()
            if existing_content:
                m_copy["content"] = f"{existing_content}\n\n{instruction_text}"
            else:
                m_copy["content"] = instruction_text
        out_messages.append(m_copy)

    if not sys_found:
        out_messages.insert(0, {"role": "system", "content": instruction_text})

    return out_messages


def normalize_history_messages(messages: list) -> list:
    """Ensures content is never null and transforms role: 'tool'/'function' messages so TabbyAPI does not drop them."""
    out_messages = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if content is None:
            content = ""

        if role == "assistant":
            # If assistant message had tool_calls, ensure they are reflected in content
            tcs = m.get("tool_calls")
            if tcs and isinstance(tcs, list):
                tc_blocks = []
                for tc in tcs:
                    func = tc.get("function", {})
                    name = func.get("name")
                    args = func.get("arguments", "{}")
                    args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                    if name and f'"{name}"' not in content and f"<{name}>" not in content:
                        tc_blocks.append(f'\n<tool_call>\n{{"name": "{name}", "arguments": {args_str}}}\n</tool_call>')
                if tc_blocks:
                    content = (content + "".join(tc_blocks)).strip()
            out_messages.append({"role": "assistant", "content": content})

        elif role in ("tool", "function"):
            tool_id = m.get("tool_call_id") or m.get("name") or ""
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


def parse_bracket_tool_call(raw: str) -> Tuple[Optional[str], Any]:
    """Parses bracketed pseudo-syntax [tool_call: ...] emitted by models mimicking few-shot examples."""
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            d = json.loads(raw)
            return normalize_tool_call_dict(d)
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
    quotes = re.findall(r"[\x22\x27]([^\x22\x27]+)[\x22\x27]", rest)

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
        m_pat = re.search(r"pattern\s+[\x22\x27]([^\x22\x27]+)[\x22\x27]", rest)
        if m_pat:
            args["pattern"] = m_pat.group(1)
        elif quotes:
            args["pattern"] = quotes[0]
        m_path = re.search(r"path\s+[\x22\x27]([^\x22\x27]+)[\x22\x27]", rest)
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


def normalize_tool_call_dict(d: dict) -> Tuple[Optional[str], Any]:
    """Extracts function name and argument dict/object from various JSON tool call shapes."""
    if not isinstance(d, dict):
        return None, None
    name = d.get("name") or d.get("function") or d.get("tool") or d.get("action")
    if not name or not isinstance(name, str):
        return None, None
    name = name.strip()
    if not name:
        return None, None

    args = None
    for k in ("arguments", "parameters", "args", "input", "action_input"):
        if k in d:
            args = d[k]
            break
    if args is None:
        other_keys = {k: v for k, v in d.items() if k not in ("name", "function", "tool", "action", "type", "id")}
        args = other_keys if other_keys else {}
    return name, args


def extract_tool_calls(content: str) -> Tuple[Optional[List[dict]], Optional[str]]:
    """Robustly extracts tool calls from content in JSON, DSML, XML, or pseudo formats."""
    if not content:
        return None, content

    tool_calls = []
    seen_sigs = set()
    first_start = len(content)

    def add_call(name: str, args: Any):
        if not name:
            return
        name = str(name).strip()
        if not name:
            return
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                args_str = json.dumps(parsed)
            except Exception:
                args_str = args
        elif isinstance(args, dict):
            args_str = json.dumps(args)
        else:
            args_str = json.dumps(args) if args is not None else "{}"

        sig = (name, args_str)
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args_str
                }
            })

    # Scan for markers that begin tool call sections
    marker_patterns = [
        re.compile(r"<\s*tool_calls?\b[^>]*>", re.IGNORECASE),
        re.compile(r"<\s*(?:[|\uff5c]\s*)?DSML", re.IGNORECASE),
        re.compile(r"(?i)\\uff5c\s*DSML"),
        re.compile(r"\[tool_call\s*:", re.IGNORECASE),
        re.compile(r"```(?:tool_call|json)?\s*\{", re.IGNORECASE),
    ]
    for pat in marker_patterns:
        m = pat.search(content)
        if m and m.start() < first_start:
            first_start = m.start()


    # 1. DSML tool invocations (every dialect, incl. escaped and bare-close forms)
    dsml_calls, dsml_start = parse_dsml_tool_calls(content)
    if dsml_start is not None:
        first_start = min(first_start, dsml_start)
    for dsml_call in dsml_calls:
        add_call(dsml_call["name"], dsml_call["arguments"])


    # 2. JSON objects/arrays scanning using raw_decode
    decoder = json.JSONDecoder()
    start_pos = first_start if first_start < len(content) else 0

    idx = start_pos
    while idx < len(content):
        ch = content[idx]
        if ch in ("{", "["):
            try:
                obj, end_pos = decoder.raw_decode(content, idx)
                if isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, dict):
                            n, a = normalize_tool_call_dict(item)
                            if n:
                                add_call(n, a)
                                first_start = min(first_start, idx)
                elif isinstance(obj, dict):
                    n, a = normalize_tool_call_dict(obj)
                    if n:
                        add_call(n, a)
                        first_start = min(first_start, idx)
                idx = end_pos
                continue
            except Exception:
                pass
        idx += 1

    # 3. XML function format <function=NAME>...
    for fn_m in re.finditer(r"<function=([^>]+)>(.*?)(?:</function>|$)", content, re.DOTALL):
        first_start = min(first_start, fn_m.start())
        fn_name = fn_m.group(1).strip()
        params = {}
        for pk, pv in re.findall(r"<parameter=([^>]+)>\s*(.*?)\s*(?:</parameter>|$)", fn_m.group(2), re.DOTALL):
            val = pv.strip()
            try:
                params[pk.strip()] = json.loads(val)
            except Exception:
                params[pk.strip()] = val
        add_call(fn_name, params)

    # 4. Bracketed formatting [tool_call: ...]
    for bm in re.finditer(r"\[tool_call:\s*(.*?)\]", content, re.DOTALL):
        first_start = min(first_start, bm.start())
        name, args = parse_bracket_tool_call(bm.group(1).strip())
        if name:
            add_call(name, args)

    if tool_calls:
        cleaned = content[:first_start].strip() or None
        return tool_calls, cleaned
    return None, content


def process_message_tools_and_thinking(msg: dict, choice: dict):
    """Parses thinking tags and extracts tool calls from message content or reasoning_content."""
    content = msg.get("content")
    reasoning = msg.get("reasoning_content")

    # If content has embedded <think>...</think> tags, extract them
    if content:
        if "<think>" in content:
            think_match = re.search(r"<think>(.*?)(?:</think>|$)", content, re.DOTALL)
            if think_match:
                if not reasoning:
                    reasoning = think_match.group(1).strip() or None
                content = re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL).strip() or None
        elif "</think>" in content:
            parts = content.split("</think>", 1)
            if not reasoning:
                reasoning = parts[0].strip() or None
            content = parts[1].strip() or None

    msg["reasoning_content"] = reasoning
    msg["content"] = content

    # Attempt to extract tool calls from content first
    extracted_calls = None
    if content:
        extracted_calls, cleaned_content = extract_tool_calls(content)
        if extracted_calls:
            msg["tool_calls"] = extracted_calls
            msg["content"] = cleaned_content
            choice["finish_reason"] = "tool_calls"
            logger.info(f"Intercepted and parsed {len(extracted_calls)} tool call(s) from content: {[c['function']['name'] for c in extracted_calls]}")
            return

    # If not in content, check reasoning_content (in case model emitted tools before </think>)
    if reasoning and not extracted_calls:
        extracted_calls, cleaned_reasoning = extract_tool_calls(reasoning)
        if extracted_calls:
            msg["tool_calls"] = extracted_calls
            msg["reasoning_content"] = cleaned_reasoning
            choice["finish_reason"] = "tool_calls"
            logger.info(f"Intercepted and parsed {len(extracted_calls)} tool call(s) from reasoning: {[c['function']['name'] for c in extracted_calls]}")
            return


    if not extracted_calls:
        raw = content or reasoning or ""
        if "DSML" in raw or "<tool_call" in raw or "\\uff5c" in raw:
            logger.warning(f"Tool-call syntax present but unparsed: {raw[:300]!r}")


async def sse_event_stream(resp_json: dict, requested_model: Optional[str] = None):
    """Converts a parsed tool-completion JSON object into OpenAI-compliant SSE chunks."""
    resp_id = resp_json.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}")
    model = requested_model or resp_json.get("model", "qwen")
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
    requested_model = patched_body.get("model")

    # Preprocess messages for TabbyAPI compatibility
    messages = patched_body.get("messages", [])
    if messages:
        messages = normalize_history_messages(messages)
        if has_tools:
            messages = inject_tools_into_messages(messages, tools)
        patched_body["messages"] = messages

    # Stop sequences to prevent model from hallucinating tool outputs or looping closing tags
    stop_tokens = [
        "\n\n[Tool Output",
        "\n[Tool Output",
        "\n[Tool Output:",
        "\nTool Output:",
        "\nObservation:",
        "\n[Observation]",
        "</ | DSML | tool_calls>",
        "</|DSML|tool_calls>",
        "</｜DSML｜tool_calls>",
        "</ ｜ DSML ｜ tool_calls>",
        "</｜DSML｜tool>\n</｜DSML｜tool>",
        "</ | DSML | tool>\n</ | DSML | tool>",
        "</｜DSML｜tool></｜DSML｜tool>",
        "</ | DSML | tool></ | DSML | tool>",
        "</tool_calls>",
        "<｜tool▁calls▁end｜>",
        "<｜tool▁call▁end｜>",
        "<｜tool calls end｜>",
        "<｜tool call end｜>",
        "<｜end▁of▁sentence｜>",
        "<|im_end|>",
        "<|endoftext|>"
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
    if "x-api-key" not in headers and "authorization" not in {k.lower(): v for k, v in headers.items()}:
        if TABBY_API_KEY:
            headers["x-api-key"] = TABBY_API_KEY

    # If streaming with no tools, pass through stream directly
    if not has_tools and is_streaming:
        async def downstream_stream():
            async with httpx.AsyncClient() as stream_client:
                async with stream_client.stream(
                    "POST",
                    f"{TABBY_API_URL}/v1/chat/completions",
                    json=patched_body,
                    headers=headers,
                    timeout=300.0
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
                timeout=300.0
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

        # Parse message content / reasoning for tool calls
        for choice in resp_data.get("choices", []):
            msg = choice.get("message", {})
            process_message_tools_and_thinking(msg, choice)

        # If the client sent stream=True, convert the parsed result to valid SSE chunks
        if has_tools and is_streaming:
            return StreamingResponse(
                sse_event_stream(resp_data, requested_model=requested_model),
                media_type="text/event-stream"
            )

        return Response(
            content=json.dumps(resp_data),
            status_code=200,
            media_type="application/json"
        )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def catch_all_proxy(request: Request, path: str):
    async with httpx.AsyncClient() as client:
        url = f"{TABBY_API_URL}/{path}"
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
        if "x-api-key" not in headers and "authorization" not in {k.lower(): v for k, v in headers.items()}:
            if TABBY_API_KEY:
                headers["x-api-key"] = TABBY_API_KEY
        req_content = await request.body()

        try:
            response = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=req_content,
                params=request.query_params,
                timeout=120.0
            )
            return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))
        except httpx.ConnectError:
            return Response(content='{"error": "Downstream proxy connection failed"}', status_code=502, media_type="application/json")



_SELFTEST_FW = "\uff5c"


def _selftest_cases():
    fw = _SELFTEST_FW
    v1 = (
        "I'll read the brief for you.\n\n"
        "<tool_call>\n"
        f'<{fw}DSML{fw} name="read_file">\n'
        f'<parameter name="file_path">/tmp/stage123.txt</{fw}DSML{fw}>\n'
        f'</{fw}DSML{fw}>\n'
    )
    v2 = (
        "<tool_call>\n"
        f'<{fw}DSML{fw}tool name="read_file">\n'
        f'<{fw}DSML{fw}parameter name="file_path">/tmp/stage121.txt</{fw}DSML{fw}parameter>\n'
        f'</{fw}DSML{fw}parameter>\n'
        f'</{fw}DSML{fw}tool>\n'
        f'<{fw}DSML{fw}tool>\n'
        f'<{fw}DSML{fw}parameter name="name">glob</{fw}DSML{fw}parameter>\n'
        f'<{fw}DSML{fw}parameter name="pattern">stage121.txt</{fw}DSML{fw}parameter>\n'
        f'</{fw}DSML{fw}tool>\n'
        "</tool_call>"
    )
    v3 = v1.replace(fw, "\\uff5c")
    return [
        ("v1 name attribute + bare closes", v1, [("read_file", {"file_path": "/tmp/stage123.txt"})], "I'll read the brief for you."),
        (
            "v2 tool/parameter tags",
            v2,
            [("read_file", {"file_path": "/tmp/stage121.txt"}), ("glob", {"pattern": "stage121.txt"})],
            None,
        ),
        ("v3 escaped bars (\\uff5c)", v3, [("read_file", {"file_path": "/tmp/stage123.txt"})], "I'll read the brief for you."),
    ]


def _selftest() -> int:
    failures = 0
    cases = _selftest_cases()
    for name, raw, expected, expected_clean in cases:
        calls, cleaned = extract_tool_calls(raw)
        got = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in (calls or [])]
        clean = cleaned or None
        ok = got == expected and clean == expected_clean
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {got} cleaned={clean!r}")
        if not ok:
            failures += 1
            print(f"     expected: {expected} cleaned={expected_clean!r}")
    print(f"{len(cases) - failures}/{len(cases)} DSML self-test cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    uvicorn.run(app, host="127.0.0.1", port=8081)
