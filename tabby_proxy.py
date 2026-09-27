import os
import re
import json
import time
import uuid
import asyncio
import logging
import sys
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
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
# An attribute whose value is a brace object rather than a quoted string, e.g.
# `args={"file_path": "/x"}`. The braces balance one level; a nested object inside
# would end the match early, which the caller's JSON repair then covers.
_ATTR_OBJ_RE = re.compile(r"([A-Za-z_]\w*)\s*=\s*(\{[^{}]*\})")
_CALL_TAGS = {"tool_call", "tool_calls", "calls", "call_tool", "tool", "invoke", "function", "function_call"}
_PARAM_TAGS = {"parameter", "param"}
# A container element holding the whole arguments object, rather than one named
# parameter. The checkpoint writes this dialect alongside the `name=` attribute on
# the call tag (observed 2026-09-26, live probe e5e8c7f9, first turn):
#
#   <tool_call>
#   <|DSML| name="read_file">
#   <arguments>{"file_path": "/home/logan/software/analysis-qt6/stage141.txt"}</arguments>
#   </|DSML|>
#   </tool_call>
#
# Without this the tag is not a parameter, its JSON is never collected, and the
# call is emitted with empty arguments -- which the client rejects as
# `invalid_tool_params` and retries.
_ARG_CONTAINER_TAGS = {"arguments", "args"}
_CALL_NAME_KEYS = {"name", "tool", "tool_name", "function"}
# Keys whose *value* is the arguments object or the arguments JSON, when they ride
# on the call tag as attributes rather than as child elements. The `call_tool`
# dialect carries both the name and the whole argument object as attributes
# (`<|DSML|call_tool name="read_file" args={...}>`), so `args` and `arguments` have
# to be readable from `attrs` -- while the `parameter` sibling keeps its value as an
# ordinary argument named `args`.
_CALL_ARGS_KEYS = {"args", "arguments"}
_CALL_META_KEYS = {"name", "function", "tool", "tool_name", "action", "type", "id"}


# A completion past roughly 180k tokens starts losing ASCII: `"` becomes `“`/`”`
# and `|` becomes the fullwidth `｜` (U+FF5C). Neither is a quoting or bar character
# to Python, so a tool-call object written in the wrong glyphs is not JSON at all
# and every call in the turn is dropped. Observed 2026-09-26 on DeepSeek-V4-Flash
# at 195k input tokens: the first call used ASCII, and everything after the first
# mangled tag used these.
_TYPOGRAPHIC_QUOTES = ("\u201c", "\u201d", "\u2018", "\u2019")
_TYPOGRAPHIC_FOLDS = str.maketrans(
    {c: '"' if c in ("\u201c", "\u201d") else "'" for c in _TYPOGRAPHIC_QUOTES}
)


def fold_typographic(text: str) -> str:
    """Fold the quote-like glyphs a long-context completion substitutes for ASCII.

    Applied as an alternative parse of the same completion, never in place: a
    literal `“` inside a string value is legitimate text, so the original form is
    always tried first and this is only a second reading. The fullwidth bar is
    deliberately *not* folded here -- it is meaningful DSML syntax (see
    `fold_dsml_syntax`).
    """
    return text.translate(_TYPOGRAPHIC_FOLDS)


def fold_dsml_syntax(text: str) -> str:
    """Rewrite a DSML tag the model mangled into the shape the scanner expects.

    Two mangles are repaired, both from the long-context glyph loss above, and both
    observed live on 2026-09-26:

      * the fullwidth bar substituted for the ASCII pipe, so the closer
        `<` `/` U+FF5C `DSML` U+FF5C `_call` `>` is not recognised as a DSML tag at
        all and the word `_call` reads as a tool name;
      * a backslash where the slash belongs, which turns the closer of a wrapper the
        model never opened into something that looks like an opening tag.

    A tag with no bar, escape or backslash is returned untouched: rewriting it would
    turn ordinary prose markup (`<code>`) into a DSML call, which is the opposite of
    the intent. A well-formed `</|DSML|tool>` is also left alone -- `split_dsml_tag`
    already classifies it.
    """
    stripped = text.strip()
    if not (stripped.startswith("<") and stripped.endswith(">")):
        return text
    inner = stripped[1:-1]
    if not ("|" in inner or "\uff5c" in inner or "\\" in inner or "DSML" in stripped.upper()):
        return text
    leading_backslash = inner.startswith("\\")
    if leading_backslash:
        inner = inner[1:].lstrip()
    inner = inner.replace("\\uff5c", "|").replace("\\u2581", "_").replace("\uff5c", "|")
    closing = inner.startswith("/")
    if closing:
        inner = inner[1:].lstrip()
    inner = inner.replace("|", "")
    inner = re.sub(r"(?i)\bDSML\b", "", inner).strip()
    if inner in _CALL_TAGS:
        return text
    if not re.fullmatch(r"[A-Za-z_][\w.\-]*", inner):
        return text
    # A backslash stood where the slash belongs, so the tag was written as a closer.
    slash = "/" if (closing or leading_backslash) else ""
    return f"<{slash}|DSML|{inner}>"


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
    # An attribute whose value is a brace object is not quoted, so the regex above
    # skips it -- and that is exactly how the `call_tool` dialect carries its
    # arguments (`args={...}`). Capture the balanced braces separately; the nested
    # braces of the object itself keep `[^{}]` from stopping at the first inner one.
    for key, value in _ATTR_OBJ_RE.findall(body):
        attrs.setdefault(key.lower(), value)
    was_dsml = bool(_DSML_WORD.search(body))
    body = _DSML_WORD.sub(" ", _DSML_BARS.sub(" ", body)).strip()
    match = re.match(r"[A-Za-z_][\w.\-]*", body)
    keyword = match.group(0).lower() if match else ""
    return keyword, is_closing, attrs, was_dsml


def classify_dsml_tag(keyword: str, is_closing: bool, attrs: Dict[str, str], was_dsml: bool) -> str:
    """Map a tag to one of: call, param, args, close, other."""
    if is_closing:
        return "close"
    if keyword in _CALL_TAGS:
        return "call"
    if keyword in _PARAM_TAGS:
        return "param"
    if keyword in _ARG_CONTAINER_TAGS:
        return "args"
    if was_dsml and not is_closing and (attrs.get("name") or keyword):
        # A DSML tag carrying a keyword is normally the `name`-as-tag-name dialect
        # (`<|DSML|read_file>`), so it reads as a call. A keyword beginning with `_`
        # is not that: it is the tail of `tool_call`/`function_call` left behind when
        # the checkpoint truncated a closer, observed 2026-09-26 as calls named
        # `_call` and `_placeholder`. No tool name starts with `_`, and a `name=`
        # attribute settles it, so only the bare fragment is refused.
        if attrs.get("name") or not keyword.startswith("_"):
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
        if frame["kind"] == "args" and keyword in _ARG_CONTAINER_TAGS:
            break
        idx -= 1
    else:
        return

    frame = stack.pop(idx)
    if frame["kind"] == "args":
        # The container carries the arguments object itself, so its members are the
        # call's parameters, merged whole rather than filed under one key. The body
        # is JSON, parsed with the same repair the bracketed forms use: the
        # checkpoint drops a closing brace there too, and a container whose braces
        # do not balance would otherwise contribute nothing at all.
        body = "".join(frame["raw"]).strip()
        parsed = None
        if body:
            try:
                parsed = json.loads(body)
            except Exception:
                parsed = repair_truncated_json(body)
        if isinstance(parsed, dict):
            frame["args"].update(parsed)
        for parent in reversed(stack):
            if parent["kind"] == "call":
                parent["args"].update(frame["args"])
                break
    elif frame["kind"] == "param":
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


def _parse_attr_arguments(value: str) -> Dict[str, Any]:
    """Read an argument object carried as an attribute value.

    The `call_tool` dialect writes `<|DSML|call_tool name="read_file" args={...}>`,
    so the object arrives as an attribute rather than as a child element. It is
    plain JSON, except that the value sat inside a `"`-quoted attribute, so the
    model's own quotes around its keys may have been consumed by `_ATTR_RE` --
    accept bare keys as well as quoted ones. A dropped closing brace lands here
    too, so the same repair the bracketed forms use applies.
    """
    value = (value or "").strip()
    if not value:
        return {}
    bare_keys = re.sub(r"([{,])\s*([A-Za-z_]\w*)\s*:", r'\1"\2":', value)
    for candidate in (value, bare_keys):
        try:
            parsed = json.loads(candidate)
        except Exception:
            parsed = repair_truncated_json(candidate)
        if isinstance(parsed, dict):
            return parsed
    return {}


def parse_dsml_tool_calls(content: str) -> Tuple[List[dict], Optional[int]]:
    """Extract DSML tool calls; also return the offset where the syntax begins."""
    calls: List[dict] = []
    stack: List[dict] = []
    first: Optional[int] = None
    cursor = 0

    for match in _TAG_RE.finditer(content):
        if stack and stack[-1]["kind"] == "param":
            stack[-1]["buf"].append(content[cursor:match.start()])
        elif stack and stack[-1]["kind"] == "args":
            stack[-1]["raw"].append(content[cursor:match.start()])
        cursor = match.end()

        keyword, is_closing, attrs, was_dsml = split_dsml_tag(fold_dsml_syntax(match.group(0)))
        kind = classify_dsml_tag(keyword, is_closing, attrs, was_dsml)
        if kind == "call":
            name = attrs.get("name") or (keyword if keyword not in _CALL_TAGS else None)
            frame = {"kind": "call", "name": name, "args": {}}
            # The `call_tool` dialect carries the whole argument object as an
            # attribute rather than as a child element; fold it in so the call is
            # not emitted with empty arguments.
            for args_key in _CALL_ARGS_KEYS:
                if args_key in attrs:
                    frame["args"].update(_parse_attr_arguments(attrs[args_key]))
                    break
            stack.append(frame)
            if first is None:
                first = match.start()
        elif kind == "param":
            stack.append({"kind": "param", "name": attrs.get("name") or "arg", "buf": []})
            if first is None:
                first = match.start()
        elif kind == "args":
            stack.append({"kind": "args", "args": {}, "raw": []})
            if first is None:
                first = match.start()
        elif kind == "close":
            _close_dsml_frame(stack, keyword, calls)

    if stack and stack[-1]["kind"] == "param":
        stack[-1]["buf"].append(content[cursor:])
    elif stack and stack[-1]["kind"] == "args":
        stack[-1]["raw"].append(content[cursor:])
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
        "- The arguments must be strict JSON: one object, no trailing commas, no comments.\n"
        "- Escape every double quote inside a string value as \\\", and write a newline inside a string as \\n. Never emit a literal line break, tab, or other control character inside a string.\n"
        '- Worked example: to pass the text  print("hi")  then a new line  use the JSON string value  "print(\\"hi\\")\\n".\n'
        "- If you need to call multiple tools, you can output multiple <tool_call> blocks or a JSON array of tool calls.\n"
        "- Every <tool_call> block must contain exactly one JSON object. Never emit an empty "
        "block, and never repeat the opening tag.\n"
        "- You may explain your thoughts or insights before the <tool_call> block.\n"
        "- Close the last block with </tool_call> and stop there. Do not emit any further tags, tool outputs, or repeat closing tags.\n"
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
    args_key = None
    for k in ("arguments", "parameters", "args", "input", "action_input"):
        if k in d:
            args = d[k]
            args_key = k
            break
    if args is None:
        other_keys = {k: v for k, v in d.items() if k not in _CALL_META_KEYS}
        args = other_keys if other_keys else {}
    elif isinstance(args, dict):
        # The model sometimes hoists a parameter *out* of `arguments` and leaves it
        # as a sibling key, e.g. {"name": …, "arguments": {"content": …},
        # "file_path": …}. Taking `arguments` verbatim drops it, so the client sees
        # a call missing a required property. Fold the siblings back in, never
        # overwriting a key the model did place inside `arguments`.
        hoisted = {
            k: v for k, v in d.items()
            if k not in _CALL_META_KEYS and k != args_key and k not in args
        }
        if hoisted:
            args = {**args, **hoisted}
    return name, args


def _unescaped_structure(text: str) -> str:
    """Drop string contents so brace counting only sees structural characters."""
    out = []
    in_str = False
    escaped = False
    for ch in text:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
                out.append(ch)
        elif ch == '"':
            in_str = True
            out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def _open_structure(structural: str) -> List[str]:
    """Open brackets that a structurally-only text leaves unclosed, outermost first."""
    stack: List[str] = []
    for ch in structural:
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    return stack


_CLOSER_FOR = {"{": "}", "[": "]"}


def _fragment_ends(text: str, start: int):
    """Candidate ends of the JSON value beginning at `start`, nearest first.

    A value whose own closer is missing is followed by wrapper markup and, often,
    sibling calls. Handing `repair_truncated_json` the rest of the completion makes
    it pick the *last* structural character in the remainder, which lies beyond that
    markup, and the repair then fails on bytes that were never part of the value --
    observed 2026-09-26 on the first of seven calls, whose `write_file` object never
    closed because the checkpoint replaced its `}` with `</|DSML|...>`.

    Only closing-tag positions are offered, never interior braces: a cut at a brace
    yields a fragment that still parses while having *silently dropped* part of the
    value, which is the one outcome the repair is written to avoid. A cut inside a
    string leaves that string unterminated, so the candidate fails and the search
    moves outward. The end of the text is offered last, keeping the single-value
    case exactly as it was.
    """
    limit = len(text)
    for match in re.finditer(r"</", text[start:]):
        yield start + match.start()
    yield limit


def repair_truncated_json(text: str) -> Optional[Any]:
    """Parse `text` as JSON, tolerating a JSON value cut short by a dropped closer.

    The checkpoint occasionally emits a tool-call argument object with its outer
    closing brace missing -- the stream ends one character early -- so
    `raw_decode` fails on the very last delimiter. Appending missing closers (in
    nesting order) recovers the call; string contents are ignored so a `{` inside
    a description cannot fool the count. The value is typically followed by
    wrapper markup (`</tool_call>`) that is not part of it, so the fragment is cut
    at its last structural character first; no earlier cut is tried, since
    truncating at an interior closer would silently drop part of the value.
    Returns None when no bounded repair makes a value parse.
    """
    try:
        return json.loads(text, strict=False)
    except Exception:
        pass

    decoder = json.JSONDecoder(strict=False)
    cuts = [len(text)]
    for pos in range(len(text) - 1, -1, -1):
        if text[pos] in ("}", "]", '"'):
            if pos + 1 != len(text):
                cuts.append(pos + 1)
            break

    for cut in cuts:
        fragment = text[:cut]
        stack = _open_structure(_unescaped_structure(fragment))
        remaining = [_CLOSER_FOR[opener] for opener in reversed(stack)]
        while True:
            try:
                obj, _end = decoder.raw_decode(fragment + "".join(remaining))
                return obj
            except Exception:
                if not remaining:
                    break
                remaining.pop()
    return None


_RAW_LOG = os.getenv("TABBY_PROXY_RAW_LOG", "").strip().lower() in ("1", "true", "yes", "on")
_RAW_MAX = int(os.getenv("TABBY_PROXY_RAW_LOG_CHARS", "20000") or "20000")


def _tool_required_params(name: str, tools: Optional[list]) -> Optional[set]:
    """Required argument names the request declares for `name`, else None if unknown."""
    if not tools:
        return None
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function")
        if not isinstance(func, dict) or func.get("name") != name:
            continue
        schema = func.get("parameters")
        if isinstance(schema, dict) and isinstance(schema.get("required"), list):
            return set(schema["required"])
        return set()
    return None


def _declared_tool_names(tools: Optional[list]) -> Optional[set]:
    """Tool names the request declares, else None if the request declared none."""
    if not tools:
        return None
    names = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function")
        if isinstance(func, dict) and func.get("name"):
            names.add(func["name"])
    return names or None


def _log_tool_call_shapes(calls: Optional[List[dict]], raw: str, tools: Optional[list]) -> None:
    """Flag extracted calls the client cannot dispatch, or that dropped a required parameter.

    Two shapes are worth a warning. A call whose *name* is not among the tools the
    request declared cannot be dispatched at all -- the model invented a name such
    as `comment_end`, and the schema comparison below cannot see it, because an
    unknown name has no schema to compare against. A call whose name *is* declared
    can still be unusable: the model occasionally emits an argument outside the
    arguments object, or omits it outright, and the client rejects the call with
    `invalid_tool_params`. Both are detectable here, where the source is identifiable.
    """
    declared = _declared_tool_names(tools)
    batch = len(calls or [])
    for call in calls or []:
        func = call.get("function") or {}
        name = func.get("name")
        if declared is not None and name not in declared:
            # The batch size goes in the warning so a row is self-describing: the
            # count is otherwise only in the separate INFO line, and the width is
            # what a query wants to correlate against (SPECIFIC.md 6).
            logger.warning(
                f"Tool call {name!r} is not one of the {len(declared)} tools this request "
                f"declared (batch={batch}); the client cannot dispatch it"
            )
            if _RAW_LOG:
                logger.warning(f"Raw tool-call payload: {raw[:_RAW_MAX]!r}")
            continue
        try:
            args = json.loads(func.get("arguments") or "{}")
        except Exception:
            continue
        if not isinstance(args, dict):
            continue
        required = _tool_required_params(name, tools)
        if required is None:
            continue
        missing = sorted(required - set(args))
        if missing:
            logger.warning(
                f"Tool call {name} is missing required parameter(s) {missing}; "
                f"emitted keys were {sorted(args)}"
            )
            if _RAW_LOG:
                logger.warning(f"Raw tool-call payload: {raw[:_RAW_MAX]!r}")


# A repetition loop of *opening* tags with nothing inside them. Captured live
# (2026-09-26, analysis-qt6 sessions e8ee13a8 and df6f0ecc) after an upstream 500:
# a completion of ~900 tokens that is nothing but `<tool_call>` repeated, with no
# closer, no DSML, no JSON and no arguments. There is no call in it, so it is
# prose and must be treated as prose.
_TOOL_CALL_OPEN_TAG_RE = re.compile(r"<\s*tool_calls?\s*>", re.IGNORECASE)


def _strip_repetition_tail(text: Optional[str]) -> Optional[str]:
    """Drop a trailing run of empty `<tool_call>` openers from `text`.

    The run is a repetition loop, not a call: it holds no JSON, no argument and no
    closer, so nothing in it can be dispatched. Only a run is removed -- an opener
    with anything but further openers after it is left for the caller's extraction
    to read as the marker it is, and the text before the run is returned intact
    (the visible answer, which the client should still see).

    Without this the run latches the streaming hold-back, `extract_tool_calls`
    finds no call in it, and the turn is answered as `finish_reason:
    "tool_calls"` with `tool_calls: null`. Qwen Code reads that pair as a
    malformed call and aborts the turn, so a repetition loop costs a session.
    """
    if not text:
        return text
    # A stream cut mid-tag leaves a half-written opener (`<tool`); it can never
    # become a call, so ignore it when judging what the run contains.
    body = text
    cut = text.rfind("<")
    if cut != -1:
        fragment = text[cut + 1:].strip().lower()
        if fragment and ("tool_call".startswith(fragment) or "tool_calls".startswith(fragment)):
            body = text[:cut]
    if not body:
        return text
    # Walk the openers backwards from the end, extending the run while only
    # whitespace separates one from the next. The first gap that holds anything
    # else is the end of the run: whatever precedes it is left alone, so an opener
    # that begins a real call still reaches the extraction.
    run_start = None
    for match in reversed(list(_TOOL_CALL_OPEN_TAG_RE.finditer(body))):
        gap_end = run_start if run_start is not None else len(body)
        if body[match.end():gap_end].strip():
            break
        run_start = match.start()
    if run_start is None:
        return text
    stripped = body[:run_start].rstrip()
    if not stripped:
        # Nothing but the run: report it so the journal shows the loop rather
        # than a silent empty completion. Bounded, since it can be huge.
        logger.warning("Completion was only repeated empty tool-call open tags; dropping the repetition")
        return None
    logger.warning("Dropped a repetition of empty tool-call open tags after the visible answer")
    return stripped


def extract_tool_calls(content: str) -> Tuple[Optional[List[dict]], Optional[str]]:
    """Robustly extracts tool calls from content in JSON, DSML, XML, or pseudo formats."""
    if not content:
        return None, content

    content = _strip_repetition_tail(content) or ""

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
                parsed = json.loads(args, strict=False)
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
    # strict=False tolerates the raw control characters the checkpoint
    # sometimes leaves inside string values (see _log_tool_call_shapes' sibling
    # note): a literal newline in a multi-line `content` is otherwise rejected
    # as `Invalid control character` and the whole call is lost.
    decoder = json.JSONDecoder(strict=False)
    start_pos = first_start if first_start < len(content) else 0

    # The typographic fold is a per-bracket alternative, not a rewrite of the whole
    # completion: a string value may legitimately contain `“`, so the model's own
    # glyph must get the first attempt and the folded reading is only a fallback.
    # Folded once here rather than inside the scan: the completion can be hundreds of
    # kilobytes and the scan visits every bracket in it.
    _foldable = any(g in content for g in _TYPOGRAPHIC_QUOTES)
    readings = (content, fold_typographic(content)) if _foldable else (content,)
    decoder_folded = json.JSONDecoder(strict=False)

    idx = start_pos
    while idx < len(content):
        ch = content[idx]
        if ch in ("{", "["):
            obj: Any = None
            end_pos = idx
            for decoder, text in ((decoder, readings[0]), (decoder_folded, readings[-1])):
                try:
                    obj, end_pos = decoder.raw_decode(text, idx)
                    break
                except Exception:
                    obj = None
            if obj is None:
                # Either reading may have failed on a value whose closer is absent
                # from the tail. Ask `repair_truncated_json` to bound the fragment
                # itself: handed the rest of the completion it finds a *later*
                # structural character first and then fails, losing a call that
                # parses once its own object is isolated -- observed 2026-09-26 on
                # the first of seven calls, whose `write_file` object simply never
                # closed.
                repaired = None
                used_limit = len(content)
                for text in readings:
                    for limit in _fragment_ends(text, idx):
                        repaired = repair_truncated_json(text[idx:limit])
                        if repaired is not None:
                            used_limit = limit
                            break
                    if repaired is not None:
                        break
                if repaired is None:
                    idx += 1
                    continue
                # Resume after the repaired value, not at the end of the text: the
                # calls that follow a malformed one are still parseable, and the
                # 2026-09-26 completion that motivated this carried six of them
                # after the truncated first.
                obj, end_pos = repaired, used_limit
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
        idx += 1

    # 3. XML function format <function=NAME>...
    for fn_m in re.finditer(r"<function=([^>]+)>(.*?)(?:</function>|$)", content, re.DOTALL):
        first_start = min(first_start, fn_m.start())
        fn_name = fn_m.group(1).strip()
        params = {}
        for pk, pv in re.findall(r"<parameter=([^>]+)>\s*(.*?)\s*(?:</parameter>|$)", fn_m.group(2), re.DOTALL):
            val = pv.strip()
            try:
                params[pk.strip()] = json.loads(val, strict=False)
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


def process_message_tools_and_thinking(msg: dict, choice: dict, tools: Optional[list] = None, native: bool = False):
    """Parses thinking tags and extracts tool calls from message content or reasoning_content.

    `native` says the caller is also holding a call that arrived over the upstream's
    native `tool_calls` channel, which this function cannot see because it only reads
    the two text fields. It suppresses the "no call was parsed" verdict and leaves
    the finish reason for the caller to settle after it has merged the native call.
    """
    content = msg.get("content")
    reasoning = msg.get("reasoning_content")

    # Kept for the diagnostic dump at the foot of this function only: the strip just
    # below can delete an *entire* completion, and the deleted bytes are precisely
    # what a `tool_calls` turn needs to show. Everything else reads the stripped
    # fields, which is what the scanner actually examined.
    original_content, original_reasoning = content, reasoning

    # Both fields are sanitised here as well as inside extract_tool_calls, because
    # this is what lands in `msg`: a repetition loop that yields no call has to be
    # dropped from the answer the client sees, not left in it as raw markup.
    content = _strip_repetition_tail(content)
    reasoning = _strip_repetition_tail(reasoning)

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
            _log_tool_call_shapes(extracted_calls, content, tools)
            logger.info(f"Intercepted and parsed {len(extracted_calls)} tool call(s) from content: {[c['function']['name'] for c in extracted_calls]}")
            return

    # If not in content, check reasoning_content (in case model emitted tools before </think>)
    if reasoning and not extracted_calls:
        extracted_calls, cleaned_reasoning = extract_tool_calls(reasoning)
        if extracted_calls:
            msg["tool_calls"] = extracted_calls
            msg["reasoning_content"] = cleaned_reasoning
            choice["finish_reason"] = "tool_calls"
            _log_tool_call_shapes(extracted_calls, reasoning, tools)
            logger.info(f"Intercepted and parsed {len(extracted_calls)} tool call(s) from reasoning: {[c['function']['name'] for c in extracted_calls]}")
            return


    if not extracted_calls and not native:
        raw = content or reasoning or ""
        marked = "DSML" in raw or "<tool_call" in raw or "\\uff5c" in raw
        if marked:
            logger.warning(f"Tool-call syntax present but unparsed: {raw[:300]!r}")
        # The raw dump is deliberately NOT gated on `marked`. A completion that
        # carried a `tool_calls` finish reason and yielded no call is precisely the
        # case the wire bytes are wanted for, and it is also the case where *no*
        # marker we recognise has to be present -- so gating on one made the family
        # that most needs the bytes the one that could not capture them. Measured
        # 2026-09-26 (session 6ce029e1): a first turn announced a call, parsed to
        # nothing, and this dump stayed silent on the marker test alone. The dump
        # below covers both fields the parser examined, not just the first non-empty
        # one: the ` thinking` branch above writes `msg["content"]` as the
        # *post*-think tail, so the text the scanner was handed and the field that
        # held the call need not be the same one.
        if _RAW_LOG and (marked or choice.get("finish_reason") == "tool_calls"):
            # Dump **both** fields, because the parser tried both: a completion whose
            # text field is clean prose can still carry the failed call in
            # `reasoning_content`, and dumping only the first non-empty field hid
            # exactly that case the moment this gate moved here (observed
            # 2026-09-26, session aba2e4e2: the dump stopped at the bare
            # announcement, three newlines and nothing else). Where the repetition
            # strip changed a field, the *emitted* text is dumped as well -- that
            # strip can delete a whole completion, and a `tool_calls` turn that
            # stripped to nothing is otherwise invisible.
            for field, shown, emitted in (
                ("completion", content, original_content),
                ("reasoning", reasoning, original_reasoning),
            ):
                if not emitted:
                    continue
                logger.warning(f"Raw unparsed {field}: {(shown or '')[:_RAW_MAX]!r}")
                if shown != emitted:
                    logger.warning(f"Raw unparsed {field} before the repetition strip: {emitted[:_RAW_MAX]!r}")
        # `tool_calls` with nothing in it is the one pair Qwen Code treats as a
        # malformed call and aborts the turn on: the client sees a finish reason
        # it cannot act on and stops the session. If the marker could not be
        # turned into a call, answer as plain text instead, so the turn survives.
        if choice.get("finish_reason") == "tool_calls":
            logger.warning(
                "Upstream said tool_calls but no call was parsed; reporting stop so the "
                "client does not abort the turn on an empty call"
            )
            choice["finish_reason"] = "stop"


# ---------------------------------------------------------------------------
# Upstream resilience and incremental streaming
#
# Two problems live here:
#
#   1. TabbyAPI answers 503 - and, while a model is loading, occasionally 502
#      or 529 - for a few seconds at a time. One such answer used to end the
#      turn outright, so every upstream call is now retried with backoff.
#
#   2. Requests carrying tools used to be forced non-streaming so the proxies
#      could parse the whole completion before answering, leaving the client
#      silent for the entire generation. The tool path now relays upstream
#      deltas as they arrive, holding back a short tail of content and of
#      reasoning_content. extract_tool_calls discards everything from the first
#      tool-syntax marker onwards and keeps everything before it, so holding
#      back the last _HOLDBACK characters is enough to guarantee a marker is
#      never emitted half-formed. If no marker appears, the held tail is
#      flushed at the end.
# ---------------------------------------------------------------------------

_TOOL_MARKER_RE = re.compile(
    r"<\s*/?\s*(?:[|\uff5c]|\\uff5c|\\u2581)?\s*(?:dsml|tool_calls?\b|tool\b|param(?:eter)?s?\b|function|invoke)"
    r"|\[tool_call\s*:"
    r"|```(?:tool_call|json)?\s*\{"
    r"|</?think>",
    re.IGNORECASE,
)
# Longest run of characters that can stand between the "<" of a marker and the
# keyword that completes it. The hold-back window has to cover this so a marker
# split across two deltas is never half-emitted.
_MARKER_PREFIX_LIMIT = 8
_HOLDBACK = 32
_MAX_ATTEMPTS = 5
_RETRY_STATUS = {429, 500, 502, 503, 504, 529}
_UPSTREAM_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
_CONNECT_ERROR = '{"error": "Could not connect to TabbyAPI downstream server."}'


def _find_tool_call_marker(text: str) -> int:
    """Index of the earliest tool-syntax marker in text, or -1 when there is none.

    A marker is only recognised once its keyword has arrived, so a partial one
    (e.g. "<tool_c") reads as prose here and stays inside the hold-back window.
    """
    match = _TOOL_MARKER_RE.search(text)
    return match.start() if match else -1


def _backoff(attempt: int) -> float:
    return float(min(2 ** attempt, 16))


async def post_upstream_json(client: httpx.AsyncClient, url: str, body: dict, headers: dict) -> httpx.Response:
    """POST a buffered completion, retrying connect failures and transient statuses."""
    last_error: Optional[Exception] = None
    for attempt in range(_MAX_ATTEMPTS):
        if attempt:
            await asyncio.sleep(_backoff(attempt))
        try:
            response = await client.post(url, json=body, headers=headers)
        except httpx.TransportError as exc:
            last_error = exc
            logger.warning(f"Upstream connection failed ({attempt + 1}/{_MAX_ATTEMPTS}): {exc}")
            continue
        if response.status_code in _RETRY_STATUS and attempt < _MAX_ATTEMPTS - 1:
            logger.warning(f"Upstream returned {response.status_code}, retrying ({attempt + 1}/{_MAX_ATTEMPTS})")
            await response.aclose()
            continue
        return response
    raise last_error if last_error else httpx.ConnectError("upstream is unreachable")


async def open_upstream_stream(client: httpx.AsyncClient, url: str, body: dict, headers: dict):
    """Open an upstream SSE stream, retrying connect failures and transient statuses.

    Returns the still-entered stream context along with its response so the
    caller can inspect the status before relaying or rejecting it.
    """
    last_error: Optional[Exception] = None
    for attempt in range(_MAX_ATTEMPTS):
        if attempt:
            await asyncio.sleep(_backoff(attempt))
        stream_ctx = client.stream("POST", url, json=body, headers=headers)
        try:
            response = await stream_ctx.__aenter__()
        except httpx.TransportError as exc:
            last_error = exc
            logger.warning(f"Upstream stream failed to open ({attempt + 1}/{_MAX_ATTEMPTS}): {exc}")
            continue
        if response.status_code in _RETRY_STATUS and attempt < _MAX_ATTEMPTS - 1:
            logger.warning(f"Upstream stream returned {response.status_code}, retrying ({attempt + 1}/{_MAX_ATTEMPTS})")
            await stream_ctx.__aexit__(None, None, None)
            continue
        return stream_ctx, response
    raise last_error if last_error else httpx.ConnectError("upstream is unreachable")


def _sse_chunk(resp_id: str, created: int, model: str, index: int, delta: dict, finish_reason: Optional[str] = None) -> str:
    payload = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _sse_usage_chunk(resp_id: str, created: int, model: str, usage: dict) -> str:
    """Terminal usage-only chunk, sent just before [DONE].

    Carries an empty choices list, matching what OpenAI and ollama emit for
    `stream_options.include_usage`; the client reads the counts from here.
    """
    payload = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage,
    }
    return f"data: {json.dumps(payload)}\n\n"


def _unstreamed_remainder(parsed: Optional[str], raw: str, sent: int, field: str) -> str:
    """The part of a parsed field that incremental streaming has not delivered yet."""
    spoken = raw[:sent]
    parsed = parsed or ""
    if not parsed.startswith(spoken):
        logger.warning(f"Streamed {field} prefix diverged from the parsed value; dropping the unsent remainder")
        return ""
    return parsed[len(spoken):]


def _merge_native_tool_calls(message: dict, native: dict) -> List[dict]:
    """Turn accumulated native `delta.tool_calls` fragments into message tool calls.

    TabbyAPI has already parsed the checkpoint's DSML into an OpenAI-shaped call,
    so the arguments are valid JSON and are passed through verbatim: re-serialising
    them through the scanner's own path would risk changing bytes the upstream had
    already validated. A native call takes precedence over anything scraped out of
    the text fields -- the upstream's parse is authoritative, and a text call that
    happens to sit beside it is usually the raw DSML the upstream just consumed.
    """
    merged = []
    for slot in native.values():
        if not slot["name"]:
            continue
        arguments = slot["arguments"] or "{}"
        merged.append({
            "id": slot["id"] or f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": slot["name"], "arguments": arguments},
        })
    if not merged:
        return message.get("tool_calls") or []
    message["tool_calls"] = merged
    return merged


async def relay_upstream_stream(stream_ctx, response, client: httpx.AsyncClient) -> AsyncIterator[bytes]:
    """Forward upstream SSE bytes untouched (used when no tools are in play)."""
    try:
        async for piece in response.aiter_raw():
            yield piece
    finally:
        await stream_ctx.__aexit__(None, None, None)
        await client.aclose()


async def stream_tools_response(stream_ctx, response, client: httpx.AsyncClient, requested_model: Optional[str], tools: Optional[list] = None) -> AsyncIterator[str]:
    """Relay a tool-bearing completion as SSE, intercepting native tool syntax."""
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    model = requested_model or ""
    index = 0
    content_text = ""
    reasoning_text = ""
    native_tool_calls = {}
    sent_content = 0
    sent_reasoning = 0
    held = False
    role_sent = False
    finish_reason = "stop"
    eos_reason = None
    usage = None

    try:
        async for line in response.aiter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                event = json.loads(payload)
            except Exception:
                continue

            if isinstance(event.get("usage"), dict):
                usage = event["usage"]

            for event_choice in event.get("choices", []):
                if event_choice.get("finish_reason"):
                    finish_reason = event_choice["finish_reason"]
                    eos_reason = event_choice.get("eos_reason", eos_reason)
                delta = event_choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    reasoning_text += delta["reasoning_content"]
                if delta.get("content"):
                    content_text += delta["content"]
                # TabbyAPI parses the checkpoint's native DSML server-side and, in
                # streaming mode, delivers the call here rather than in `content`.
                # Reading only `content`/`reasoning_content` dropped it: the scanner
                # then found nothing, `finish_reason` stayed `tool_calls` and was
                # downgraded to `stop`, so the client got neither a call nor a
                # finished text turn and the session wedged (observed 2026-09-26,
                # #431, session aba2e4e2). Collect the native fragments so they can
                # be emitted once the stream ends.
                for fragment in delta.get("tool_calls") or []:
                    slot = native_tool_calls.setdefault(fragment.get("index", 0), {
                        "id": None, "type": "function", "name": "", "arguments": "",
                    })
                    if fragment.get("id"):
                        slot["id"] = fragment["id"]
                    function = fragment.get("function") or {}
                    if function.get("name"):
                        slot["name"] += function["name"]
                    if function.get("arguments"):
                        slot["arguments"] += function["arguments"]

                if not role_sent:
                    role_sent = True
                    yield _sse_chunk(resp_id, created, model, index, {"role": "assistant"})

                if not held and (
                    _find_tool_call_marker(content_text) != -1
                    or _find_tool_call_marker(reasoning_text) != -1
                ):
                    held = True

                if held:
                    continue

                limit = max(0, len(reasoning_text) - _HOLDBACK)
                if limit > sent_reasoning:
                    yield _sse_chunk(resp_id, created, model, index, {"reasoning_content": reasoning_text[sent_reasoning:limit]})
                    sent_reasoning = limit

                limit = max(0, len(content_text) - _HOLDBACK)
                if limit > sent_content:
                    yield _sse_chunk(resp_id, created, model, index, {"content": content_text[sent_content:limit]})
                    sent_content = limit
    finally:
        await stream_ctx.__aexit__(None, None, None)
        await client.aclose()

    if not role_sent:
        yield _sse_chunk(resp_id, created, model, index, {"role": "assistant"})

    message = {"role": "assistant", "content": content_text or None, "reasoning_content": reasoning_text or None}
    choice = {"index": index, "finish_reason": finish_reason}
    process_message_tools_and_thinking(message, choice, tools, native=bool(native_tool_calls))
    final_content = message.get("content")
    final_reasoning = message.get("reasoning_content")
    # A native call settles the finish reason: the upstream said `tool_calls`
    # because it parsed one, so the empty-call downgrade inside the processor --
    # which only saw the text fields -- must not undo it.
    tool_calls = _merge_native_tool_calls(message, native_tool_calls)
    if tool_calls and finish_reason == "tool_calls":
        choice["finish_reason"] = "tool_calls"

    for field, parsed, raw, sent in (
        ("reasoning_content", final_reasoning, reasoning_text, sent_reasoning),
        ("content", final_content, content_text, sent_content),
    ):
        remainder = _unstreamed_remainder(parsed, raw, sent, field)
        if remainder:
            yield _sse_chunk(resp_id, created, model, index, {field: remainder})

    for tool_index, tool_call in enumerate(tool_calls or []):
        yield _sse_chunk(resp_id, created, model, index, {"tool_calls": [{
            "index": tool_index,
            "id": tool_call["id"],
            "type": "function",
            "function": {
                "name": tool_call["function"]["name"],
                "arguments": tool_call["function"]["arguments"],
            },
        }]})

    if not tool_calls and not final_content and not final_reasoning:
        logger.warning(f"Upstream produced an empty completion (finish_reason={choice.get('finish_reason')!r}, eos_reason={eos_reason!r})")

    yield _sse_chunk(resp_id, created, model, index, {}, choice.get("finish_reason", finish_reason))
    if usage:
        yield _sse_usage_chunk(resp_id, created, model, usage)
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
        # A second opener straight after an empty one is a loop, never two calls:
        # a real second call is preceded by the first one's closer. Stopping here
        # ends the run at the source, and upstream then reports `stop` rather than
        # `tool_calls`, so the client never sees a call that is not one.
        "<tool_call>\n<tool_call>",
        "<tool_call>\r\n<tool_call>",
        "<tool_call><tool_call>",
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

    url = f"{TABBY_API_URL}/v1/chat/completions"
    client = httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT)

    if is_streaming:
        stream_body = dict(patched_body)
        stream_body["stream"] = True
        if has_tools:
            # The proxy synthesises the SSE framing, but usage is upstream's
            # accounting for the turn and survives it: keep include_usage on so
            # the counts are available when the intercepted calls are emitted.
            opts = stream_body.get("stream_options")
            if not isinstance(opts, dict):
                opts = {}
            opts.setdefault("include_usage", True)
            stream_body["stream_options"] = opts

        try:
            stream_ctx, response = await open_upstream_stream(client, url, stream_body, headers)
        except httpx.TransportError:
            await client.aclose()
            return Response(content=_CONNECT_ERROR, status_code=502, media_type="application/json")

        if response.status_code != 200:
            upstream_error = await response.aread()
            await stream_ctx.__aexit__(None, None, None)
            await client.aclose()
            return Response(content=upstream_error, status_code=response.status_code, media_type="application/json")

        if has_tools:
            return StreamingResponse(
                stream_tools_response(stream_ctx, response, client, requested_model, tools),
                media_type="text/event-stream"
            )

        return StreamingResponse(
            relay_upstream_stream(stream_ctx, response, client),
            media_type="text/event-stream"
        )

    try:
        try:
            response = await post_upstream_json(client, url, patched_body, headers)
        except httpx.TransportError:
            return Response(content=_CONNECT_ERROR, status_code=502, media_type="application/json")

        if response.status_code != 200:
            return Response(content=response.content, status_code=response.status_code, media_type="application/json")

        try:
            resp_data = response.json()
        except Exception:
            return Response(content=response.content, status_code=response.status_code, media_type="application/json")

        # Parse message content / reasoning for tool calls
        for choice in resp_data.get("choices", []):
            msg = choice.get("message", {})
            # TabbyAPI can hand back a native call here (its `deepseek_v4` format
            # parses the checkpoint's DSML server-side), where the text fields hold
            # nothing to scrape. Tell the processor so, and leave the finish reason
            # for the guard below: with the processor silenced the call is already in
            # `msg`, so the guard sees it and holds.
            native = bool(msg.get("tool_calls"))
            process_message_tools_and_thinking(msg, choice, tools, native=native)
            # Same guard as the streaming path: a finish reason of `tool_calls`
            # with an empty list is read as a malformed call and ends the turn.
            if choice.get("finish_reason") == "tool_calls" and not msg.get("tool_calls"):
                logger.warning(
                    "Upstream said tool_calls but no call was parsed; reporting stop so the "
                    "client does not abort the turn on an empty call"
                )
                choice["finish_reason"] = "stop"

        return Response(
            content=json.dumps(resp_data),
            status_code=200,
            media_type="application/json"
        )
    finally:
        await client.aclose()


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
    # The checkpoint occasionally drops the argument object's outer closing
    # brace -- the stream ends one byte early -- and the wrapper markup still
    # follows it. Captured verbatim from a live qwen-code session (2026-09-25)
    # as an ask_user_question call; the brace count there was short by exactly
    # one and every brace sat outside a string.
    v4_args = (
        '{"questions": [{"question": "Which example?", "header": "Type", '
        '"options": [{"label": "GUI counter app", "description": "tkinter vs egui"}, '
        '{"label": "CLI tool", "description": "wc in both languages"}]}]}'
    )
    v4 = (
        "Sure, let me confirm what you want.\n\n"
        "<tool_call>\n"
        '{"name": "ask_user_question", "arguments": ' + v4_args + "\n"
        "</tool_call>\n"
        "</tool_call>"
    )
    return [
        ("v1 name attribute + bare closes", v1, [("read_file", {"file_path": "/tmp/stage123.txt"})], "I'll read the brief for you."),
        (
            "v2 tool/parameter tags",
            v2,
            [("read_file", {"file_path": "/tmp/stage121.txt"}), ("glob", {"pattern": "stage121.txt"})],
            None,
        ),
        ("v3 escaped bars (\\uff5c)", v3, [("read_file", {"file_path": "/tmp/stage123.txt"})], "I'll read the brief for you."),
        (
            "v4 json missing one closing brace",
            v4,
            [("ask_user_question", json.loads(v4_args))],
            "Sure, let me confirm what you want.",
        ),
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

    # The hold-back window only protects the stream while it covers the longest
    # marker prefix; once a marker is whole, latching on it takes over.
    if _HOLDBACK < _MARKER_PREFIX_LIMIT:
        print(f"FAIL hold-back window {_HOLDBACK} is shorter than the marker prefix limit ({_MARKER_PREFIX_LIMIT})")
        failures += 1
    else:
        print(f"ok   hold-back window {_HOLDBACK} covers the marker prefix limit ({_MARKER_PREFIX_LIMIT})")

    fw = _SELFTEST_FW
    lt, gt, low = "<", ">", chr(0x2581)
    marker_cases = [
        ("no marker", "All done, nothing to call.", -1),
        ("bare tag", lt + "tool_call" + gt, 0),
        ("marker after prose", "ab" + lt + "tool_call" + gt, 2),
        ("case-insensitive", "ab" + lt + "|DSML|read_file" + gt, 2),
        ("escaped bars", "ab" + lt + "\\uff5cDSML\\uff5c tool" + gt, 2),
        ("name attribute dialect", "ab" + lt + fw + "DSML" + fw + ' name="read_file"' + gt, 2),
        ("wrapper closer", "ab" + lt + "/" + fw + "DSML" + fw + "tool" + gt, 2),
        ("escaped wrapper closer", "ab" + lt + "/\\uff5cDSML\\uff5c tool" + gt, 2),
        ("end of sentence closer", "ab" + lt + "/" + fw + "tool" + low + "calls" + low + "end" + fw + gt, 2),
        ("xml function form", "ab" + lt + "function=glob" + gt, 2),
        ("bracketed form", "ab[tool_call: read_file]", 2),
        ("inline think tag", "ab" + lt + "think" + gt, 2),
        ("closing think tag", "ab" + lt + "/think" + gt, 2),
        ("fenced json call", "ab```json" + chr(10) + '{"name": "x"}', 2),
        ("parameter tag", 'ab' + lt + 'parameter name="p"' + gt + "v", 2),
        ("half keyword", "ab" + lt + "tool_c", -1),
        ("half escaped bar", "ab" + lt + "\\uff5c", -1),
        ("half raw bar", "ab" + lt + fw, -1),
        ("bare fence", "ab```", -1),
        ("fence then language", "ab```json", -1),
        ("bracket without colon", "ab[tool_call", -1),
        ("ordinary code block", "ab```python" + chr(10) + "print(1)", -1),
        ("html tag in prose", "wrap it in a <div>", -1),
    ]
    for name, text, expected in marker_cases:
        got = _find_tool_call_marker(text)
        ok = got == expected
        print(f"{'ok  ' if ok else 'FAIL'} marker {name}: {got}")
        if not ok:
            failures += 1
            print(f"     expected: {expected}")

    remainder_cases = [
        ("nothing sent yet", "Hello world", "Hello world<tool_call>x", 0, "Hello world"),
        ("partly sent", "Hello world", "Hello world<tool_call>x", 5, " world"),
        ("marker-only tail", "Hello", "Hello<tool_call>", 0, "Hello"),
        ("empty after drop", None, "<tool_call>", 0, ""),
        ("divergent prefix is dropped", "World", "abc", 2, ""),
    ]
    for name, parsed, raw, sent, expected in remainder_cases:
        got = _unstreamed_remainder(parsed, raw, sent, "content")
        ok = got == expected
        print(f"{'ok  ' if ok else 'FAIL'} remainder {name}: {got!r}")
        if not ok:
            failures += 1
            print(f"     expected: {expected!r}")

    repair_cases = [
        ("missing brace", '{"a": 1', {"a": 1}),
        ("missing bracket", '{"a": [1, 2', {"a": [1, 2]}),
        ("missing both", '{"q": [{"l": "b"', {"q": [{"l": "b"}]}),
        ("brace inside a string", '{"a": "x{y", "b": 2', {"a": "x{y", "b": 2}),
        ("already complete", '{"a": 1}', {"a": 1}),
        ("wrapper markup after value", '{"a": 1}\n</tool_call>', {"a": 1}),
        ("wrapper plus missing brace", '{"a": {"b": 1}\n</tool_call>', {"a": {"b": 1}}),
        ("trailing comma is not a closer", '{"a": 1,', None),
        ("not json at all", '{"a": ,', None),
    ]
    for name, fragment, expected in repair_cases:
        got = repair_truncated_json(fragment)
        ok = got == expected
        print(f"{'ok  ' if ok else 'FAIL'} repair {name}: {got!r}")
        if not ok:
            failures += 1
            print(f"     expected: {expected!r}")

    # A string value carrying raw control characters (a literal newline in a
    # multi-line `content`) is invalid strict JSON, but is what the checkpoint
    # actually emits. Captured live 2026-09-25: an extraction returned nothing
    # over one such call, which then reached the client as raw text and halted
    # the session on its own stricter parser.
    control_cases = [
        (
            "newline inside a string",
            '<tool_call>\n{"name": "write_file", "arguments": '
            '{"file_path": "/tmp/a", "content": "one\ntwo"}}\n</tool_call>',
            "write_file",
            {"file_path": "/tmp/a", "content": "one\ntwo"},
        ),
        (
            "tab inside a string",
            '<tool_call>\n{"name": "grep_search", "arguments": '
            '{"pattern": "a\tb"}}\n</tool_call>',
            "grep_search",
            {"pattern": "a\tb"},
        ),
    ]
    for name, raw, want_name, want_args in control_cases:
        got_calls, _ = extract_tool_calls(raw)
        ok = bool(got_calls) and got_calls[0]["function"]["name"] == want_name
        if ok:
            got_args = json.loads(got_calls[0]["function"]["arguments"])
            ok = got_args == want_args
        print(f"{'ok  ' if ok else 'FAIL'} control char {name}")
        if not ok:
            failures += 1
            print(f"     expected {want_name} {want_args!r}")

    # A call whose name is not among the tools the request declared can never be
    # dispatched, and the required-parameter check cannot see it: an unknown name
    # has no schema to compare against. Captured live as one-off names such as
    # `comment_end`, `tool_invocation` and `skills`.
    name_tools = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {"type": "object", "required": ["file_path"]},
            },
        }
    ]
    name_cases = [
        ("undeclared name is flagged", "comment_end", name_tools, True),
        ("declared name is silent", "write_file", name_tools, False),
        ("no tools sent stays silent", "comment_end", None, False),
    ]
    for label, call_name, tools_arg, want_warning in name_cases:
        captured: List[str] = []
        original_warning = logger.warning
        logger.warning = lambda message, *a, **k: captured.append(str(message))
        try:
            _log_tool_call_shapes(
                [{"function": {"name": call_name, "arguments": "{}"}}], "raw", tools_arg
            )
        finally:
            logger.warning = original_warning
        got_warning = any("cannot dispatch" in message for message in captured)
        ok = got_warning == want_warning
        print(f"{'ok  ' if ok else 'FAIL'} {label}")
        if not ok:
            failures += 1
            print(f"     expected warning={want_warning}, captured {captured!r}")

    # The warning carries the batch width so one journal row answers both "which
    # invented name" and "how wide was the turn that produced it" -- without it
    # the width lives only in the separate INFO line, and a query correlating
    # width against invention (SPECIFIC.md 6) has to join two rows by hand. The
    # raw dump is a third row and must not stand in for the warning.
    width_cases = [
        (
            "batch width appears in the warning",
            [{"function": {"name": name}} for name in ("comment_end", "write_file", "glob")],
            "batch=3",
        ),
        (
            "a lone invented call reports width 1",
            [{"function": {"name": "comment_end"}}],
            "batch=1",
        ),
    ]
    for label, calls_arg, want_substring in width_cases:
        captured: List[str] = []
        original_warning = logger.warning
        logger.warning = lambda message, *a, **k: captured.append(str(message))
        try:
            _log_tool_call_shapes(calls_arg, "raw", name_tools)
        finally:
            logger.warning = original_warning
        ok = any(want_substring in message for message in captured)
        print(f"{'ok  ' if ok else 'FAIL'} {label}")
        if not ok:
            failures += 1
            print(f"     expected {want_substring!r} in {captured!r}")

    # TABBY_PROXY_RAW_LOG gates every raw dump because a dump is `_RAW_MAX` chars
    # -- 20 000 by default -- per occurrence, and it captures the payload that
    # contains an invented name. With it off, the decoded warning above still
    # answers the shape questions; only the wire bytes need the flag.
    raw_log_captured: List[str] = []
    original_warning = logger.warning
    logger.warning = lambda message, *a, **k: raw_log_captured.append(str(message))
    try:
        _log_tool_call_shapes([{"function": {"name": "comment_end"}}], "WIRE-PAYLOAD", name_tools)
    finally:
        logger.warning = original_warning
    raw_logged = any("WIRE-PAYLOAD" in message for message in raw_log_captured)
    ok = raw_logged == _RAW_LOG
    print(
        f"{'ok  ' if ok else 'FAIL'} raw dump follows the TABBY_PROXY_RAW_LOG flag: "
        f"flag={_RAW_LOG}, dumped={raw_logged}"
    )
    if not ok:
        failures += 1

    # The model sometimes leaves a parameter beside `arguments` instead of inside
    # it. Taking `arguments` verbatim dropped it, and the client then rejected the
    # call with `invalid_tool_params`. Captured live as a missing `file_path`.
    hoist_cases = [
        (
            "hoisted sibling is folded in",
            {"name": "write_file", "arguments": {"content": "x"}, "file_path": "/tmp/a"},
            {"content": "x", "file_path": "/tmp/a"},
        ),
        (
            "sibling never overwrites an inner key",
            {"name": "write_file", "arguments": {"file_path": "/inner"}, "file_path": "/outer"},
            {"file_path": "/inner"},
        ),
        (
            "meta keys are not mistaken for parameters",
            {"name": "write_file", "type": "function", "id": "c1", "arguments": {"content": "x"}},
            {"content": "x"},
        ),
    ]
    for label, shape, want_args in hoist_cases:
        _, got_args = normalize_tool_call_dict(shape)
        ok = got_args == want_args
        print(f"{'ok  ' if ok else 'FAIL'} {label}")
        if not ok:
            failures += 1
            print(f"     expected {want_args!r}, got {got_args!r}")

    # A run of empty `<tool_call>` openers with no closer is a repetition loop, not
    # a call. Left alone it latches the hold-back, extracts nothing, and is
    # answered as `finish_reason: "tool_calls"` with a null call list -- which Qwen
    # Code aborts the turn on. Captured verbatim from session e8ee13a8 on
    # 2026-09-26: 200 openers, no closer, no JSON.
    loop_only = "<tool_call>\n" * 200 + "<tool"
    loop_blob = "page` and the legs that touch it.\n" + loop_only
    repetition_cases = [
        ("pure loop is dropped", loop_only, None),
        ("loop after prose keeps the prose", loop_blob, "page` and the legs that touch it."),
        ("a single trailing opener is dropped", "All done.\n<tool_call>", "All done."),
        ("a lone opener with no argument", "<tool_call>", None),
        ("text after an opener is left alone", "Here:\n<tool_call>\nnothing", "Here:\n<tool_call>\nnothing"),
        ("a run after a real call does not touch the call",
         "Done.\n<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n<tool_call>",
         "Done.\n<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>"),
    ]
    for name, raw, expected in repetition_cases:
        got = _strip_repetition_tail(raw)
        ok = got == expected
        print(f"{'ok  ' if ok else 'FAIL'} repetition {name}: {got!r}")
        if not ok:
            failures += 1
            print(f"     expected: {expected!r}")

    # The same blob must not produce a call, and must not claim one.
    loop_calls, loop_cleaned = extract_tool_calls(loop_blob)
    ok = loop_calls is None and loop_cleaned == "page` and the legs that touch it."
    print(f"{'ok  ' if ok else 'FAIL'} repetition loop extracts no call: {loop_calls} cleaned={loop_cleaned!r}")
    if not ok:
        failures += 1

    # A `tool_calls` finish reason that yielded no call is downgraded, in both the
    # streaming and the buffered path, so the client does not abort the turn.
    downgrade_cases = [
        ("unparsed marker downgrades to stop", {"finish_reason": "tool_calls"}, "stop"),
        ("a real call keeps tool_calls", {"finish_reason": "tool_calls", "has_call": True}, "tool_calls"),
        ("a plain stop is untouched", {"finish_reason": "stop"}, "stop"),
    ]
    for name, scenario, expected in downgrade_cases:
        msg = {"content": loop_blob, "reasoning_content": None}
        if scenario.get("has_call"):
            msg["content"] = '<tool_call>\n{"name": "read_file", "arguments": {"file_path": "/tmp/a"}}\n</tool_call>'
        choice = {"finish_reason": scenario["finish_reason"]}
        original_warning = logger.warning
        logger.warning = lambda *a, **k: None
        try:
            process_message_tools_and_thinking(msg, choice, None)
        finally:
            logger.warning = original_warning
        got = choice["finish_reason"]
        ok = got == expected
        print(f"{'ok  ' if ok else 'FAIL'} finish reason {name}: {got!r}")
        if not ok:
            failures += 1
            print(f"     expected: {expected!r}")

    # The `no call was parsed` family, added 2026-09-26 after session 6ce029e1: the
    # first turn of that probe announced a call, the upstream set `tool_calls`, and
    # the proxy parsed nothing -- yet the raw dump stayed silent, because it was
    # gated on a recognised marker and this completion carried none. The dump must
    # follow the finish reason instead. The three legs below are the whole rule:
    # a markerless unparsed call dumps, an unmarked *clean* stop does not, and a
    # genuinely empty completion does not dump an empty string.
    dump_cases = [
        ("markerless unparsed call dumps raw", {"finish_reason": "tool_calls"}, "markerless", True),
        ("a clean stop dumps nothing", {"finish_reason": "stop"}, "markerless", False),
        ("an empty completion dumps nothing", {"finish_reason": "tool_calls"}, "", False),
        ("a marked unparsed call still dumps", {"finish_reason": "tool_calls"}, "marker", True),
    ]
    for name, scenario, content_kind, want_dump in dump_cases:
        if content_kind == "marker":
            # A marker the parser recognises but cannot turn into a call -- the
            # `marked` half of the rule. (A truncated-JSON wrapper would *not* do:
            # the scanner salvages a call from it, so it never reaches this branch.)
            payload = "Prose before.\n<tool_call>\nthis is not json\n</tool_call>"
        elif content_kind == "":
            payload = ""
        else:
            payload = "Let me read the brief now."
        msg = {"content": payload, "reasoning_content": None}
        choice = {"finish_reason": scenario["finish_reason"]}
        captured: List[str] = []
        original_warning = logger.warning
        logger.warning = lambda message, *a, **k: captured.append(str(message))
        try:
            process_message_tools_and_thinking(msg, choice, None)
        finally:
            logger.warning = original_warning
        dumped = any("Raw unparsed completion" in message for message in captured)
        # The dump follows the flag, so the expectation has to as well: this file is
        # run both from the unit (where `~/.config/tabby-proxy.env` arms the flag)
        # and by hand from a shell (where it is off), the same way the raw-dump leg
        # above compares against `_RAW_LOG` rather than against a literal.
        ok = dumped == (want_dump and _RAW_LOG)
        print(f"{'ok  ' if ok else 'FAIL'} rawdump {name}: dumped={dumped} (flag={_RAW_LOG})")
        if not ok:
            failures += 1
            print(f"     expected dumped={want_dump and _RAW_LOG}, warnings={captured!r}")

    # The dump covers both fields the parser examined, not just the first non-empty
    # one. Observed live 2026-09-26 (session aba2e4e2): the text field was the bare
    # announcement *"I'll start by reading the stage brief."* and the dump stopped
    # there, so whether the failed call sat in `reasoning_content` went unrecorded.
    two_field_msg = {
        "content": "Let me read the brief now.",
        "reasoning_content": "Prose.\n<tool_call>\nthis is not json\n</tool_call>",
    }
    captured = []
    original_warning = logger.warning
    logger.warning = lambda message, *a, **k: captured.append(str(message))
    try:
        process_message_tools_and_thinking(dict(two_field_msg), {"finish_reason": "tool_calls"}, None)
    finally:
        logger.warning = original_warning
    got_fields = [m.split(":", 1)[0] for m in captured if "Raw unparsed" in m]
    both_fields = (not _RAW_LOG) or ("Raw unparsed reasoning" in got_fields and "Raw unparsed completion" in got_fields)
    print(f"{'ok  ' if both_fields else 'FAIL'} rawdump covers the reasoning field too: {got_fields}")
    if not both_fields:
        failures += 1

    # A `tool_calls` turn the repetition strip emptied is the other half of that
    # blind spot: the loop is deleted before the dump sees it, so the emitted text
    # has to be dumped as well or the turn leaves no evidence at all.
    loop_only = {"content": "<tool_call>" * 5, "reasoning_content": None}
    captured = []
    original_warning = logger.warning
    logger.warning = lambda message, *a, **k: captured.append(str(message))
    try:
        process_message_tools_and_thinking(dict(loop_only), {"finish_reason": "tool_calls"}, None)
    finally:
        logger.warning = original_warning
    prestrip = [m for m in captured if "before the repetition strip" in m]
    strip_ok = (not _RAW_LOG) or bool(prestrip)
    print(f"{'ok  ' if strip_ok else 'FAIL'} rawdump shows a stripped-away completion: {len(prestrip)} line(s)")
    if not strip_ok:
        failures += 1

    # A completion past roughly 180k tokens loses ASCII: `"` becomes `“`/`”` and `|`
    # becomes the fullwidth `｜`. Neither reads as a quote or a bar to Python, so a
    # call written in those glyphs is not JSON at all and every call in the turn was
    # lost. Captured verbatim from session e8ee13a8 on 2026-09-26 (195k input
    # tokens), together with the mangled closing tags the same glyph loss produced.
    typographic = (
        "Reading the view.\n"
        "<tool_call>\n"
        "{“name”: “glob”, “arguments”: {“pattern”:“src/analysis_qt6/*.py”}}\n"
        f"</{fw}>"
    )
    truncated_then_valid = (
        "<tool_call>\n"
        '{"name": "write_file", "arguments": {"content": "x"\n'
        f"</{fw}DSML{fw}>\n<{fw}DSML{fw}>\n"
        '{"name": "glob", "arguments": {"pattern": "y"}}\n'
        f"</{fw}DSML{fw}>\n"
    )
    glyph_cases = [
        ("typographic quotes still parse", typographic, ["glob"]),
        ("stray closer is not a call", f"Prose.\n</{fw}DSML{fw}_call>", None),
        ("stray opener-fragment is not a call", f"Prose.\n<\\{fw}DSML{fw}_placeholder>", None),
        ("a truncated call does not hide the next", truncated_then_valid, ["write_file", "glob"]),
        ("ordinary prose markup is not a call", "<code>\nls\n</code>\n<description>list</description>", None),
    ]
    for label, raw, want in glyph_cases:
        calls, _cleaned = extract_tool_calls(raw)
        got = [c["function"]["name"] for c in calls] if calls else None
        ok = got == want
        print(f"{'ok  ' if ok else 'FAIL'} glyph {label}: {got}")
        if not ok:
            failures += 1
            print(f"     expected: {want!r}")

    # A native `delta.tool_calls` stream is the case the proxy used to drop:
    # TabbyAPI parses the checkpoint's DSML server-side and delivers the call in
    # the delta, so no marker ever reaches a text field, the scanner finds
    # nothing, and the `tool_calls` finish reason is downgraded to `stop` -- the
    # client then gets neither a call nor a finished turn and the session wedges
    # (observed 2026-09-26, #431, session aba2e4e2). Two legs: the merger must
    # preserve the upstream arguments byte-for-byte, and the accumulator must
    # actually read the delta.
    native_args = '{"file_path": "/tmp/stage123.txt"}'
    native_slots = {
        0: {"id": "call_abc123", "type": "function", "name": "read_file", "arguments": native_args},
        1: {"id": None, "type": "function", "name": "glob", "arguments": ""},
    }
    native_msg = {"content": None, "reasoning_content": None}
    merged = _merge_native_tool_calls(native_msg, native_slots)
    merged_pairs = [(c["function"]["name"], c["function"]["arguments"]) for c in merged]
    merge_ok = (
        merged_pairs == [("read_file", native_args), ("glob", "{}")]
        and merged[0]["id"] == "call_abc123"
        and merged[1]["id"].startswith("call_")
        and native_msg["tool_calls"] is merged
    )
    print(f"{'ok  ' if merge_ok else 'FAIL'} native merge keeps upstream arguments: {merged_pairs}")
    if not merge_ok:
        failures += 1
        print(f"     expected: [('read_file', {native_args!r}), ('glob', '{{}}')] with ids call_abc123/'call_*'")

    class _FakeStream:
        def __init__(self, lines):
            self._lines = lines

        async def aiter_lines(self):
            for line in self._lines:
                yield line

    class _NoopCtx:
        async def __aexit__(self, *exc):
            return False

    class _NoopClient:
        async def aclose(self):
            return None

    def _evt(delta, finish=None, eos=None):
        choice = {"index": 0, "delta": delta}
        if finish:
            choice["finish_reason"] = finish
        if eos:
            choice["eos_reason"] = eos
        return "data: " + json.dumps({"choices": [choice]})

    # The arguments arrive split across two deltas, as a real stream sends them.
    delta_events = [
        _evt({"role": "assistant"}),
        _evt({"reasoning_content": "Let me read it."}),
        _evt({"tool_calls": [{"index": 0, "id": "call_abc123", "type": "function",
                              "function": {"name": "read_file", "arguments": '{"file_path": "/tmp/stage'}}]}),
        _evt({"tool_calls": [{"index": 0, "function": {"arguments": '123.txt"}'}}]}),
        _evt({}, finish="tool_calls", eos="eos"),
        "data: [DONE]",
    ]
    split_args = '{"file_path": "/tmp/stage123.txt"}'  # the two fragments joined

    async def _drive():
        collected = []
        async for chunk in stream_tools_response(
            _NoopCtx(), _FakeStream(delta_events), _NoopClient(), "test-model", None
        ):
            collected.append(chunk)
        return collected

    seen_calls = []
    seen_finish = None
    for chunk in asyncio.run(_drive()):
        if not chunk.startswith("data: "):
            continue
        body = chunk[len("data: "):].strip()
        if not body or body == "[DONE]":
            continue
        event = json.loads(body)
        for choice in event.get("choices", []):
            for call in (choice.get("delta") or {}).get("tool_calls") or []:
                seen_calls.append((call.get("index"), call["function"]["name"], call["function"]["arguments"]))
            if choice.get("finish_reason"):
                seen_finish = choice["finish_reason"]
    delta_ok = seen_calls == [(0, "read_file", split_args)] and seen_finish == "tool_calls"
    print(f"{'ok  ' if delta_ok else 'FAIL'} native delta.tool_calls is emitted: calls={seen_calls} finish={seen_finish!r}")
    if not delta_ok:
        failures += 1
        print(f"     expected: [(0, 'read_file', {split_args!r})] finish='tool_calls'")

    # The `<arguments>` container element: the checkpoint writes the arguments
    # object inside a tag of that name instead of one `<parameter name=...>` per
    # argument. Captured verbatim from the live probe e5e8c7f9 (2026-09-26, its
    # first turn): the tag was not a parameter, so its JSON was never collected
    # and the call went out with empty arguments -- which the client rejects as
    # `invalid_tool_params` and makes the model retry the identical call. The
    # known-good `<parameter>` dialect must keep working, and prose that merely
    # mentions the tag must not invent a call.
    arg_container = (
        "<tool_call>\n"
        f'<{fw}DSML{fw} name="read_file">\n'
        '<arguments>{"file_path": "/home/logan/software/analysis-qt6/stage141.txt"}</arguments>\n'
        f"</{fw}DSML{fw}>\n"
        "</tool_call>"
    )
    arg_parameter = (
        "<tool_call>\n"
        f'<{fw}DSML{fw} name="read_file">\n'
        '<parameter name="file_path">/tmp/stage123.txt</parameter>\n'
        f"</{fw}DSML{fw}>\n"
        "</tool_call>"
    )
    arg_cases = [
        ("arguments element is read",
         arg_container,
         [("read_file", {"file_path": "/home/logan/software/analysis-qt6/stage141.txt"})]),
        ("parameter element still reads",
         arg_parameter,
         [("read_file", {"file_path": "/tmp/stage123.txt"})]),
        ("prose mentioning the tag is not a call",
         "The <arguments> tag holds the call's values.",
         None),
    ]
    for name, raw, expected in arg_cases:
        calls, _cleaned = extract_tool_calls(raw)
        got = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in (calls or [])]
        ok = got == (expected or [])
        print(f"{'ok  ' if ok else 'FAIL'} arguments container {name}: {got}")
        if not ok:
            failures += 1
            print(f"     expected: {expected!r}")

    # Two more shapes from the same live probe (2026-09-26), both of which produced
    # a call the client refused. `call_tool` carries the name *and* the whole
    # argument object as attributes, so the object was never read; the `calls`
    # wrapper (a DSML element named for the plural) yielded a call literally called
    # `calls`, which no registry has. The `parameter`-named-`args` leg is the
    # guard against reading the new attribute key too eagerly: there `args` is a
    # genuine argument name and its value must stay filed under it.
    round3_cases = [
        ("call_tool name+args attributes are read",
         "<tool_call>\n"
         f'<{fw}DSML{fw}call_tool name="read_file" args={{"file_path": "/tmp/a.py", "offset": 3720, "limit": 130}}>\n'
         f"</{fw}DSML{fw}call_tool>\n</tool_call>",
         [("read_file", {"file_path": "/tmp/a.py", "offset": 3720, "limit": 130})]),
        ("DSML calls wrapper does not become a call",
         "<tool_call>\n"
         f"<{fw}DSML{fw}calls>\n"
         '<tool_call>\n{"name": "read_file", "arguments": {"file_path": "/tmp/b.py"}}\n'
         f"</{fw}DSML{fw}calls>\n"
         '<tool_call>\n{"name": "glob", "arguments": {"pattern": "src/**"}}\n</tool_call>',
         [("read_file", {"file_path": "/tmp/b.py"}), ("glob", {"pattern": "src/**"})]),
        ("a parameter named args is not swallowed",
         "<tool_call>\n"
         f'<{fw}DSML{fw} name="read_file">\n'
         '<parameter name="args">{"a": 1}</parameter>\n'
         f"</{fw}DSML{fw}>\n</tool_call>",
         [("read_file", {"args": {"a": 1}})]),
    ]
    for name, raw, expected in round3_cases:
        calls, _cleaned = extract_tool_calls(raw)
        got = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in (calls or [])]
        ok = got == expected
        print(f"{'ok  ' if ok else 'FAIL'} round3 {name}: {got}")
        if not ok:
            failures += 1
            print(f"     expected: {expected!r}")

    # The "no call was parsed" verdict is a claim about the two text fields, so it
    # must not be made on a turn whose call arrived over the native channel -- the
    # processor cannot see that channel, and emitting the line for a call that was
    # in fact delivered is a false alarm in the log (observed live 2026-09-26: the
    # three native calls of session e5e8c7f9 each logged it, yet all reached the
    # client). The finish reason must also be left alone in that case, since the
    # caller settles it once it has merged the native call.
    native_msg = {"content": "Let me read it.", "reasoning_content": None}
    native_choice = {"finish_reason": "tool_calls"}
    captured = []
    original_warning = logger.warning
    logger.warning = lambda message, *a, **k: captured.append(str(message))
    try:
        process_message_tools_and_thinking(native_msg, native_choice, None, native=True)
    finally:
        logger.warning = original_warning
    quiet = not any("no call was parsed" in m for m in captured)
    held_reason = native_choice["finish_reason"] == "tool_calls"
    print(f"{'ok  ' if quiet and held_reason else 'FAIL'} native flag silences the verdict: quiet={quiet} finish={native_choice['finish_reason']!r}")
    if not (quiet and held_reason):
        failures += 1
        print(f"     expected quiet=True finish='tool_calls'; warnings={captured!r}")

    total = (
        len(cases)
        + len(marker_cases)
        + len(remainder_cases)
        + len(repair_cases)
        + len(control_cases)
        + len(name_cases)
        + len(width_cases)
        + 1
        + len(hoist_cases)
        + len(repetition_cases)
        + len(downgrade_cases)
        + len(glyph_cases)
        + 2
        + 2
        + len(arg_cases)
        + len(round3_cases)
        + 1
    )
    print(f"{total - failures}/{total} self-test cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    # TABBY_PROXY_HOST exists because a containerised client (OpenWebUI) cannot
    # reach the host's loopback: rootless docker routes it through slirp4netns,
    # so the only address that answers is the host's own routable IP. Bind that
    # one deliberately rather than widening the default. TABBY_PROXY_PORT is here
    # for the same reason: to let a deployment move the port without a patch.
    # Upstream stays TABBY_API_URL (default 127.0.0.1:5000), unaffected by either.
    uvicorn.run(
        app,
        host=os.getenv("TABBY_PROXY_HOST", "127.0.0.1"),
        port=int(os.getenv("TABBY_PROXY_PORT", "8081")),
    )
