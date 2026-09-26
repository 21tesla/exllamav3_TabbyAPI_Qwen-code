#!/usr/bin/env python3
"""Watch the Qwen Code <-> TabbyAPI connection only.

Two streams, both about the local TabbyAPI path:

  1. Transcripts whose model is the TabbyAPI model, across every project. Records
     from sessions on a different backend (Ollama cloud) are ignored entirely.
  2. The proxy's own journal, filtered to non-2xx responses and error lines.

Failures reported: api_error telemetry, tool results that did not succeed (a
user-cancelled call is a decision, not a defect, so it is skipped), calls to tool
names that do not exist, and proxy-side 4xx/5xx.

An undeclared tool name and the failure of that same call are two transcript
records for one call id, so they are held briefly and reported as one line rather
than two -- each printed line is a separate monitor event.

Usage: tabby_watch.py [projects-glob]

Environment:
  TABBY_WATCH_MODEL    override the model name that marks a transcript as TabbyAPI
  TABBY_WATCH_PROJECTS override the transcript search path
  TABBY_PROXY_UNIT     override the proxy unit; by default the instance name is
                       derived from $HOME the way install.sh derives it, so this
                       script carries no machine-specific name.
"""
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time


def _default_unit() -> str:
    """The proxy unit for this home, named the way `systemd-escape` names it."""
    try:
        escaped = subprocess.run(
            ["systemd-escape", "--path", os.path.expanduser("~")],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        # No systemd-escape (or it failed): fall back to the path with separators
        # flattened, which is how systemd names the instance for a plain home.
        escaped = os.path.expanduser("~").replace("/", "-").lstrip("-")
    return f"tabby-proxy@{escaped}.service"


PROJECTS_GLOB = os.environ.get(
    "TABBY_WATCH_PROJECTS", os.path.expanduser("~/.qwen/projects/*/chats")
)
TABBY_MODEL = os.environ.get("TABBY_WATCH_MODEL", "DeepSeek-V4-Flash-0731-exl3-2.32bpw")
PROXY_UNIT = os.environ.get("TABBY_PROXY_UNIT") or _default_unit()
HEARTBEAT_S = 540
POLL_S = 2
MODEL_SCAN_LINES = 120
# Hold a finding this long so the other record for the same call arrives and the
# pair merges into one report. Longer than one poll, far shorter than a heartbeat.
MERGE_WINDOW_S = 6

KNOWN_TOOLS = {
    "read_file", "write_file", "edit", "glob", "grep_search", "run_shell_command",
    "agent", "list_agents", "tool_search", "tool_call", "get_goal", "update_goal",
    "notebook_edit", "ask_user_question", "structured_output", "web_fetch",
    "send_message", "task_stop", "monitor", "loop_wakeup", "cron_list",
}

STATUS_RE = re.compile(r"HTTP/1\.1[^0-9]*(\d{3})")
LOG_KEYWORDS = ("retrying", "traceback", "exception", "upstream", "malformed")


def classify(rec):
    """Return (tag, detail, call_id) for a record worth reporting, else None.

    The call id lets the caller merge the two records that describe one failing
    call -- the assistant's undeclared-name record and the tool result for it --
    into a single monitor event.
    """
    kind = rec.get("type")

    if kind == "system":
        event = (rec.get("systemPayload") or {}).get("uiEvent") or {}
        if event.get("event.name") == "api_error":
            return "API_ERROR", (
                f"{event.get('error_type')} :: "
                f"{str(event.get('error_message'))[:160]} "
                f"(dur={event.get('duration_ms')}ms id={event.get('response_id')})"
            ), None
        return None

    if kind == "tool_result":
        result = rec.get("toolCallResult") or {}
        status = result.get("status")
        if status == "success" and not result.get("errorType"):
            return None
        # A cancelled call is a decision: the user declined, or a rule blocked
        # it. Not a defect, and reporting it would be noise.
        if status == "cancelled":
            return None
        name, message, call_id = "?", "", None
        for part in rec.get("message", {}).get("parts", []):
            response = part.get("functionResponse")
            if response:
                name = response.get("name")
                body = response.get("response") or {}
                message = str(body.get("error") or body.get("output") or "")
                # `functionResponse.id` is the pairing key here; the separate
                # `toolCallResult.callId` field is declared but never populated.
                call_id = response.get("id")
        # Collapse newlines: a shell error carries its whole multi-line output,
        # and one line here means one monitor event instead of a dozen.
        message = " ".join(message.split())
        return "TOOL_ERROR", f"{name} [{result.get('errorType')}] {message[:220]}", call_id

    if kind == "assistant":
        named = [
            (part["functionCall"].get("name"), part["functionCall"].get("id"))
            for part in rec.get("message", {}).get("parts", [])
            if part.get("functionCall")
        ]
        unknown = [(n, i) for n, i in named if n not in KNOWN_TOOLS]
        if unknown:
            names = [n for n, _ in unknown]
            return "ODD_CALL", f"non-standard tool name(s): {names}", unknown[0][1]
        return None

    return None


def detect_model(path):
    """The model a transcript is pinned to, read once at startup."""
    try:
        with open(path, errors="replace") as handle:
            for _ in range(MODEL_SCAN_LINES):
                line = handle.readline()
                if not line:
                    break
                if '"model"' not in line:
                    continue
                try:
                    model = json.loads(line).get("model")
                except ValueError:
                    continue
                if model:
                    return model
    except OSError:
        pass
    return None


def transcript_dirs(pattern):
    return sorted(glob.glob(pattern))


def follow_proxy_log():
    """Stream the proxy journal, printing only problem lines."""
    try:
        proc = subprocess.Popen(
            ["journalctl", "--user", "-u", PROXY_UNIT, "-f", "-n", "0", "--no-pager", "-o", "cat"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
    except Exception as exc:
        print(f"-- proxy log unavailable: {exc}", flush=True)
        return
    print("-- proxy log attached", flush=True)
    for line in proc.stdout or []:
        text = line.rstrip()
        if not text:
            continue
        match = STATUS_RE.search(text)
        bad_status = bool(match) and int(match.group(1)) >= 400
        low = text.lower()
        if bad_status or any(word in low for word in LOG_KEYWORDS):
            print(f"-- proxy: {text}", flush=True)


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else PROJECTS_GLOB
    dirs = transcript_dirs(pattern)
    tracked, models = {}, {}
    for directory in dirs:
        for name in os.listdir(directory):
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(directory, name)
            try:
                tracked[path] = os.path.getsize(path)
            except OSError:
                continue
            models[path] = detect_model(path)

    on_tabby = sum(1 for m in models.values() if m == TABBY_MODEL)
    print(
        f"monitor up: {len(dirs)} project(s), {len(tracked)} transcript(s) at EOF, "
        f"{on_tabby} on the TabbyAPI model ({TABBY_MODEL}); "
        f"other backends ignored",
        flush=True,
    )
    threading.Thread(target=follow_proxy_log, daemon=True).start()

    last_report = time.time()
    counts = {}
    pending = False
    # call_id -> (tag, stamp, detail, held_since); a finding that pairs with a
    # later record for the same call is held here until its partner arrives.
    pending_calls = {}

    while True:
        for directory in dirs:
            for name in sorted(os.listdir(directory)):
                if not name.endswith(".jsonl"):
                    continue
                path = os.path.join(directory, name)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                start = tracked.get(path)
                if start is None:
                    tracked[path] = 0
                    models[path] = detect_model(path)
                    start = 0
                if size <= start:
                    continue
                try:
                    with open(path, errors="replace") as handle:
                        handle.seek(start)
                        chunk = handle.read()
                except OSError:
                    continue
                # Consume whole lines only; a partial trailing line is re-read
                # next poll once the writer has finished it.
                if not chunk.endswith("\n"):
                    cut = chunk.rfind("\n")
                    if cut == -1:
                        continue
                    tracked[path] = start + cut + 1
                    chunk = chunk[: cut + 1]
                else:
                    tracked[path] = size
                pending = True

                for line in chunk.splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if rec.get("model"):
                        models[path] = rec["model"]
                    if models.get(path) != TABBY_MODEL:
                        continue
                    found = classify(rec)
                    if found:
                        tag, detail, call_id = found
                        stamp = rec.get("timestamp", "")[11:19] or time.strftime("%H:%M:%S")
                        # One report is one event; a detail with newlines would fan
                        # a single failure out into many event notifications.
                        detail = " ".join(str(detail).split())
                        if call_id:
                            # Two records can describe one call: the assistant's
                            # call and the result for it. Emit the first, merge the
                            # second. Emitting both would cost two events per call.
                            held = pending_calls.get(call_id)
                            if held:
                                del pending_calls[call_id]
                                counts[held[0]] = counts.get(held[0], 0) + 1
                                counts[tag] = counts.get(tag, 0) + 1
                                print(
                                    f"{held[1]} [{held[0]}] {os.path.basename(path)[:8]} "
                                    f"{held[2]} | {tag}: {detail}",
                                    flush=True,
                                )
                                last_report = time.time()
                            else:
                                pending_calls[call_id] = (tag, stamp, detail, time.time())
                        else:
                            counts[tag] = counts.get(tag, 0) + 1
                            print(
                                f"{stamp} [{tag}] {os.path.basename(path)[:8]} {detail}",
                                flush=True,
                            )
                            last_report = time.time()

        # Emit a held call before the merge window closes; otherwise its merge
        # partner never arrived and the finding would be lost entirely.
        now = time.time()
        for call_id in [c for c, h in pending_calls.items() if now - h[3] >= MERGE_WINDOW_S]:
            tag, stamp, detail, _ = pending_calls.pop(call_id)
            counts[tag] = counts.get(tag, 0) + 1
            print(
                f"{stamp} [{tag}] unpaired {call_id} {detail}",
                flush=True,
            )
            last_report = time.time()

        quiet = time.time() - last_report
        if quiet >= HEARTBEAT_S:
            # Fires even with no traffic: the harness kills a monitor that stops
            # printing, and the pulse also proves the tail is still alive.
            tally = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
            seen = "traffic" if pending else "no traffic"
            print(f"-- heartbeat: quiet {int(quiet)}s, {seen}, problems so far: {tally}", flush=True)
            last_report = time.time()
            pending = False

        time.sleep(POLL_S)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
