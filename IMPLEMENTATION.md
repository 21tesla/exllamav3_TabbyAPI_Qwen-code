# Implementation notes

The detail behind [README.md](README.md). Read that first for orientation; this file explains
*why* each piece is the way it is, and what goes wrong if it is changed.

`install.sh` is the executable form of this document. Wherever the two disagree, the script is
what runs.

---

## 1. The stack

```
qwen (Qwen Code CLI, host) ─┐
                            ├─> tabby_proxy :8081 ─> TabbyAPI :5000 ─> model
open-webui (container) ─────┘
```

Both clients go through `tabby_proxy`. TabbyAPI is bound to **loopback only** (`-p 127.0.0.1:5000`),
so it is not on the network; the proxy is where a containerised client has to go instead.

Four parts, each with an owner:

| Part | What it is | Where it lives |
|---|---|---|
| TabbyAPI | inference server, a Docker container | image `ghcr.io/theroyallab/tabbyapi:cu13` |
| ExLlamaV3 fork (anemone) | the build that serves the pack | `~/software/exllamav3-anemone`, and its `venv` |
| `tabby_proxy.py` | the schema guard, and the reason this repo exists | **this repository** |
| `tabby-proxy@.service` | runs the proxy as a user service | this repository, installed to `~/.config/systemd/user` |

The unit's `ExecStart` names two directories on purpose: the **script** from this repo, the
**interpreter** from the fork's venv (`uvicorn` and `httpx` are installed only there). An earlier
revision ran the fork's own copy of the proxy, which meant every fix had to be copied across by hand
into a read-only checkout and the two copies drifted silently. Keeping the script here is what makes
`systemctl --user restart` sufficient after an edit.

---

## 2. What the proxy fixes, and why it has to exist

### 2.1 `422` on requests that register tools

Qwen Code registers built-in tools (`get_goal`, `list_agents`, ...). For a tool that takes no
arguments it submits a schema with no `"parameters"` key. TabbyAPI validates tool schemas with
strict Pydantic models and rejects the whole request with `422 Unprocessable Content`.

`patch_tools_schema` walks the `tools` array and fills in a default empty parameters object before
the request is forwarded.

### 2.2 `finish_reason: "tool_calls"` with `tool_calls: null`

The more damaging one, because it fails *silently*. With `tools` in the request this build answers:

```json
{"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": null, "content": "..."}}]}
```

The call itself is not lost — it is **DeepSeek DSML text** inside `content` (or
`reasoning_content`). A client that trusts `finish_reason` sees `tool_calls: null`, renders an empty
assistant turn, and the session stops with no error at all. Nothing is logged anywhere.

The same response sent directly to TabbyAPI (bypassing the proxy) confirms it: `finish_reason` was
`"stop"` rather than `"tool_calls"`, with no `tool_calls` at all and the call sitting in `content`
as prose in a fenced JSON block that even used the wrong argument name (`"path"` instead of
`"file_path"`). The proxy's version of the same request returned a structured call with the right
argument name.

The proxy:

1. injects the tool catalogue and the tool-call format into the system prompt,
2. parses the call back out of the text **as it streams**,
3. re-emits it as a valid OpenAI SSE stream carrying real `tool_calls` deltas.

### 2.3 The dialects

The DSML this checkpoint emits is not stable between turns. At least four shapes appear, so the
parser walks the tags rather than matching one regex:

```
<tool_call>
<|DSML|tool name="read_file">            name attribute on the open tag
<parameter name="file_path">p</parameter>
</|DSML|tool>
</tool_call>

<tool_call>
<|DSML| name="read_file">                no tag name, only the attribute
<parameter name="file_path">p</parameter>
</|DSML|>                                bare closing tags
</|DSML|>
</tool_call>

<tool_call>
<\uff5cDSML\uff5c name="read_file">      the bars arrive as escaped text
...

<tool_call>
<|DSML|read_file>                        function name used as the tag name
...
```

If a dialect still escapes the parser, the proxy logs `Tool-call syntax present but unparsed: ...`
to the journal instead of dropping the call silently. That log line is the thing to search for when
a tool call goes missing:

```bash
journalctl --user -u "tabby-proxy@$home.service" -f
```

### 2.4 Streaming, and the hold-back window

Requests carrying `tools` used to be forced non-streaming, so that the whole completion could be
parsed before answering — the client sat silent for the entire generation. The tool path is now
incremental: upstream deltas are relayed as they arrive, with the last 32 characters of `content`
and of `reasoning_content` held back.

`extract_tool_calls` discards everything from the first tool-syntax marker onwards and keeps
everything before it. Holding back that window guarantees a marker is never emitted half-formed; a
marker spanning two deltas is caught by the tail of the window. When no marker appears, the held
tail is flushed at the end, and the final `finish_reason` and `tool_calls` deltas come from the same
parser the non-streaming path uses.

The hold-back exists because markers are recognised only once their keyword has arrived: `<tool_c`
reads as ordinary prose, so without it a split `<tool_call>` could leak into the visible answer.
`_HOLDBACK` must stay greater than `_MARKER_PREFIX_LIMIT`, which the self-test asserts.

`reasoning_content` streams separately from `content`, and that is where most of the visible latency
lives, so the first reasoning delta reaches the client within milliseconds of generation starting
rather than after it completes.

### 2.5 Retries while a model loads

TabbyAPI answers `503` for a few seconds while loading or reloading a model, and occasionally
`502`/`529` under load. A single such answer used to end the turn. Every upstream call now retries
up to 5 times with exponential backoff (2 s, 4 s, 8 s, 16 s) on
`429`/`500`/`502`/`503`/`504`/`529` and on connection failures, so a model reload no longer kills a
request. A non-retryable status is returned to the client unchanged.

### 2.6 Token usage and `stream_options`

Upstream TabbyAPI reports token counts only when the request sets
`stream_options.include_usage`, and it reports them in a final usage-only chunk whose `choices`
list is empty. The proxy used to drop `stream_options` whenever `tools` were present, on the
reasoning that the synthesised stream made upstream accounting moot. It does not: usage is
upstream's accounting for the turn, not part of the SSE framing, so discarding it left every
client showing `0 / 0` tokens for tool-bearing requests — `/stats` in Qwen Code, and the token
counters in OpenWebUI.

The tool path now keeps `stream_options` and forces `include_usage` on when the client has not
asked for it, so the counts do not depend on what the client happened to send. The usage object
from upstream is captured and re-emitted unchanged as a terminal chunk before `[DONE]`, in the
same shape OpenAI and ollama use:

```
data: {"id": "chatcmpl-…", "object": "chat.completion.chunk", …,
       "choices": [], "usage": {"prompt_tokens": 217, "completion_tokens": 58, "total_tokens": 275}}
data: [DONE]
```

TabbyAPI enriches that object with `prompt_time`, `prompt_tokens_per_sec` and acceptance counts;
it is forwarded verbatim rather than normalised, so those survive for clients that want them.
Requests without `tools` take the pass-through path and have always received usage unchanged.

### 2.7 A JSON tool call missing its closing brace

Roughly once per long session the checkpoint ends a JSON tool call one character early: the outer
argument object's `}` is absent, and the `</tool_call>` wrapper still follows it. The client sees
a malformed call and aborts the turn with `InvalidStreamError: Model response contained a
malformed tool call.`, so the whole turn is lost for the sake of one byte.

Captured live (2026-09-25) as an `ask_user_question` call: 7 `{` against 6 `}`, balanced brackets,
every brace outside a string, and `json.loads(value + "}")` parsing cleanly to a valid call.

`repair_truncated_json()` handles this without loosening the parser generally. When `raw_decode`
fails at a starting bracket, the parser asks it for a repair: string contents are removed so a `{`
inside a description cannot skew the count, the brackets left open are collected as a stack, and
those closers are appended in nesting order and re-parsed. Because the wrapper markup is not part
of the value, a fragment whose last character is not a closer is first cut at its last structural
character — never earlier, since truncating at an interior closer would silently drop part of the
value. The repair only succeeds when the result parses, so prose containing a brace still yields
no call, and the extracted call then goes through the same `normalize_tool_call_dict` validation
as any other. At most `len(stack)` closers are tried, and each is tried once.

### 2.8 Self-test

The parser ships with fixtures for every dialect above, plus fixtures for the streaming hold-back
window and for the brace repair. Run it after any change to the parser:

```bash
~/software/exllamav3-anemone/venv/bin/python tabby_proxy.py --selftest
```

---

## 3. Getting the pieces

### 3.1 The model pack

```bash
hf download anoane/DeepSeek-V4-Flash-0731-exl3-2.32bpw \
  --local-dir ~/models/DeepSeek-V4-Flash-0731-exl3-2.32bpw
```

85 GB, 12 shards. The pack ships the DeepSeek chat template and the sampler preset under `serve/`.
TabbyAPI reads the template from the model directory root, so `install.sh` also copies
`serve/chat_template.jinja` there and warns if the two copies ever differ.

### 3.2 ExLlamaV3 (the anemone fork)

```bash
git clone https://github.com/anoane/exllamav3-anemone
cd exllamav3-anemone
python -m venv venv
./venv/bin/pip install --upgrade pip setuptools wheel
./venv/bin/pip install -r requirements.txt
TORCH_CUDA_ARCH_LIST="12.0" ./venv/bin/pip install -e . --no-build-isolation
```

`12.0`, not `12.0a`. A single-arch `12.0a` build breaks the DSA decode graph path, so the main
extension targets plain `12.0`; the optional FP4 prefill kernel targets `sm_120a` separately from
`exllamav3/anemone_fp4/`.

The **published TabbyAPI image bundles upstream ExLlamaV3**, and that is sufficient for this pack —
its `quantization_config.json` carries no per-expert K table, so the fork's mixed-K loader is not
needed to *serve* it. The fork is needed for host-side conversion, evaluation and examples.

### 3.3 The TabbyAPI image

```bash
docker pull ghcr.io/theroyallab/tabbyapi:cu13
```

Docker runs **rootless** here (`unix:///run/user/1000/docker.sock`), so neither the pull nor the
service needs root.

**If the pull answers `denied: denied`, the image is not private.** A stored ghcr credential is
being presented that cannot read it: `~/.docker/config.json` may hold an expired or scope-limited
token for `ghcr.io`, and Docker will not fall back to anonymous access on its own. Confirm that
anonymous access works, then pull without the credential:

```bash
curl -s "https://ghcr.io/token?scope=repository:theroyallab/tabbyapi:pull&service=ghcr.io"   # a token, not a denial
mkdir -p /tmp/docker-nocreds && printf '{}' > /tmp/docker-nocreds/config.json
DOCKER_CONFIG=/tmp/docker-nocreds docker pull ghcr.io/theroyallab/tabbyapi:cu13
```

An empty `DOCKER_CONFIG` also drops the rootless *context*, so add
`DOCKER_HOST=unix:///run/user/1000/docker.sock` if docker then reports
`permission denied ... /var/run/docker.sock`. `install.sh` does all of this automatically.

---

## 4. Starting TabbyAPI

```bash
docker run --gpus all --shm-size=8g --name tabbyapi \
  -d \
  -p 127.0.0.1:5000:5000 \
  --entrypoint python3 \
  -v ~/models:/app/models \
  -v ~/software/tabbyapi-config/config.yml:/app/config.yml:ro \
  -v ~/software/tabbyapi-config/api_tokens.yml:/app/api_tokens.yml \
  -v ~/software/tabbyapi-config/sampler_overrides:/app/sampler_overrides:ro \
  --restart unless-stopped \
  ghcr.io/theroyallab/tabbyapi:cu13 \
  main.py --host 0.0.0.0
```

Every part of that is load-bearing, and each one fails **silently** when it is wrong.

### 4.1 `config.yml` and `api_tokens.yml` are mounted as files, not directories

TabbyAPI reads both by bare relative path — `pathlib.Path("config.yml")`,
`AUTH_FILE = "api_tokens.yml"` — against its working directory, and the image's own code lives at
`/app` (`WORKDIR /app`, then `COPY . .`). Mounting a **directory** at `/app` would hide the config
*and* shadow the application: the container would start on defaults with no error anywhere.

`/app/sampler_overrides` is resolved the same relative way, which is why the preset directory is
mounted there rather than left next to the config.

### 4.2 `api_tokens.yml` must be mounted, or the key changes without asking

With no auth file at `/app/api_tokens.yml`, TabbyAPI generates a fresh `token_hex(16)` at startup,
writes it inside the container, and prints it to the log. Every client request then fails `401`
against the key in `~/.qwen/settings.json`. It is mounted **writable** so that generation still
works if the file is ever missing.

This was the first failure mode hit on this machine: the log read
`Your API key is: 0da2b04f372b9be171332ab8528eab96`, not the key in `settings.json`.

### 4.3 `--host 0.0.0.0`

The default is `127.0.0.1`, which inside a container is the container's own loopback — unreachable
through `-p`. Command-line arguments override `config.yml`, so this and `network.host` must agree or
the port mapping silently breaks.

### 4.4 No `model_name` in `config.yml`

Setting `model.model_name` makes TabbyAPI load the model at startup, where it then holds ~90 of the
98 GB of VRAM around the clock, including between sessions. Left out, the server starts empty and
`inline_model_loading: true` loads the model on the first request that names it.

### 4.5 `--shm-size=8g`

Not optional. ExLlamaV3 keeps tensor-parallel and CPU-MoE handoff buffers in `/dev/shm`, where
Docker's 64 MiB default fails.

### 4.6 No `--ulimit memlock=-1`

Upstream suggests it; it cannot be used here. Docker is **rootless**, so the daemon is unprivileged
and cannot raise `RLIMIT_MEMLOCK`. Passing it aborts the container at OCI-create time:

```
error setting rlimit type 8: operation not permitted
```

### 4.7 `--restart unless-stopped`

This is what brings the container back after a reboot. Without it the container does not return and
nothing announces that. Note the limit of the mechanism: **Docker restart policies only revive
containers that still exist.** Delete the container and the next boot will not recreate it — that
case needs `./install.sh tabby`.

---

## 5. Configuring the model to load on demand

`install.sh` writes `~/software/tabbyapi-config/config.yml` (and never overwrites an existing one,
because it is the file most likely to have been tuned):

```yaml
network:
  host: 0.0.0.0        # must match --host on the docker run line
  port: 5000

model:
  model_dir: /app/models
  # No model_name, deliberately. See 4.4.
  inline_model_loading: true
  use_dummy_models: true

sampling:
  override_preset: deepseek_v4
```

* **The first request pays, the rest do not.** A cold load is 48 modules / ~90 GB and takes 10–20 s,
  during which the caller waits. Afterwards the model stays resident.
* **Inline loading is strict.** `endpoints/OAI/utils/common_.py` returns `404` for a model not in
  `model_dir` and `503` for one that fails to load, rather than answering from whatever is resident.
  `use_dummy_models: true` exempts the names some clients always send (`gpt-3.5-turbo`), which then
  run on the loaded model.
* **Switching models needs an admin key.** For a non-admin key and a non-dummy model,
  `load_inline_model` raises `401`. Here `api_key` and `admin_key` are the same value, so either
  client can trigger the load.
* **Nothing unloads it.** There is no `ttl`, `keep_alive` or `unload_after` anywhere in TabbyAPI —
  grep the tree. A loaded model occupies VRAM until `POST /v1/model/unload`.
* The log line `Draft model is disabled because a model name wasn't provided` at startup is
  **expected** with no `model_name`, not a fault.

### 5.1 Sampler settings — the setting most likely to be wrong

TabbyAPI's `config_sample.yml` default preset `safe_defaults` fills **temperature 0.8, min_p 0.05**
into any request that omits samplers. ANEMONE.md §4 says that is exactly what drives this model into
repetition loops during long reasoning. DeepSeek specify **temperature 1.0, top_p 1.0** (`0.95`
agentic), and `min_p` is not one of their recommendations.

So the config selects the pack's own `sampler_overrides/deepseek_v4.yml`. TabbyAPI resolves that
directory relative to its **working directory (`/app`)**, not relative to the mounted config, which
is why `install.sh` also copies the preset into the container path.

Sampling should not be pushed low and the output budget must be large: DeepSeek recommend up to
384K output tokens at `high`/`max` effort. Capping generation low returns a large
`reasoning_content` with empty `content`, which looks like a model fault and is not one.

---

## 6. The proxy service

`tabby-proxy@.service` is a **user** unit template, and that is the only supported install path. It
runs under `systemctl --user`, where `%h` means *your* home, and needs no sudo.

The instance name is the **path-encoded home directory**, not the username:

```bash
home=$(systemd-escape --path "$HOME")                       # -> home-logan
install -Dm644 tabby-proxy@.service ~/.config/systemd/user/tabby-proxy@.service
systemctl --user daemon-reload
systemctl --user enable --now "tabby-proxy@$home.service"
systemctl --user status "tabby-proxy@$home.service"
journalctl --user -u "tabby-proxy@$home.service" -f
```

### 6.1 Lingering

A user service is stopped at logout unless lingering is on, which is what lets it start at boot
without anyone logging in:

```bash
loginctl enable-linger "$USER"
loginctl show-user "$USER" -p Linger
```

### 6.2 The environment file

The optional `EnvironmentFile` holds anything the unit cannot take from the client:

```
# ~/.config/tabby-proxy.env
TABBY_API_URL=http://127.0.0.1:5000     # where TabbyAPI is
TABBY_PROXY_HOST=0.0.0.0                # where the proxy listens; default 127.0.0.1
TABBY_PROXY_PORT=8081                   # default 8081
TABBY_API_KEY=...                       # only for clients that send no key of their own
```

`TABBY_API_URL` points *upstream*; `TABBY_PROXY_HOST`/`TABBY_PROXY_PORT` are the proxy's own bind
and default to `127.0.0.1:8081`. Keeping the bind in the environment file rather than in the unit
means a deployment choice survives a reinstall of the unit.

`install.sh` writes the file only when `PROXY_HOST` is not the default:

```bash
PROXY_HOST=0.0.0.0 ./install.sh proxy
```

An existing file that says something else is **not** rewritten — it may hold a `TABBY_API_URL` or
API key placed there deliberately, and a repair run is not the moment to silently move a port. The
script reports the difference and leaves it.

### 6.3 Reaching the proxy from a container

Docker here is **rootless**, and rootless containers reach the host through `slirp4netns`, which
gives them no route to the host's `127.0.0.1` — nor to the bridge gateway `172.17.0.1`. The only
address that answers is the host's own routable IP:

```bash
ip -4 -o addr show scope global | awk '{print $4}' | cut -d/ -f1   # e.g. 130.63.104.106
```

So a containerised client needs the proxy on `0.0.0.0` and addressed by that IP:

```
http://130.63.104.106:8081/v1
```

That exposes the port to the LAN, which is the trade-off. The proxy authenticates with the same key
as TabbyAPI, so what is reachable is a key-checked API, not an open one; TabbyAPI itself stays on
loopback. For a host-only proxy, leave `TABBY_PROXY_HOST` unset and give the container a
host-network or a tunnel instead.

Check where a running proxy **actually** listens, rather than what a file says — a variable in the
environment does nothing if the running code does not read it:

```bash
ss -Hltnp 'sport = :8081'
```

### 6.4 Why not a system unit

An earlier revision shipped `tabby-proxy.service` as a **system** unit, installed by an
`install-tabby-proxy-service.sh` helper. Neither file is here any more and the system route is not
supported. In a system unit `%h` expands to the *system manager's* home (`/root`), not to the
`User=`'s home, so a `%h`-based `ExecStart` failed with `status=203/EXEC`. The two units would also
both bind the same port, so running both makes one crash-loop. If a system unit survives from an
earlier install:

```bash
sudo systemctl disable --now tabby-proxy.service
sudo rm /etc/systemd/system/tabby-proxy.service
```

---

## 7. Client configuration

### 7.1 Qwen Code

`~/.qwen/settings.json`:

```json
{
  "$version": 4,
  "env": { "OPENAI_API_KEY": "<your tabbyapi key>" },
  "modelProviders": {
    "openai": [
      {
        "id": "DeepSeek-V4-Flash-0731-exl3-2.32bpw",
        "name": "DeepSeek-V4-Flash-0731-exl3-2.32bpw",
        "baseUrl": "http://127.0.0.1:8081/v1",
        "envKey": "OPENAI_API_KEY"
      }
    ]
  },
  "model": { "name": "DeepSeek-V4-Flash-0731-exl3-2.32bpw" },
  "security": { "auth": { "baseUrl": "http://127.0.0.1:8081/v1", "selectedType": "openai" } }
}
```

`127.0.0.1` is correct here: Qwen Code runs on this host, and a proxy bound to `0.0.0.0` answers on
loopback too, so the address works either way. The model name must match the directory under
`model_dir` exactly — inline loading is strict, and a misspelt name gets a `404`.

The model's context window has to be declared on the **provider entry**, not in
`model.generationConfig`:

```json
{
  "id": "DeepSeek-V4-Flash-0731-exl3-2.32bpw",
  "name": "DeepSeek-V4-Flash-0731-exl3-2.32bpw",
  "baseUrl": "http://127.0.0.1:8081/v1",
  "envKey": "OPENAI_API_KEY",
  "generationConfig": { "contextWindowSize": 1048576 }
}
```

The docs list `contextWindowSize` under `model.generationConfig`, with a caveat about provider
models that reads as if it were specific to `enableRequestMetadata`. It is not: the setting is in
`MODEL_GENERATION_CONFIG_FIELDS`, and when a matching provider entry exists Qwen Code assigns
every field of that list from the entry unconditionally, overwriting the top-level copy rather
than merging it. The top-level form is only merged on the manual-credentials path. So the provider
entry is the effective placement, and a top-level `contextWindowSize` is silently ignored.

Getting this wrong is quiet rather than loud: with nothing declared, Qwen Code falls back to
`tokenLimit(modelId, "input")`, which for a model id it does not recognise — this one included —
returns a flat **1000000**, not the pack's real **1048576**. `/stats` then reports against the
wrong denominator. Verify the resolved window from the startup log:

```bash
node ~/.local/lib/qwen-code/lib/cli.js -m DeepSeek-V4-Flash-0731-exl3-2.32bpw -d -p "hi"
grep -o 'contextLimit=[0-9]*' "$(ls -t ~/.qwen/debug/*.txt | head -1)"
# contextLimit=1048576
```

`/v1/models` reports `n_ctx: null` for this upstream, so the client cannot infer the window from
the model list and the declaration above is the only way to get it right.

### 7.2 OpenWebUI

OpenWebUI is containerised, so it uses the host IP (`http://<host-ip>:8081/v1`) and the same key.

The address lives in OpenWebUI's database, not a config file, and the container's
`OPENAI_API_BASE_URL` is empty — so the database is what wins:

```bash
DB=~/.local/share/docker/volumes/open-webui/_data/webui.db
sqlite3 "$DB" "select json_extract(data,'\$.openai') from config;"
```

To change it, **stop the container first**, or the running app writes its in-memory copy back over
your edit on the next settings save. Back the database up first:

```bash
sqlite3 "$DB" ".backup '$DB.bak'"
```

The **admin** key is what makes `/v1/models` list the model even when nothing is resident; a
non-admin key only ever sees what is already loaded. Since the WebUI is also the client that
triggers the first load, it wants the admin key.

---

## 8. The API key

One key (`secrets.token_hex(16)`, 32 hex chars) is shared by TabbyAPI, the proxy and both clients.
It lives in `~/.qwen/settings.json` (`env.TABBY_API_KEY`) and is pinned into
`tabbyapi-config/api_tokens.yml` so that a container rebuild cannot rotate it silently.

`api_tokens.yml` accepts a single key or a list:

```yaml
api_key: <key>
admin_key: <key>
```

TabbyAPI reloads this file on change, so keys can be rotated without a restart.

**The proxy stores no key.** It forwards the client's own `Authorization` / `x-api-key` header. A
client that sends none gets `401` unless `TABBY_API_KEY` is exported in the service environment.

The load and unload scripts derive the key at runtime (`$TABBY_API_KEY`, else
`~/.qwen/settings.json`) rather than embedding it, so rotating the key cannot silently break them
the way a hardcoded token does.

---

## 9. Verifying

```bash
./install.sh verify   # container state, :5000, :8081, bind address, units, one authenticated completion
qwen --prompt "Solve 5 + 5"
```

By hand, after a change to the layout:

```bash
# Nothing is in VRAM at startup; the model appears only after a request names it
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# 200 with the admin key, 401 without one
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $KEY" \
    http://127.0.0.1:8081/v1/models

# Reachability from a container: 401 means reachable (it wants a key), 000 means nothing there
docker exec open-webui curl -s -o /dev/null -w '%{http_code}\n' \
    http://130.63.104.106:8081/v1/models
```

Status codes, and what each means:

| Code | Meaning |
|---|---|
| `401` | the service is alive and wants a key |
| `200` | alive, key accepted |
| `404` | inline loading rejected the model name — it is not in `model_dir` |
| `503` | a model is loading, or failed to load |
| `502` from the proxy | upstream is unreachable — the container is down |
| `000` | nothing is listening |

A live stream, bypassing the client:

```bash
curl -N -s http://127.0.0.1:8081/v1/chat/completions \
  -H "x-api-key: $TABBY_API_KEY" -H 'content-type: application/json' \
  -d '{"model": "DeepSeek-V4-Flash-0731-exl3-2.32bpw",
       "messages": [{"role": "user", "content": "Count from 1 to 5."}],
       "stream": true}'
```

---

## 10. Failure modes seen on this machine

Each of these actually happened, and each is now handled or documented.

| What it looked like | What it was | Handling |
|---|---|---|
| `denied: denied` pulling a public image | a stored ghcr credential without `read:packages`; Docker does not fall back to anonymous | `install.sh` retries with an empty `DOCKER_CONFIG` |
| container aborts at create | `--ulimit memlock=-1` on a rootless daemon | flag dropped (§4.6) |
| every request `401` after a rebuild | `api_tokens.yml` not mounted, so TabbyAPI generated its own key | mount it writable (§4.2) |
| container healthy, all requests fail to connect | `127.0.0.1` inside the container, unreachable through `-p` | `--host 0.0.0.0` (§4.3) |
| proxy answers `502`, upstream `000` | container did not come back after a reboot | `./install.sh tabby`; the container restart policy only revives containers that still exist |
| a variable in `tabby-proxy.env` had no effect | the unit was running a *different copy* of the proxy | unit repointed at this repo (§1) |
| `000` from a container to `:5000` | a container has no route to the host's loopback | route container clients through the proxy (§6.3) |
| `HTTP 401 Invalid API key` from the load/unload scripts | a hardcoded token had gone stale | key derived at runtime (§8) |
| a turn dies with `malformed tool call` | the model ended the call's JSON one brace early | bounded brace repair before giving up (§2.7) |
| `/stats` reports `0 / 0` tokens on tool turns | `stream_options` was dropped whenever tools were present | keep it, force `include_usage` (§2.6) |
| context usage computed against 1000000 | the window was never declared, so a generic fallback was used | declare it on the provider entry (§7.1) |

---

## 11. Machine-specific notes

`LD-INSTALL.md` (git-ignored) holds the things that are true only of this machine: absolute paths,
the GPU, the port layout, and a log of the 2026-09-25 repairs with the states verified after each.

Three files live outside this repository and are not installed by `install.sh`, so a machine rebuild
loses them: `~/.local/bin/tabby-reboot-verify.sh`, its
`~/.config/systemd/user/tabby-reboot-verify@.service`, and
`~/software/tools/{load,unload}-deepseek.bash`. The reboot-verify script matches `:8081` rather
than a literal address, because the proxy's bind is `TABBY_PROXY_HOST` and a pinned `127.0.0.1`
silently stops matching once the proxy is on `0.0.0.0`.
