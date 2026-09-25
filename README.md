# Integrating TabbyAPI with Qwen Code CLI

This document outlines the setup and configuration required to successfully run the **Qwen Code CLI** alongside **TabbyAPI** using a local schema guard middleware.

---

## Architecture Overview

Qwen Code CLI communicates with TabbyAPI via a lightweight FastAPI proxy. This proxy intercepts chat completion requests to fix tool schema definitions so they conform to TabbyAPI's strict validation rules.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Qwen Code   │ ──> │  tabby_proxy │ ──> │   TabbyAPI   │
│  Interactive │     │ (Port 8081)  │     │ (Port 5000)  │
└──────────────┘     └──────────────┘     └──────────────┘
```

`install.sh` performs every step below in order, and is idempotent — re-running it is the intended
way to repair a machine whose TabbyAPI container did not come back after a reboot:

```bash
./install.sh                    # all steps
./install.sh tabby proxy        # just those two: the usual post-reboot repair
./install.sh --help
```

It takes the same values as environment overrides (`MODELS_DIR`, `CONFIG_DIR`, `TABBY_API_KEY`,
...). The rest of this document explains what each step does and why.

---

## Download the model 

Deepseek for Blackwell RTX-6000:

```
hf download anoane/DeepSeek-V4-Flash-0731-exl3-2.32bpw \
--local-dir ./DeepSeek-V4-Flash-0731-exl3-2.32bpw
```

## Download exllama 

I used the anemone version

```
git clone https://github.com/anoane/exllamav3-anemone
```
Building it

```
python -m venv venv

./venv/bin/pip install --upgrade pip setuptools wheel

./venv/bin/pip install -r requirements.txt

TORCH_CUDA_ARCH_LIST="12.0" ./venv/bin/pip install -e . \
 --no-build-isolation

```
## Obtain TabbyAPI

```
docker pull ghcr.io/theroyallab/tabbyapi:cu13
```

Neither the pull nor the service needs root: Docker runs in **rootless** mode here
(`unix:///run/user/1000/docker.sock`), so plain `docker` reaches the daemon.

If the pull answers **`denied: denied`**, the image is not private — a stored ghcr credential is
being presented that cannot read it. `~/.docker/config.json` may hold an expired or scope-limited
token for `ghcr.io`, and Docker will not fall back to anonymous access on its own. Verify that
anonymity works, then pull without the credential:

```bash
curl -s "https://ghcr.io/token?scope=repository:theroyallab/tabbyapi:pull&service=ghcr.io"   # a token, not a denial
mkdir -p /tmp/docker-nocreds && printf '{}' > /tmp/docker-nocreds/config.json
DOCKER_CONFIG=/tmp/docker-nocreds docker pull ghcr.io/theroyallab/tabbyapi:cu13
```

`install.sh` does this automatically: it retries anonymously if the authenticated pull fails.

## Start TabbyAPI

```bash
docker run --gpus all --shm-size=8g --name tabbyapi \
  -d \
  -p 127.0.0.1:5000:5000 \
  --entrypoint python3 \
  -v /home/logan/models:/app/models \
  -v /home/logan/software/tabbyapi-config/config.yml:/app/config.yml:ro \
  -v /home/logan/software/tabbyapi-config/api_tokens.yml:/app/api_tokens.yml \
  -v /home/logan/software/tabbyapi-config/sampler_overrides:/app/sampler_overrides:ro \
  --restart unless-stopped \
  ghcr.io/theroyallab/tabbyapi:cu13 \
  main.py --host 0.0.0.0
```

Every part of that command is load-bearing, and these are easy to get wrong in ways that fail
*silently* rather than loudly:

* **`config.yml` and `api_tokens.yml` are mounted as files, not as a directory.** TabbyAPI reads
  `pathlib.Path("config.yml")` and `pathlib.Path("api_tokens.yml")` relative to its working
  directory, and the image's own code lives in `/app` (`WORKDIR /app`, then `COPY . .`). Mounting a
  directory at `/app` would shadow the application *and* hide the config from it — the container
  would start on defaults with no error. `/app/sampler_overrides` is resolved the same relative
  way, which is why the preset directory is mounted there.
* **`api_tokens.yml` must be mounted, or your key changes without asking.** With no auth file at
  `/app/api_tokens.yml`, TabbyAPI generates a fresh `token_hex(16)` at startup, writes it into the
  container, and prints it to the log. Every request then fails `401` against the key in
  `~/.qwen/settings.json`. It is mounted writable so that generation still works.
* **`--host 0.0.0.0`.** The default is `127.0.0.1`, which inside a container is the container's own
  loopback that `-p 5000:5000` cannot reach. Command-line arguments override `config.yml`, so this
  and `network.host` must agree.
* **`model_name` must be set in `config.yml`.** TabbyAPI only loads a model at startup when
  `model.model_name` is present. With `model_dir` alone it starts an empty server that answers
  every request with "no model loaded".
* `--restart unless-stopped` is what brings the container back after a reboot. Without it the
  container does not return, and nothing announces that.
* The model directory is mounted read-only in effect (`~/models` → `/app/models`) and the server
  listens on `5000`, bound to loopback so it is not exposed to the network.
* No `--ulimit memlock=-1`, although upstream suggests it. Docker here is **rootless**, and an
  unprivileged daemon cannot raise `RLIMIT_MEMLOCK`; passing it fails the container at OCI-create
  time with `error setting rlimit type 8: operation not permitted`.


---

## Configure Qwen CLI Settings

The `~/.qwen/settings.json` file must be configured to route requests through the local proxy and select the loaded model by default.

### Key Modifications in `~/.qwen/settings.json`:

```json
{
  "$version": 4,
  "env": {
    "OPENAI_API_KEY": "<your tabbyapi key>",
    "OPENAI_API_BASE": "http://127.0.0.1:8081/v1"
  },
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
  "model": {
    "name": "DeepSeek-V4-Flash-0731-exl3-2.32bpw"
  },
  "security": {
    "auth": {
      "baseUrl": "http://127.0.0.1:8081/v1",
      "selectedType": "openai"
    }
  }
}
```

---

## TabbyAPI Schema Guard Middleware (`tabby_proxy.py`)

###  Problem
When Qwen Code CLI registers built-in tools (such as `get_goal` or `list_agents`), it submits tool schemas that omit the `"parameters"` field since they take no arguments. However, TabbyAPI runs strict Pydantic model validation on incoming tool schemas and will reject any request missing the `"parameters"` field with a `422 Unprocessable Content` error.

###  Solution
A pre-validation block was written the exllamav3 directory, in my case, it was`/home/logan/software/exllamav3-anemone/tabby_proxy.py`. The `patch_tools_schema`  intercepts and supplies a default empty schema parameter object.

### Install the proxy service

The proxy ships as `tabby-proxy@.service`, a **user** unit template, and that is the only
supported install path. It runs under `systemctl --user`, where `%h` means *your* home, and it
needs no sudo.

The instance name is the **path-encoded home directory**, not the username:

```bash
home=$(systemd-escape --path "$HOME")                          # -> home-logan
systemctl --user daemon-reload
install -Dm644 tabby-proxy@.service ~/.config/systemd/user/tabby-proxy@.service
systemctl --user enable --now "tabby-proxy@$home.service"      # instance: tabby-proxy@home-logan.service
systemctl --user status "tabby-proxy@$home.service"
journalctl --user -u "tabby-proxy@$home.service" -f
```

The unit's `ExecStart` runs the proxy from the ExLlamaV3 checkout
(`%h/software/exllamav3-anemone/tabby_proxy.py`), so that copy is the one that serves; keep it in
step with the copy in this repository.

To reload after editing the proxy:

```bash
systemctl --user restart "tabby-proxy@$home.service"
```

A user service is stopped at logout unless lingering is on, which is what lets it start at boot
without you logging in:

```bash
loginctl enable-linger "$USER"
loginctl show-user "$USER" -p Linger
```

The optional `EnvironmentFile` holds anything the unit cannot pick up from the client (see
[Upstream API key](#upstream-api-key)):

```
# /home/logan/.config/tabby-proxy.env
TABBY_API_URL=http://127.0.0.1:5000
TABBY_API_KEY=...
```

#### Why not a system unit

An earlier revision of this repository shipped `tabby-proxy.service` as a **system** unit, to be
copied into `/etc/systemd/system/` by an `install-tabby-proxy-service.sh` helper. Neither file is
here any more, and the system route is not supported. In a system unit `%h` expands to the *system
manager's* home (`/root`), not to the `User=`'s home, so a `%h`-based `ExecStart` failed to start
with `status=203/EXEC`. The two units also both bind `127.0.0.1:8081`, so running both makes one
crash-loop. If a system unit survives from an earlier install, remove it before starting the user
instance:

```bash
sudo systemctl disable --now tabby-proxy.service
sudo rm /etc/systemd/system/tabby-proxy.service
```

---

## 4. Verification

To verify everything is working end-to-end, run a quick headless command from the terminal:

```bash
qwen --prompt "Solve 5 + 5"
```


---

## Tool Calling: DSML Interception

### Problem
With `tools` in the request, this ExLlama/TabbyAPI build answers
`finish_reason: "tool_calls"` but leaves **`tool_calls: null`**. The model's tool
call then exists only as DeepSeek **DSML text** inside `content` (or
`reasoning_content`). A client that trusts `finish_reason` renders an empty
assistant turn and the session stops with no error at all.

### Solution
The proxy injects the tool catalogue and the tool-call format into the system
prompt, parses the tool call back out of the text as it arrives, and re-emits the
result as a valid OpenAI SSE stream carrying proper `tool_calls` deltas (see
[Streaming and upstream retries](#streaming-and-upstream-retries)).

The DSML the model emits is not stable between turns — this checkpoint uses at
least four dialects, so the parser walks the tags instead of matching one regex:

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

If a dialect still escapes it, the proxy logs
`Tool-call syntax present but unparsed: ...` to the journal rather than silently
dropping the call:

```bash
journalctl --user -u "tabby-proxy@$home.service" -f
```

### Self-test
The parser ships with fixtures for each dialect above, plus fixtures for the
streaming hold-back window:

```bash
/home/logan/software/exllamav3-anemone/venv/bin/python tabby_proxy.py --selftest
```

### Upstream API key
The proxy forwards the client's own `Authorization` / `x-api-key` header, so no
key is stored in this repository or in the service unit. A client that sends no
key receives `401` unless `TABBY_API_KEY` is exported in the service
environment, e.g. through `/home/logan/.config/tabby-proxy.env`:

```
# /home/logan/.config/tabby-proxy.env
TABBY_API_URL=http://127.0.0.1:5000
TABBY_API_KEY=...
```

---

## Streaming and upstream retries

### Retries while a model loads
TabbyAPI answers `503` for a few seconds while it loads or reloads a model, and
occasionally `502`/`529` under load. A single such answer used to end the turn.
Every upstream call now retries up to 5 times with exponential backoff (2 s, 4 s,
8 s, 16 s) on `429`/`500`/`502`/`503`/`504`/`529` and on connection failures, so
a model reload no longer kills a request. A non-retryable status is returned to
the client unchanged.

### Tool calls without the stall
Requests carrying `tools` used to be forced non-streaming so the whole
completion could be parsed before answering, so the client sat silent for the
entire generation. The tool path is now incremental: upstream deltas are relayed
as they arrive, with the last 32 characters of `content` and of
`reasoning_content` held back. `extract_tool_calls` discards everything from the
first tool-syntax marker onwards and keeps everything before it, so holding back
that window guarantees a marker is never emitted half-formed; a marker spanning
two deltas is caught by the tail of the window. When no marker appears, the held
tail is flushed at the end, and the final `finish_reason` and `tool_calls` deltas
come from the same parser the non-streaming path uses.

The hold-back exists because markers are recognised only once their keyword has
arrived: `<tool_c` reads as prose, so without it a split `<tool_call>` could
leak into the visible answer. `_HOLDBACK` must stay above
`_MARKER_PREFIX_LIMIT`, which the self-test checks.

### Reasoning streams too
TabbyAPI streams `reasoning_content` separately from `content`, and that is
where most of the visible latency lives, so the first reasoning delta now
reaches the client within milliseconds of the model starting rather than after
the whole generation.

### Verifying a live stream
```bash
curl -N -s http://127.0.0.1:8081/v1/chat/completions \
  -H "x-api-key: $TABBY_API_KEY" -H 'content-type: application/json' \
  -d '{"model": "DeepSeek-V4-Flash-0731-exl3-2.32bpw",
       "messages": [{"role": "user", "content": "Count from 1 to 5."}],
       "stream": true}'
```
