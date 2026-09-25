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
                            ^                     │
                            │                     v
                     ┌──────────────┐     ┌──────────────┐
                     │  OpenWebUI   │     │    model     │
                     │ (Port 3001)  │     │ (loads on    │
                     └──────────────┘     │  first use)  │
                                          └──────────────┘
```

Both clients point at the proxy, not at TabbyAPI directly. Qwen Code reaches it on `127.0.0.1`;
OpenWebUI runs in a container and reaches it on the host's routable address. The model is not
resident until something asks for it.

### Model loading on demand

The server starts with no model in VRAM. `config.yml` carries no `model_name`, and
`inline_model_loading: true` makes TabbyAPI load the model named in a request the first time one
arrives:

```yaml
model:
  model_dir: /app/models
  inline_model_loading: true
  use_dummy_models: true
```

* **The first request pays, the rest do not.** A cold load takes 10–20 s (48 modules, ~90 GB) and
  the caller waits through it. Afterwards the model stays resident and requests run at full speed.
* **Inline loading is strict.** A request naming a model that is not in `model_dir` gets a `404`
  and a model that fails to load gets a `503`, rather than silently being answered by whatever is
  loaded. `use_dummy_models: true` exempts the fixed names some clients always send
  (`gpt-3.5-turbo`), which then run on the loaded model.
* **The loading request needs an admin key.** TabbyAPI refuses to swap models for a non-admin key,
  because otherwise any caller could move ~90 GB. Use the same key for `api_key` and `admin_key`
  if the same client does both.
* **Nothing unloads it again.** TabbyAPI has no idle timeout, so a loaded model occupies VRAM until
  `POST /v1/model/unload`. Use `software/tools/unload-deepseek.bash` to give the GPU back.

If you would rather have the model ready before the first user arrives, load it ahead of time:

```bash
software/tools/load-deepseek.bash      # ~11 s, returns when the model is resident
software/tools/unload-deepseek.bash    # gives the VRAM back
```

Both scripts read the key from `TABBY_API_KEY`, falling back to `~/.qwen/settings.json`, so neither
carries a copy of it.

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
* **`model_name` is deliberately left out of `config.yml`.** Setting it makes TabbyAPI load the
  model at startup, where it then holds ~90 GB of VRAM around the clock, including between
  sessions. Left out, the server starts with nothing resident and `inline_model_loading: true`
  loads the model on the first request that names it — the first caller waits 10–20 s, everyone
  after that does not. The cost is a cold-start delay instead of a permanently occupied GPU; run
  `software/tools/load-deepseek.bash` ahead of time if you would rather pay it early. There is no
  idle-unload in TabbyAPI (no `ttl`, `keep_alive` or `unload_after` anywhere in the tree), so once
  loaded the model stays until `POST /v1/model/unload`.
* **`--restart unless-stopped` is what brings the container back after a reboot.** Without it the
  container does not return, and nothing announces that. Note that Docker restart policies only
  revive containers that still exist: delete the container and the next boot will not recreate it.
* The model directory is mounted read-only in effect (`~/models` → `/app/models`) and the server
  listens on `5000`. That port is bound to **loopback only**, so TabbyAPI itself is not reachable
  from the network. Clients that cannot use `127.0.0.1` go through the proxy instead — see
  [Reaching the proxy from a container](#reaching-the-proxy-from-a-container).
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

`127.0.0.1` is right here: Qwen Code runs on this host. The proxy's *default* bind is
`127.0.0.1:8081`, and a proxy bound to `0.0.0.0` answers on loopback too — so this address keeps
working either way. The model name must match the directory under `model_dir` exactly: inline
loading is strict, and a misspelt name gets a `404` rather than an answer from whatever is loaded.

---

## Configure OpenWebUI

OpenWebUI runs in a container, so it cannot use the same address Qwen Code does. Point it at the
proxy on the host's routable IP:

```
http://<host-ip>:8081/v1
```

and give it the same API key. The proxy has to be listening on `0.0.0.0` for this to work — see
[Reaching the proxy from a container](#reaching-the-proxy-from-a-container).

The address is stored in OpenWebUI's database, not in a config file, and the container's
`OPENAI_API_BASE_URL` is empty, so the database is what wins. Read it back with:

```bash
DB=~/.local/share/docker/volumes/open-webui/_data/webui.db
sqlite3 "$DB" "select json_extract(data,'\$.openai') from config;"
```

If you edit it directly, **stop the container first** — otherwise the running app writes its
in-memory copy back over your change. Back the database up before writing to it:

```bash
sqlite3 "$DB" ".backup '$DB.bak'"
```

The admin key is what makes `/v1/models` list the model even when nothing is loaded; a non-admin key
only ever sees what is already resident. Since the WebUI is also the client that triggers the first
load, it wants the admin key.

---

## TabbyAPI Schema Guard Middleware (`tabby_proxy.py`)

###  Problem
When Qwen Code CLI registers built-in tools (such as `get_goal` or `list_agents`), it submits tool schemas that omit the `"parameters"` field since they take no arguments. However, TabbyAPI runs strict Pydantic model validation on incoming tool schemas and will reject any request missing the `"parameters"` field with a `422 Unprocessable Content` error.

###  Solution
`tabby_proxy.py` in this repository. Its `patch_tools_schema` intercepts requests and supplies a
default empty schema parameter object.

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

The unit names two directories, on purpose:

* **the proxy** comes from this repository — `%h/software/exllamav3_TabbyAPI_Qwen-code/tabby_proxy.py`,
  which is also the unit's `WorkingDirectory`
* **the interpreter** comes from the ExLlamaV3 checkout —
  `%h/software/exllamav3-anemone/venv/bin/python`, the only place `uvicorn` and `httpx` are installed

Splitting them is what lets the proxy be edited here and take effect on `systemctl --user restart`,
with nothing copied into a read-only upstream checkout. An earlier revision ran the fork's copy of
the proxy instead, which meant every fix had to be copied across by hand and the two silently
drifted.

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

The optional `EnvironmentFile` holds anything the unit cannot pick up from the client, including
where the proxy itself listens:

```
# /home/logan/.config/tabby-proxy.env
TABBY_API_URL=http://127.0.0.1:5000
TABBY_PROXY_HOST=0.0.0.0
TABBY_PROXY_PORT=8081
TABBY_API_KEY=...
```

`TABBY_API_URL` points at TabbyAPI; `TABBY_PROXY_HOST` and `TABBY_PROXY_PORT` are where the proxy
accepts connections and default to `127.0.0.1:8081` when unset. `install.sh` writes the file when
`PROXY_HOST` is not the default:

```bash
PROXY_HOST=0.0.0.0 ./install.sh proxy
```

#### Reaching the proxy from a container

Docker here is **rootless**, and rootless containers reach the host through `slirp4netns`, which
gives them no route to the host's `127.0.0.1` — nor to the bridge gateway `172.17.0.1`. The only
address that answers is the host's own routable IP:

```bash
ip -4 -o addr show scope global | awk '{print $4}' | cut -d/ -f1   # e.g. 130.63.104.106
```

So a containerised client (OpenWebUI) needs the proxy on `0.0.0.0` and addressed by that IP:

```
http://130.63.104.106:8081/v1
```

That does expose the port to the LAN, which is the trade-off. The proxy authenticates with the
same key as TabbyAPI, so what is reachable is a key-checked API, not an open one. TabbyAPI itself
stays on loopback. If the proxy should stay host-only, leave `TABBY_PROXY_HOST` unset and give the
container a host-network or a tunnel instead.

Check where a running proxy actually listens, rather than what the file says:

```bash
ss -Hltnp 'sport = :8081'
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

`./install.sh verify` checks the same chain without a client: container state, upstream on `:5000`,
proxy on `:8081`, the proxy's bind address, which units are active, and one authenticated completion
through the proxy. A `401` there means the key in `api_tokens.yml` and the one in
`~/.qwen/settings.json` have diverged.

Two things are worth checking by hand after a change to the layout:

```bash
# Nothing is in VRAM at startup; the model appears only after a request names it
nvidia-smi --query-gpu=memory.used --format=csv,noheader
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $KEY" \
    http://127.0.0.1:8081/v1/models      # 200 with the admin key, 401 without one

# A containerised client can reach the proxy (401 = reachable, needs a key)
docker exec open-webui curl -s -o /dev/null -w '%{http_code}\n' \
    http://130.63.104.106:8081/v1/models
```

If OpenWebUI reports that it cannot connect, check its stored base URL rather than the UI — the
value lives in its database and an empty `OPENAI_API_BASE_URL` in the container environment means
the database wins:

```bash
DB=~/.local/share/docker/volumes/open-webui/_data/webui.db
sqlite3 "$DB" "select json_extract(data,'\$.openai.api_base_urls') from config;"
```

Change it with the container **stopped**, or the running app will write its in-memory copy back
over your edit on the next settings save.


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
