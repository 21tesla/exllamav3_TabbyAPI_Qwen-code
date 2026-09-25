# Integrating TabbyAPI with Qwen Code CLI

Run a local **DeepSeek-V4-Flash** model behind **TabbyAPI** and use it from **Qwen Code** and
**OpenWebUI**. This repository is the adapter between the clients and TabbyAPI: a small FastAPI
proxy, a systemd user unit, and one install script.

The proxy is not optional scaffolding. TabbyAPI and this model fail in two specific ways that a
client cannot work around, and the proxy is where both are fixed.

## What this solves

| Symptom without the proxy | Cause | Fixed by |
|---|---|---|
| `422 Unprocessable Content` on any request | Qwen Code sends tool schemas with no `parameters` key; TabbyAPI validates strictly | `patch_tools_schema`, which supplies an empty schema |
| `finish_reason: "tool_calls"` but `tool_calls: null` | This build emits the call as DeepSeek **DSML text** in `content`, leaving `tool_calls` empty; a client that trusts `finish_reason` renders a blank turn and stops | the DSML parser, which converts the text back into real `tool_calls` deltas |

Both are covered in depth in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Architecture

```
┌──────────────┐
│  Qwen Code   │ ────┐
│  (host)      │     │     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
└──────────────┘     ├───> │  tabby_proxy │ ──> │   TabbyAPI   │ ──> │    model     │
                     │     │    :8081     │     │    :5000     │     │ (loads on    │
┌──────────────┐     │     │              │     │  loopback    │     │  first use)  │
│  OpenWebUI   │ ────┘     └──────────────┘     │   only       │     └──────────────┘
│  (container) │                               └──────────────┘
└──────────────┘
```

Both clients point at the proxy, never at TabbyAPI directly. Qwen Code runs on this host and uses
`127.0.0.1:8081`; OpenWebUI runs in a container, cannot reach the host's loopback, and uses the
host's routable IP on the same port.

## Quick start

```bash
./install.sh              # every step, in order, idempotent
./install.sh tabby proxy  # the usual repair after a reboot
./install.sh verify       # check the chain without a client
./install.sh --help
```

`install.sh` is the executable form of this document and of [IMPLEMENTATION.md](IMPLEMENTATION.md):
it downloads the model and the ExLlamaV3 build, writes the TabbyAPI config, starts the container,
installs the proxy service, and verifies the result. It needs no root, and re-running it is the
intended way to repair a stack that came back wrong. Paths and names are overridable —
`MODELS_DIR=... ./install.sh`, `PROXY_HOST=0.0.0.0 ./install.sh proxy`.

Prerequisites — the script checks for each of these and stops with a clear message if one is
missing:

| Needed | For |
|---|---|
| Docker, with the daemon running (rootless here, so plain `docker` works) | the TabbyAPI container |
| systemd user session (`systemctl --user`) | the proxy service |
| `python3` | to build the fork's venv |
| the ExLlamaV3 fork, already cloned | the CUDA extension build — the script does not clone it |
| the `hf` CLI (`pip install huggingface_hub`) | the model download step only |
| an NVIDIA GPU and a recent driver | everything |

Nothing here needs root.

## What runs where

| Component | Port | Reachable from | Started by |
|---|---|---|---|
| TabbyAPI (Docker) | 5000 | **loopback only** | `--restart unless-stopped`, plus the `docker` user unit |
| `tabby_proxy` (user service) | 8081 | `0.0.0.0` — loopback and the LAN | `tabby-proxy@<home>.service`, enabled, with lingering on |
| OpenWebUI (Docker) | 3001 | `0.0.0.0` | its own container policy |

TabbyAPI is deliberately *not* exposed to the network. The proxy is, because a containerised client
has no other route to it — but the proxy authenticates with the same key, so what is reachable is a
key-checked API rather than an open one.

Reboot behaviour: the proxy, the container and OpenWebUI all come back on their own. Docker restart
policies only revive containers that still exist, so a *deleted* container does not return — that is
what `./install.sh tabby` is for.

## The model loads on demand

TabbyAPI starts with **nothing in VRAM**. The model is loaded by the first request that names it,
which is why a cold request takes 10–20 s and every later one does not.

```bash
# trigger the load without waiting for a user, ~11 s
curl -N -sS -X POST http://127.0.0.1:8081/v1/model/load \
  -H "Authorization: Bearer $TABBY_API_KEY" -H 'content-type: application/json' \
  -d '{"model_name": "DeepSeek-V4-Flash-0731-exl3-2.32bpw"}'

# give the ~90 GB back (the container stays up)
curl -N -sS -X POST http://127.0.0.1:8081/v1/model/unload \
  -H "Authorization: Bearer $TABBY_API_KEY"
```

This is a deliberate trade: a cold-start delay when the model is first used, instead of a
permanently occupied GPU. **TabbyAPI has no idle timeout**, so nothing hands the memory back on its
own — unload is explicit, or the model stays resident until the container restarts.

## Configure Qwen Code

Point `~/.qwen/settings.json` at the proxy and name the model:

```json
{
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
  "model": { "name": "DeepSeek-V4-Flash-0731-exl3-2.32bpw" }
}
```

The model name must match the directory under `model_dir` **exactly** — inline loading is strict, so
a typo gets a `404` rather than a fallback answer.

## Configure OpenWebUI

OpenWebUI cannot use `127.0.0.1`, so give it the host's routable IP and the same key:

```
http://<host-ip>:8081/v1
```

The address is stored in OpenWebUI's database (`webui.db`), not in a config file. Editing it
directly requires stopping the container first, or the running app writes its in-memory copy back
over your change. [IMPLEMENTATION.md](IMPLEMENTATION.md) has the exact SQL and the backup step.

## Verify it works

```bash
qwen --prompt "Solve 5 + 5"     # end-to-end through the proxy
./install.sh verify             # container, both ports, units, one authenticated completion
```

`401` means a service is alive but wants a key; `000` means nothing is listening.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `422` on every request | the client is talking to `:5000` directly — route it through `:8081` |
| `finish_reason: "tool_calls"` with no `tool_calls` | same: the DSML fix lives in the proxy |
| `401 Invalid API key` | the key in `api_tokens.yml` and the client's key have diverged |
| proxy answers `502`, upstream `000` | the TabbyAPI container is not running — `./install.sh tabby` |
| OpenWebUI reports it cannot connect | its stored base URL, or the proxy is not bound to `0.0.0.0` |
| first request takes ~15 s | not a fault — that is the model loading |
| `denied: denied` when pulling the image | a stored ghcr credential; `install.sh` retries anonymously |

## Documentation

| File | Role |
|---|---|
| `README.md` (this file) | orientation: what it does, how to run it, how to tell it is working |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | the detail: why each mount and flag is load-bearing, the service, the parser internals, the API key, the failure modes behind each design choice |
| `install.sh` | the executable form of both documents |
| `LD-INSTALL.md` | **this machine only** — absolute paths, the GPU, and a log of the 2026-09-25 repairs. Deliberately git-ignored |
| `tabby-proxy@.service`, `tabby_proxy.py` | the two artifacts that actually ship |
