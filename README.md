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

I needed to do this privileged

```
sudo docker pull ghcr.io/theroyallab/tabbyapi:cu13
```


## Start TabbyAPI

To run the local model server, start TabbyAPI with a Docker command: My installation was in ~/software/tabbyapi and my DeepSeek model was in ~/models

```bash
sudo docker run --gpus all --shm-size=8g --name tabbyapi \
  -d \
  -p 5000:5000 \
  -v /home/logan/models:/app/models \
  -v /home/logan/software/tabbyapi-config:/app/config \
  --restart unless-stopped \
  ghcr.io/theroyallab/tabbyapi:cu13
```

* This commands exposes the local model directory (`~/models`) into TabbyAPI's workspace and uses port `5000`. Docker runs in the background and will restart if the system is rebooted.

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

### Add a system service
Create a new entry:

```
sudo nano /etc/systemd/system/tabby-proxy.service
```
Adjust the contents to reflect the location of exllamav3-anemone and the username

```                                   
[Unit]
Description=TabbyAPI Schema Guard Middleware Proxy
After=network.target docker.service

[Service]
Type=simple
User=logan
WorkingDirectory=/home/logan/software/exllamav3-anemone
ExecStart=/home/logan/software/exllamav3-anemone/venv/bin/python /home/logan/software/exllamav3-anemone/tabby_proxy.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Start the service

```
sudo systemctl daemon-reload
sudo systemctl enable tabby-proxy.service
sudo systemctl start tabby-proxy.service
```
---

## 4. Verification

To verify everything is working end-to-end, run a quick headless command from the terminal:

```bash
qwen --prompt "Solve 5 + 5"
```

