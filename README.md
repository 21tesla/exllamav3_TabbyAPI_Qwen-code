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

#### contents of `tabby_proxy.py`:

```python
import os
import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

app = FastAPI(title="TabbyAPI Schema Guard Middleware")

TABBY_API_URL = os.getenv("TABBY_API_URL", "http://127.0.0.1:5000")

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
                    
                    # Fix empty function description
                    if not func.get("description"):
                        func["description"] = f"Executes {func.get('name', 'action')} routine."
                    
                    # Fix missing or null parameters (required by TabbyAPI)
                    if "parameters" not in func or func["parameters"] is None:
                        func["parameters"] = {"type": "object", "properties": {}}
                    
                    # Fix empty parameter/property descriptions
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

@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request):
    try:
        body = await request.json()
    except Exception:
        return Response(content='{"error": "Invalid JSON raw input body"}', status_code=400, media_type="application/json")
        
    patched_body = patch_tools_schema(body)
    
    async with httpx.AsyncClient() as client:
        # CRITICAL FIX: Strip content-length and host so httpx recalculates them for the larger patched body
        headers = {
            k: v for k, v in request.headers.items() 
            if k.lower() not in ("host", "content-length")
        }
        
        try:
            response = await client.post(
                f"{TABBY_API_URL}/v1/chat/completions",
                json=patched_body,
                headers=headers,
                timeout=60.0
            )
            return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))
        except httpx.ConnectError:
            return Response(
                content='{"error": "Could not connect to TabbyAPI downstream server."}', 
                status_code=502, 
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
```

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

