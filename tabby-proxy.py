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
