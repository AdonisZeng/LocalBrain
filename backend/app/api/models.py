from fastapi import APIRouter, HTTPException
import httpx
import os

router = APIRouter(tags=["models"])


def get_no_proxy_client():
    env = {
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "http_proxy": "",
        "https_proxy": "",
    }
    return httpx.AsyncClient(timeout=10.0, trust_env=True)


def get_models_list_url(provider: str, base_url: str) -> str:
    """
    获取模型列表的 URL
    - LM Studio: base_url 可能是 http://localhost:1234/v1，需要去掉 /v1 后缀
    - Ollama: 直接使用 base_url
    """
    url = base_url.rstrip("/")
    if provider == "lmstudio":
        if url.endswith("/v1"):
            url = url[:-3]
    return url


@router.get("/available")
async def get_available_models(provider: str, base_url: str = ""):
    """获取 LM Studio 或 Ollama 中已加载的模型列表"""
    try:
        old_http = os.environ.pop("HTTP_PROXY", None)
        old_https = os.environ.pop("HTTPS_PROXY", None)
        old_http_lower = os.environ.pop("http_proxy", None)
        old_https_lower = os.environ.pop("https_proxy", None)
        
        try:
            if provider == "ollama":
                async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
                    response = await client.get(f"{base_url}/api/tags")
                    if response.status_code == 200:
                        data = response.json()
                        models = [m["name"] for m in data.get("models", [])]
                        return {"models": models}
                    return {"models": [], "error": "Failed to connect to Ollama"}
            
            elif provider == "lmstudio":
                url = get_models_list_url(provider, base_url)
                async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
                    response = await client.get(url + "/v1/models")
                    if response.status_code == 200:
                        data = response.json()
                        models = [m["id"] for m in data.get("data", [])]
                        return {"models": models}
                    return {"models": [], "error": f"Failed to connect to LM Studio: {response.status_code}"}
            
            else:
                return {"models": [], "error": "Unsupported provider"}
        finally:
            if old_http: os.environ["HTTP_PROXY"] = old_http
            if old_https: os.environ["HTTPS_PROXY"] = old_https
            if old_http_lower: os.environ["http_proxy"] = old_http_lower
            if old_https_lower: os.environ["https_proxy"] = old_https_lower
            
    except Exception as e:
        return {"models": [], "error": str(e)}


@router.post("/test-connection")
async def test_model_connection(provider: str, base_url: str, model_name: str = "", api_key: str = ""):
    """测试模型连接是否正常"""
    try:
        old_http = os.environ.pop("HTTP_PROXY", None)
        old_https = os.environ.pop("HTTPS_PROXY", None)
        old_http_lower = os.environ.pop("http_proxy", None)
        old_https_lower = os.environ.pop("https_proxy", None)
        
        try:
            if provider == "ollama":
                async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
                    if model_name:
                        response = await client.post(
                            f"{base_url}/api/generate",
                            json={"model": model_name, "prompt": "Hi", "stream": False}
                        )
                        if response.status_code == 200:
                            return {"success": True, "message": f"成功连接到 Ollama，模型 {model_name} 可用"}
                        return {"success": False, "error": f"模型 {model_name} 不可用"}
                    else:
                        response = await client.get(f"{base_url}/api/tags")
                        if response.status_code == 200:
                            return {"success": True, "message": "成功连接到 Ollama"}
                        return {"success": False, "error": "无法连接到 Ollama"}
            
            elif provider == "lmstudio":
                url = base_url.rstrip("/")
                async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
                    if model_name:
                        response = await client.post(
                            f"{url}/chat/completions",
                            json={
                                "model": model_name,
                                "messages": [{"role": "user", "content": "Hi"}],
                                "max_tokens": 5
                            }
                        )
                        if response.status_code == 200:
                            return {"success": True, "message": f"成功连接到 LM Studio，模型 {model_name} 可用"}
                        return {"success": False, "error": f"模型 {model_name} 不可用: {response.status_code}"}
                    else:
                        models_url = get_models_list_url(provider, base_url)
                        response = await client.get(models_url + "/v1/models")
                        if response.status_code == 200:
                            return {"success": True, "message": "成功连接到 LM Studio"}
                        return {"success": False, "error": "无法连接到 LM Studio"}
            
            elif provider == "openai":
                if not api_key:
                    return {"success": False, "error": "需要提供 API Key"}
                async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
                    headers = {"Authorization": f"Bearer {api_key}"}
                    response = await client.get(f"{base_url}/models", headers=headers)
                    if response.status_code == 200:
                        return {"success": True, "message": "成功连接到 OpenAI"}
                    return {"success": False, "error": f"连接失败: {response.status_code}"}
            
            elif provider == "anthropic":
                if not api_key:
                    return {"success": False, "error": "需要提供 API Key"}
                return {"success": True, "message": "Anthropic 配置已保存，请在实际使用时验证"}
            
            elif provider == "huggingface":
                return {"success": True, "message": f"HuggingFace 嵌入模型 {model_name} 已配置"}
            
            elif provider == "custom":
                if not base_url:
                    return {"success": False, "error": "需要提供 Base URL"}
                async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
                    headers = {}
                    if api_key:
                        headers["Authorization"] = f"Bearer {api_key}"
                    test_url = base_url.rstrip("/") + "/models"
                    response = await client.get(test_url, headers=headers)
                    if response.status_code == 200:
                        return {"success": True, "message": "成功连接到自定义服务"}
                    return {"success": False, "error": f"连接失败: {response.status_code}"}
            
            else:
                return {"success": False, "error": "不支持的提供商"}
        finally:
            if old_http: os.environ["HTTP_PROXY"] = old_http
            if old_https: os.environ["HTTPS_PROXY"] = old_https
            if old_http_lower: os.environ["http_proxy"] = old_http_lower
            if old_https_lower: os.environ["https_proxy"] = old_https_lower
            
    except Exception as e:
        return {"success": False, "error": str(e)}
