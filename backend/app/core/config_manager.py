from pathlib import Path
from typing import Optional
import yaml
import shutil
from datetime import datetime


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
_backup_dir = PROJECT_ROOT / "data" / "config_backups"


def _ensure_backup_dir():
    _backup_dir.mkdir(parents=True, exist_ok=True)


def _backup_config():
    _ensure_backup_dir()
    if CONFIG_FILE.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = _backup_dir / f"config_{timestamp}.yaml"
        shutil.copy2(CONFIG_FILE, backup_path)


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(config_dict: dict):
    _backup_config()
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config_dict, f, allow_unicode=True, default_flow_style=False)


def get_model_settings() -> dict:
    config = load_config()
    
    models = config.get("models", {})
    
    llm = models.get("llm", {})
    llm_providers = llm.get("providers", {})
    
    current_llm_provider = llm.get("provider", "")
    if current_llm_provider and current_llm_provider in llm_providers:
        llm_settings = llm_providers[current_llm_provider]
    else:
        llm_settings = {}
    
    embedding = models.get("embedding", {})
    embedding_providers = embedding.get("providers", {})
    
    current_embedding_provider = embedding.get("provider", "huggingface")
    if current_embedding_provider in embedding_providers:
        embedding_settings = embedding_providers[current_embedding_provider]
    else:
        embedding_settings = {}
    
    return {
        "llm": {
            "provider": current_llm_provider,
            "base_url": llm_settings.get("base_url", "https://api.openai.com/v1"),
            "model_name": llm_settings.get("model_name", "gpt-3.5-turbo"),
            "api_key": llm_settings.get("api_key", ""),
        },
        "embedding": {
            "provider": current_embedding_provider,
            "base_url": embedding_settings.get("base_url", ""),
            "model_name": embedding_settings.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            "dimension": embedding_settings.get("dimension", 384),
        }
    }


def update_model_settings(llm_config: dict, embedding_config: dict):
    config = load_config()
    
    if "models" not in config:
        config["models"] = {}
    
    if "llm" not in config["models"]:
        config["models"]["llm"] = {"providers": {}}
    
    if "embedding" not in config["models"]:
        config["models"]["embedding"] = {"providers": {}}
    
    llm_provider = llm_config.get("provider", "openai")
    embedding_provider = embedding_config.get("provider", "huggingface")
    
    config["models"]["llm"]["provider"] = llm_provider
    config["models"]["llm"]["providers"][llm_provider] = {
        "base_url": llm_config.get("base_url", "https://api.openai.com/v1"),
        "model_name": llm_config.get("model_name", "gpt-3.5-turbo"),
        "api_key": llm_config.get("api_key", ""),
    }
    
    config["models"]["embedding"]["provider"] = embedding_provider
    config["models"]["embedding"]["providers"][embedding_provider] = {
        "base_url": embedding_config.get("base_url", ""),
        "model_name": embedding_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        "dimension": embedding_config.get("dimension", 384),
    }
    
    save_config(config)


def update_rag_settings(parent_document: dict, compression: dict):
    config = load_config()
    
    if "models" not in config:
        config["models"] = {}
    
    if "vectorstore" not in config["models"]:
        config["models"]["vectorstore"] = {}
    
    config["models"]["vectorstore"]["parent_document"] = parent_document
    config["models"]["vectorstore"]["compression"] = compression
    
    save_config(config)
