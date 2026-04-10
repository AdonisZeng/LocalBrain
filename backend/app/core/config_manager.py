from pathlib import Path
from typing import Optional
import yaml
import shutil
from datetime import datetime

from app.core.paths import get_config_path, get_config_backups_dir, get_app_dir, is_frozen


def _backup_config():
    config_file = get_config_path()
    backup_dir = get_config_backups_dir()
    backup_dir.mkdir(parents=True, exist_ok=True)
    if config_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"config_{timestamp}.yaml"
        shutil.copy2(config_file, backup_path)


def load_config() -> dict:
    config_file = get_config_path()

    # First run: copy config from _internal to exe directory
    if not config_file.exists() and is_frozen():
        internal_config = get_app_dir() / "_internal" / "config.yaml"
        if internal_config.exists():
            import shutil
            shutil.copy2(internal_config, config_file)

    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(config_dict: dict):
    _backup_config()
    config_file = get_config_path()
    with open(config_file, 'w', encoding='utf-8') as f:
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
