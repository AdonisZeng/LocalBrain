from pathlib import Path
from typing import Optional
from functools import lru_cache
import yaml
from pydantic import BaseModel


class AppConfig(BaseModel):
    name: str = "LocalBrain"
    version: str = "1.0.0"
    data_dir: str = "~/LocalBrain_Data"
    watch_directory: str = "~/LocalBrain_Data/documents"
    environment: str = "development"


class RunModeConfig(BaseModel):
    mode: str = "local"
    allowed_ips: list = []
    allow_remote: bool = False


class EmbeddingProviderConfig(BaseModel):
    model_name: str = ""
    base_url: str = "http://localhost:11434"
    dimension: int = 768
    api_key: Optional[str] = None
    device: str = "cpu"
    cache_folder: Optional[str] = None


class LLMProviderConfig(BaseModel):
    model_name: str = ""
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 4096
    context_window: int = 8192
    api_key: Optional[str] = None


class EmbeddingConfig(BaseModel):
    provider: str = ""
    providers: dict = {}


class LLMConfig(BaseModel):
    provider: str = ""
    providers: dict = {}


class VectorStoreConfig(BaseModel):
    provider: str = "chroma"
    persist_directory: str = "./data/chroma_db"
    parent_document: dict = {"enabled": True, "parent_chunk_size": 2000, "child_chunk_size": 400}
    compression: dict = {"enabled": True, "threshold": 0.5, "max_context_chars": 4000}


class KeywordSearchConfig(BaseModel):
    provider: str = "bm25"


class DocumentProcessingConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50
    supported_formats: list = ["md", "txt", "pdf"]
    splitting_strategies: dict = {}


class HybridSearchConfig(BaseModel):
    rrf_k: int = 60
    reranking: dict = {"enabled": True, "model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "top_n": 10}
    weights: dict = {"semantic": 0.5, "keyword": 0.5}


class EvaluationConfig(BaseModel):
    enabled: bool = True
    relevance_threshold: float = 0.3
    ragas: dict = {"metrics": ["faithfulness", "answer_relevance", "context_precision", "context_recall"], "sample_size": 100}


class ModelsConfig(BaseModel):
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    vectorstore: VectorStoreConfig = VectorStoreConfig()
    keyword_search: KeywordSearchConfig = KeywordSearchConfig()
    document_processing: DocumentProcessingConfig = DocumentProcessingConfig()
    hybrid_search: HybridSearchConfig = HybridSearchConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


class SecurityConfig(BaseModel):
    auth_enabled: bool = False
    api_key_header: str = "X-API-Key"
    rate_limit: dict = {"enabled": True, "requests_per_minute": 60, "lan_requests_per_minute": 120}


class CacheConfig(BaseModel):
    enabled: bool = True
    type: str = "memory"
    ttl_seconds: int = 3600
    max_size_mb: int = 512


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    output: list = ["console", "file"]
    file_path: str = "./logs/localbrain.log"
    rotation: dict = {"max_bytes": 10485760, "backup_count": 5}


class BackupConfig(BaseModel):
    enabled: bool = True
    schedule: str = "0 2 * * *"
    retention_days: int = 30
    destinations: list = [{"type": "local", "path": "./backups"}]


class ImportExportConfig(BaseModel):
    export_formats: list = ["obsidian", "json", "markdown"]
    obsidian: dict = {"enabled": True, "include_metadata": True, "link_format": "wikilink", "frontmatter_format": "yaml"}
    import_support: dict = {"support_obsidian": True, "support_notion": True, "support_logseq": True}


class Config(BaseModel):
    app: AppConfig = AppConfig()
    run_mode: RunModeConfig = RunModeConfig()
    models: ModelsConfig = ModelsConfig()
    security: SecurityConfig = SecurityConfig()
    cache: CacheConfig = CacheConfig()
    logging: LoggingConfig = LoggingConfig()
    backup: BackupConfig = BackupConfig()
    import_export: ImportExportConfig = ImportExportConfig()


def load_yaml_config(config_path: str) -> dict:
    path = Path(config_path)
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def resolve_env_vars(config: dict) -> dict:
    import os
    import re
    
    def _resolve(value):
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            var_name = value[2:-1]
            return os.environ.get(var_name, value)
        elif isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_resolve(item) for item in value]
        return value
    
    return _resolve(config)


@lru_cache()
def get_config(config_path: str = None) -> Config:
    if config_path is None:
        config_path = get_project_root() / "config.yaml"
    else:
        config_path = Path(config_path)
    config_dict = load_yaml_config(str(config_path))
    config_dict = resolve_env_vars(config_dict)
    return Config(**config_dict)


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent


def get_data_dir() -> Path:
    config = get_config()
    data_dir_path = Path(config.app.data_dir)
    if data_dir_path.is_absolute():
        data_dir = data_dir_path.expanduser()
    else:
        data_dir = get_project_root() / data_dir_path
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_documents_dir() -> Path:
    docs_dir = get_data_dir() / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    return docs_dir
