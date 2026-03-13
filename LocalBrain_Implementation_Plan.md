# LocalBrain 个人本地知识库系统实施方案（增强版 V3）

## 一、项目概述

### 1.1 项目定位与目标

LocalBrain 是一个专注于隐私保护的本地优先知识管理系统，采用 Python 后端与网页前端相结合的双层架构。该系统的核心设计理念是将用户本地的各类文档（Markdown、PDF、TXT 等）进行统一管理和智能索引，同时支持传统的关键词搜索和基于向量的语义搜索两种检索模式。项目旨在为研究者、开发者和写作者提供一个离线可用、数据自主可控、智能检索能力强大的个人知识库解决方案。

本系统的核心特性体现在四个维度。首先是本地优先原则，所有数据存储在用户本地设备上，无需依赖云服务，确保数据隐私和安全。其次是混合搜索能力，结合关键词精确匹配和语义相似度检索，即使查询语句与文档内容表述不同，也能找到相关文档。第三是双向链接支持，借鉴 Wiki 系统的链接机制，支持文档间的相互引用和关系网络可视化。第四是高度可配置的模型支持，原生支持 Ollama 和 LMStudio 等本地大模型服务，用户可根据硬件条件灵活选择和切换。

### 1.2 技术选型概览

| 技术层级 | 技术选型 | 版本建议 | 说明 |
|----------|----------|----------|------|
| 后端框架 | FastAPI | 0.109+ | 高性能异步框架，原生支持异步操作 |
| RAG 框架 | LangChain | 0.2+ | 完整的 RAG 流程封装 |
| 工作流编排 | LangGraph | 0.1+ | 复杂对话流程支持 |
| 前端框架 | React + Vite | React 18+ | 快速构建和热更新 |
| UI 组件 | Tailwind + Shadcn/UI | 最新稳定版 | 现代化响应式界面 |
| 关系数据库 | SQLite | - | 嵌入式，无需独立服务器 |
| 向量数据库 | ChromaDB | 0.4+ | 本地向量存储和检索 |
| 嵌入模型 | Sentence-Transformers | all-MiniLM-L6-v2 | 轻量级高效的嵌入向量生成 |
| 本地大模型 | Ollama / LMStudio | latest | 本地 LLM 运行时 |
| 测试框架 | pytest + Playwright | 最新版 | 单元测试、集成测试、端到端测试 |
| 日志系统 | structlog | 最新版 | 结构化日志输出 |
| 数据库迁移 | Alembic | 最新版 | 数据库版本管理 |
| RAG 评估 | RAGAS | 最新版 | 检索质量评估框架 |

---

## 二、系统架构设计

### 2.1 整体架构

系统采用分层客户端-服务器架构，专门针对本地部署场景进行优化。整体架构分为五个层次：前端展示层负责用户界面交互和结果呈现；后端 API 层处理请求路由、文件操作和业务逻辑编排；RAG 管道层集成 LangChain 实现文档处理和向量检索；可选的 LLM 问答层使用 LangGraph 编排复杂的多轮对话流程；存储层则负责原始文件、元数据和向量数据的持久化管理。

```
┌─────────────────────────────────────────────────────────────────────┐
│                         前端展示层 (React)                          │
│  ┌─────────────┐  ┌─────────────────────┐  ┌──────────────────┐   │
│  │  左侧边栏    │  │     主内容区         │  │   右侧边栏       │   │
│  │  导航树     │  │  文档查看器/搜索结果  │  │   元数据/目录   │   │
│  │  标签云     │  │  问答对话界面         │  │   链接提及      │   │
│  │  快速添加   │  │                     │  │                  │   │
│  └─────────────┘  └─────────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       后端 API 层 (FastAPI)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  │
│  │  文档管理   │  │   搜索API    │  │   标签API   │  │ 问答API  │  │
│  │  CRUD操作   │  │  混合搜索    │  │   层级标签   │  │ RAG问答  │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    中间件层                                   │   │
│  │  认证中间件  │  异常处理中间件  │  日志中间件  │  缓存中间件  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RAG 管道层 (LangChain + LangGraph)               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    文档处理管道                                 │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐     │  │
│  │  │  文档加载器  │→│  智能分割器  │→│   向量嵌入模型      │     │  │
│  │  │ (多格式支持) │  │ (策略矩阵)   │  │                   │     │  │
│  │  └────────────┘  └────────────┘  └────────────────────┘     │  │
│  │         │                                    │               │  │
│  │         ▼                                    ▼               │  │
│  │  ┌───────────────────────────────────────────────────────┐   │  │
│  │  │              ChromaDB 向量存储                         │   │  │
│  │  │         (小块向量 + 父文档索引)                        │   │  │
│  │  └───────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                    │                                │
│                                    ▼                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    检索器 (Retriever)                         │  │
│  │ 语义搜索 │ 关键词搜索 │ BM25搜索 │ RRF混合搜索 │ Reranking   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                    │                                │
│                                    ▼                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    检索质量评估 (RAGAS)                       │  │
│  │         相关性评估 │ 忠实度评估 │ 答案相关性                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                    │                                │
│                                    ▼                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                 LangGraph 工作流 (可选)                        │  │
│  │           问答链  │  引用溯源  │  多轮对话                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          存储层                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │   SQLite        │  │   ChromaDB      │  │   文件系统        │  │
│  │   元数据存储     │  │   向量存储       │  │   原始文档        │  │
│  │   - 文档表      │  │   - 文档块       │  │   - Markdown     │  │
│  │   - 标签表      │  │   - 嵌入向量     │  │   - PDF          │  │
│  │   - 链接表      │  │   - 元数据       │  │   - TXT          │  │
│  │   - 审计日志表   │  │   - 父文档索引   │  │   - 导入导出     │  │
│  └─────────────────┘  └─────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 技术架构分层说明

前端展示层采用 React + Vite 构建单页应用，使用 Tailwind CSS 实现样式，Shadcn/UI 提供基础组件。界面采用经典的三栏布局，左侧为导航和文件树，中间为文档查看器或搜索结果，右侧显示元数据和目录信息。搜索功能通过全局命令面板（Ctrl/Cmd + K）唤起，支持实时搜索建议和结果预览。前端支持深色/浅色主题切换和多语言国际化。

后端 API 层基于 FastAPI 构建，提供完整的 RESTful 接口。文档管理接口支持文档的创建、读取、更新、删除和搜索操作。标签管理接口支持层级标签的 CRUD 和文档关联。问答接口集成 RAG 管道，支持自然语言提问并返回带有引用来源的答案。所有接口均支持异步处理，能够高效应对大量并发请求。中间件层提供认证、异常处理、日志记录和缓存等功能。

RAG 管道层是系统的核心智能层，完全基于 LangChain 构建。文档处理管道支持多种格式的文档加载和标准化处理。文本分割采用智能分割策略矩阵，根据不同文档类型选择最优分割方式。向量嵌入支持多种模型提供商，用户可在配置中自由切换。检索器支持语义搜索、关键词搜索、RRF混合搜索和Reranking重排序，可根据场景选择最优策略。

---

## 三、模型配置系统

### 3.1 配置架构设计

系统采用配置驱动的架构，通过 YAML 配置文件定义模型参数，支持在不修改代码的情况下切换不同的模型提供商。这种设计使得用户可以根据自己的硬件条件和偏好，灵活选择和组合不同的嵌入模型和大语言模型。

配置系统采用层级结构设计，顶层定义当前使用的模型提供商，providers 节点下定义各提供商的详细配置。系统内置环境变量引用机制，允许在配置中使用 `${VAR_NAME}` 格式引用系统环境变量，这对于存储敏感的 API Key 特别有用，避免敏感信息明文存储在配置文件中。

```yaml
# ============================================================
# LocalBrain 配置文件
# ============================================================
# 
# 首次使用说明：
# 1. 请在 models.embedding 和 models.llm 中添加您的模型配置
# 2. 至少配置一个嵌入模型（embedding）才能使用搜索功能
# 3. 配置LLM模型后可使用问答功能
# 4. 支持的提供商：ollama, lmstudio, huggingface, openai, anthropic
#
# ============================================================

app:
  name: "LocalBrain"
  version: "1.0.0"
  data_dir: "~/LocalBrain_Data"
  watch_directory: "~/LocalBrain_Data/documents"
  environment: "development"  # development, testing, production

# 运行模式配置
run_mode:
  # local: 本地单用户模式（简化认证，适合个人使用）
  # lan: 局域网共享模式（启用完整认证，适合团队共享）
  mode: "local"
  allowed_ips: []        # 局域网模式下允许的IP范围
  allow_remote: false    # 是否允许远程访问

# ============================================================
# 模型配置（请根据需要添加）
# ============================================================
# 
# 示例配置格式：
# 
# embedding:
#   provider: "ollama"           # 当前使用的提供商
#   providers:
#     ollama:                    # Ollama 配置
#       model_name: "nomic-embed-text"
#       base_url: "http://localhost:11434"
#       dimension: 768
#
# llm:
#   provider: "ollama"
#   providers:
#     ollama:
#       model_name: "llama3:8b"
#       base_url: "http://localhost:11434"
#       temperature: 0.7
#       max_tokens: 4096
#
# 详细配置示例见下方"模型配置示例"章节
# ============================================================

models:
  # -------------------- 嵌入模型配置 --------------------
  # 用于文档向量化和语义搜索
  # 必须配置至少一个嵌入模型
  embedding:
    provider: ""          # 请填写：ollama, lmstudio, huggingface, openai
    providers: {}         # 请添加您的模型配置

  # -------------------- LLM 模型配置 --------------------
  # 用于问答功能
  # 如不需要问答功能，可暂时留空
  llm:
    provider: ""          # 请填写：ollama, lmstudio, openai, anthropic
    providers: {}         # 请添加您的模型配置

  # -------------------- 向量数据库配置 --------------------
  vectorstore:
    provider: "chroma"
    persist_directory: "./data/chroma_db"
    # 父文档检索器配置
    parent_document:
      enabled: true
      parent_chunk_size: 2000
      child_chunk_size: 400

  # -------------------- 关键词搜索配置 --------------------
  keyword_search:
    provider: "bm25"      # 可选：bm25, whoosh, fts5

  # -------------------- 文档处理配置 --------------------
  document_processing:
    chunk_size: 500
    chunk_overlap: 50
    supported_formats:
      - "md"
      - "txt"
      - "pdf"
    # 文本分割策略矩阵
    splitting_strategies:
      markdown:
        strategy: "markdown_header"
        chunk_size: 500
        chunk_overlap: 50
        strip_headers: false
      pdf:
        strategy: "semantic"
        chunk_size: 800
        chunk_overlap: 100
        respect_page_boundaries: true
      code:
        strategy: "ast_aware"
        chunk_size: 1000
        chunk_overlap: 100
        language: "auto"
      txt:
        strategy: "recursive"
        chunk_size: 500
        chunk_overlap: 50

  # -------------------- 混合搜索配置 --------------------
  hybrid_search:
    rrf_k: 60
    reranking:
      enabled: true
      model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
      top_n: 10
    weights:
      semantic: 0.5
      keyword: 0.5

  # -------------------- 检索质量评估配置 --------------------
  evaluation:
    enabled: true
    relevance_threshold: 0.3
    ragas:
      metrics:
        - "faithfulness"
        - "answer_relevance"
        - "context_precision"
        - "context_recall"
      sample_size: 100

# -------------------- 安全配置 --------------------
security:
  auth_enabled: false    # local模式下自动禁用
  api_key_header: "X-API-Key"
  rate_limit:
    enabled: true
    requests_per_minute: 60
    lan_requests_per_minute: 120

# -------------------- 缓存配置 --------------------
cache:
  enabled: true
  type: "memory"         # 可选：memory, file, redis
  ttl_seconds: 3600
  max_size_mb: 512

# -------------------- 日志配置 --------------------
logging:
  level: "INFO"
  format: "json"
  output:
    - "console"
    - "file"
  file_path: "./logs/localbrain.log"
  rotation:
    max_bytes: 10485760  # 10MB
    backup_count: 5

# -------------------- 备份配置 --------------------
backup:
  enabled: true
  schedule: "0 2 * * *"  # 每天凌晨2点
  retention_days: 30
  destinations:
    - type: "local"
      path: "./backups"

# -------------------- 导入导出配置 --------------------
import_export:
  export_formats:
    - "obsidian"
    - "json"
    - "markdown"
  obsidian:
    enabled: true
    include_metadata: true
    link_format: "wikilink"
    frontmatter_format: "yaml"
  import:
    support_obsidian: true
    support_notion: true
    support_logseq: true
```

---

### 模型配置示例

以下是各提供商的完整配置示例，请根据需要复制到 `models` 配置中：

#### 嵌入模型（Embedding）配置示例

```yaml
# ------------------- 示例1：Ollama 嵌入模型 -------------------
embedding:
  provider: "ollama"
  providers:
    ollama:
      model_name: "nomic-embed-text"    # 推荐：nomic-embed-text, mxbai-embed-large
      base_url: "http://localhost:11434"
      dimension: 768

# ------------------- 示例2：LMStudio 嵌入模型 -------------------
embedding:
  provider: "lmstudio"
  providers:
    lmstudio:
      model_name: "text-embedding-3-small"
      base_url: "http://localhost:1234/v1"
      api_key: "not-required"
      dimension: 1536

# ------------------- 示例3：HuggingFace 本地模型 -------------------
embedding:
  provider: "huggingface"
  providers:
    huggingface:
      model_name: "sentence-transformers/all-MiniLM-L6-v2"
      device: "cpu"                      # 可选：cpu, cuda
      cache_folder: "~/.cache/huggingface"
      dimension: 384

# ------------------- 示例4：OpenAI 云端模型 -------------------
embedding:
  provider: "openai"
  providers:
    openai:
      model_name: "text-embedding-3-small"
      api_key: "${OPENAI_API_KEY}"       # 建议使用环境变量
      base_url: "https://api.openai.com/v1"
      dimension: 1536
```

#### LLM 模型配置示例

```yaml
# ------------------- 示例1：Ollama LLM -------------------
llm:
  provider: "ollama"
  providers:
    ollama:
      model_name: "llama3:8b"            # 推荐：llama3:8b, mistral:7b, qwen2:7b
      base_url: "http://localhost:11434"
      temperature: 0.7
      max_tokens: 4096
      context_window: 8192

# ------------------- 示例2：LMStudio LLM -------------------
llm:
  provider: "lmstudio"
  providers:
    lmstudio:
      model_name: "llama3-8b-instruct"
      base_url: "http://localhost:1234/v1"
      api_key: "not-required"
      temperature: 0.7
      max_tokens: 4096

# ------------------- 示例3：OpenAI GPT -------------------
llm:
  provider: "openai"
  providers:
    openai:
      model_name: "gpt-4o"               # 可选：gpt-4o, gpt-4-turbo, gpt-3.5-turbo
      api_key: "${OPENAI_API_KEY}"
      temperature: 0.7
      max_tokens: 4096

# ------------------- 示例4：Anthropic Claude -------------------
llm:
  provider: "anthropic"
  providers:
    anthropic:
      model_name: "claude-3-opus-20240229"  # 可选：claude-3-opus, claude-3-sonnet
      api_key: "${ANTHROPIC_API_KEY}"
      temperature: 0.7
      max_tokens: 4096
```

#### 完整配置示例（Ollama本地模型）

```yaml
models:
  embedding:
    provider: "ollama"
    providers:
      ollama:
        model_name: "nomic-embed-text"
        base_url: "http://localhost:11434"
        dimension: 768
  
  llm:
    provider: "ollama"
    providers:
      ollama:
        model_name: "llama3:8b"
        base_url: "http://localhost:11434"
        temperature: 0.7
        max_tokens: 4096
  
  vectorstore:
    provider: "chroma"
    persist_directory: "./data/chroma_db"
  
  # ... 其他配置保持默认
```

### 3.2 模型抽象层实现

为了支持灵活的模型切换，系统采用工厂模式和抽象接口的设计。定义抽象基类规范模型接口，不同的模型提供商实现具体的逻辑。这种设计使得新增模型支持时无需修改现有代码，只需添加新的实现类即可。

嵌入模型抽象基类定义了 embed_documents 和 embed_query 两个核心方法，分别用于批量文档向量化和单个查询语句向量化。get_dimension 方法返回向量维度，用于向量数据库初始化。大语言模型抽象基类定义了 generate 方法用于文本生成，get_model_name 方法返回当前使用的模型名称。

```python
# models/base.py - 抽象基类定义
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class EmbeddingModel(ABC):
    """嵌入模型抽象基类，定义向量化的标准接口"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将文档列表转换为向量列表

        Args:
            texts: 待向量化的文本列表

        Returns:
            向量列表，每个向量为浮点数列表
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """将查询语句转换为向量

        Args:
            text: 查询文本

        Returns:
            向量，浮点数列表
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """返回向量的维度

        Returns:
            向量维度整数
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """返回模型名称

        Returns:
            模型名称字符串
        """
        pass

class LLMModel(ABC):
    """大语言模型抽象基类，定义文本生成的标准接口"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """根据提示词生成文本响应

        Args:
            prompt: 提示词
            **kwargs: 其他生成参数如 temperature, max_tokens 等

        Returns:
            生成的文本内容
        """
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs):
        """流式生成文本响应（支持WebSocket实时输出）

        Args:
            prompt: 提示词
            **kwargs: 其他生成参数

        Yields:
            生成的文本片段
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """返回模型名称

        Returns:
            模型名称字符串
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查模型服务是否可用

        Returns:
            可用返回 True，否则返回 False
        """
        pass
```

---

## 四、RAG 管道实现（增强版）

### 4.1 文本分割策略矩阵（新增核心改进）

系统的 RAG 管道完全基于 LangChain 构建，提供了从文档加载到问答检索的完整流程。针对不同文档类型采用差异化的分割策略，这是RAG系统质量的关键因素之一。

#### 分割策略矩阵设计

| 文档类型 | 推荐分割策略 | 说明 | 配置参数 |
|----------|--------------|------|----------|
| Markdown | MarkdownHeaderTextSplitter | 按标题层级分割，保留文档结构 | chunk_size=500, strip_headers=false |
| PDF | 语义分割 + 页面边界 | 按页 + 段落语义分割，尊重页面边界 | chunk_size=800, respect_page_boundaries=true |
| 代码文件 | AST感知分割 | 按函数/类进行语法感知分割 | chunk_size=1000, language=auto |
| TXT | RecursiveCharacterSplitter | 递归字符分割，通用处理 | chunk_size=500, chunk_overlap=50 |

```python
# rag/text_splitter.py - 智能文本分割器
from typing import List, Dict, Any, Optional, Literal
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
    PythonCodeTextSplitter,
)
import structlog

logger = structlog.get_logger()

class SplittingStrategy:
    """分割策略枚举"""
    MARKDOWN_HEADER = "markdown_header"
    SEMANTIC = "semantic"
    AST_AWARE = "ast_aware"
    RECURSIVE = "recursive"


class SmartTextSplitter:
    """智能文本分割器
    
    根据文档类型自动选择最优分割策略，
    实现差异化的文档处理。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化智能分割器
        
        Args:
            config: 分割策略配置
        """
        self._config = config
        self._splitters = {}
        self._initialize_splitters()
    
    def _initialize_splitters(self):
        """初始化各类型分割器"""
        strategies = self._config.get("splitting_strategies", {})
        
        # Markdown分割器 - 按标题层级分割
        if "markdown" in strategies:
            md_config = strategies["markdown"]
            self._splitters["md"] = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "header1"),
                    ("##", "header2"),
                    ("###", "header3"),
                    ("####", "header4"),
                ],
                strip_headers=md_config.get("strip_headers", False),
            )
            # 二次分割器，用于处理过大的块
            self._splitters["md_secondary"] = RecursiveCharacterTextSplitter(
                chunk_size=md_config.get("chunk_size", 500),
                chunk_overlap=md_config.get("chunk_overlap", 50),
                separators=["\n\n", "\n", " ", ""]
            )
        
        # PDF分割器 - 语义分割
        if "pdf" in strategies:
            pdf_config = strategies["pdf"]
            self._splitters["pdf"] = RecursiveCharacterTextSplitter(
                chunk_size=pdf_config.get("chunk_size", 800),
                chunk_overlap=pdf_config.get("chunk_overlap", 100),
                separators=["\n\n\n", "\n\n", "\n", " ", ""]
            )
        
        # 代码分割器 - AST感知
        if "code" in strategies:
            code_config = strategies["code"]
            self._splitters["py"] = PythonCodeTextSplitter(
                chunk_size=code_config.get("chunk_size", 1000),
                chunk_overlap=code_config.get("chunk_overlap", 100),
            )
            # 其他语言的通用分割器
            self._splitters["code_generic"] = RecursiveCharacterTextSplitter(
                chunk_size=code_config.get("chunk_size", 1000),
                chunk_overlap=code_config.get("chunk_overlap", 100),
                separators=["\nclass ", "\ndef ", "\nfunction ", "\n\n", "\n", " ", ""]
            )
        
        # TXT分割器 - 递归分割
        if "txt" in strategies:
            txt_config = strategies["txt"]
            self._splitters["txt"] = RecursiveCharacterTextSplitter(
                chunk_size=txt_config.get("chunk_size", 500),
                chunk_overlap=txt_config.get("chunk_overlap", 50),
                separators=["\n\n", "\n", " ", ""]
            )
        
        # 默认分割器
        default_config = strategies.get("txt", {})
        self._splitters["default"] = RecursiveCharacterTextSplitter(
            chunk_size=default_config.get("chunk_size", 500),
            chunk_overlap=default_config.get("chunk_overlap", 50),
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info("分割器初始化完成", splitters=list(self._splitters.keys()))
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """根据文档类型智能分割
        
        Args:
            documents: 文档列表
            
        Returns:
            分割后的文档块列表
        """
        all_chunks = []
        
        for doc in documents:
            file_type = doc.metadata.get("file_type", "txt")
            chunks = self._split_by_type(doc, file_type)
            all_chunks.extend(chunks)
        
        logger.info("文档分割完成", 
                   input_docs=len(documents), 
                   output_chunks=len(all_chunks))
        return all_chunks
    
    def _split_by_type(self, document: Document, file_type: str) -> List[Document]:
        """根据文件类型选择分割策略
        
        Args:
            document: 待分割文档
            file_type: 文件类型
            
        Returns:
            分割后的文档块列表
        """
        if file_type in ["md", "markdown"]:
            return self._split_markdown(document)
        elif file_type == "pdf":
            return self._split_pdf(document)
        elif file_type in ["py", "js", "ts", "java", "go"]:
            return self._split_code(document, file_type)
        else:
            return self._split_text(document)
    
    def _split_markdown(self, document: Document) -> List[Document]:
        """Markdown文档分割
        
        使用MarkdownHeaderTextSplitter按标题层级分割，
        保留文档结构信息。
        """
        try:
            # 首先按标题分割
            header_splits = self._splitters["md"].split_text(document.page_content)
            
            chunks = []
            for split in header_splits:
                # 如果块过大，进行二次分割
                if len(split.page_content) > 1000:
                    secondary_splits = self._splitters["md_secondary"].split_documents([split])
                    chunks.extend(secondary_splits)
                else:
                    chunks.append(split)
            
            # 保留原始元数据
            for chunk in chunks:
                chunk.metadata.update(document.metadata)
            
            return chunks
        except Exception as e:
            logger.warning("Markdown分割失败，使用默认分割", error=str(e))
            return self._splitters["default"].split_documents([document])
    
    def _split_pdf(self, document: Document) -> List[Document]:
        """PDF文档分割
        
        语义分割，尊重页面边界。
        """
        chunks = self._splitters["pdf"].split_documents([document])
        
        # 为每个块添加页面信息
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata.update(document.metadata)
        
        return chunks
    
    def _split_code(self, document: Document, language: str) -> List[Document]:
        """代码文件分割
        
        AST感知分割，按函数/类分割。
        """
        if language == "py":
            chunks = self._splitters["py"].split_documents([document])
        else:
            chunks = self._splitters["code_generic"].split_documents([document])
        
        for chunk in chunks:
            chunk.metadata["language"] = language
            chunk.metadata.update(document.metadata)
        
        return chunks
    
    def _split_text(self, document: Document) -> List[Document]:
        """普通文本分割"""
        chunks = self._splitters.get("txt", self._splitters["default"]).split_documents([document])
        for chunk in chunks:
            chunk.metadata.update(document.metadata)
        return chunks
```

### 4.2 Parent Document Retriever（新增核心改进）

Parent Document Retriever 解决了"检索精确但上下文丢失"的经典问题。检索时匹配小块以获得精确的相关性，返回时扩展到父级大块以提供完整上下文。

```python
# rag/parent_document_retriever.py - 父文档检索器
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain.retrievers import ParentDocumentRetriever as LC_ParentDocumentRetriever
from langchain.storage import InMemoryStore, LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import structlog

logger = structlog.get_logger()


class ParentDocumentRetriever:
    """父文档检索器
    
    核心思想：检索小块，返回大块
    - 小块用于精确匹配，提高检索精度
    - 大块用于提供上下文，保持语义完整性
    
    适用场景：
    - 长文档检索
    - 需要完整上下文的问答
    - 技术文档查询
    """
    
    def __init__(
        self,
        vectorstore,
        config: Dict[str, Any],
        docstore_path: Optional[str] = None
    ):
        """初始化父文档检索器
        
        Args:
            vectorstore: 向量存储（存储小块向量）
            config: 配置参数
            docstore_path: 文档存储路径（可选）
        """
        self._vectorstore = vectorstore
        self._config = config
        
        # 父文档配置
        parent_config = config.get("parent_document", {})
        self._parent_chunk_size = parent_config.get("parent_chunk_size", 2000)
        self._child_chunk_size = parent_config.get("child_chunk_size", 400)
        self._enabled = parent_config.get("enabled", True)
        
        # 父文档分割器
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._parent_chunk_size,
            chunk_overlap=200,
            separators=["\n\n\n", "\n\n", "\n", " ", ""]
        )
        
        # 子文档分割器
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._child_chunk_size,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # 父文档存储
        if docstore_path:
            self._docstore = LocalFileStore(docstore_path)
        else:
            self._docstore = InMemoryStore()
        
        # 父子关系映射
        self._parent_child_map: Dict[str, List[str]] = {}
        self._child_parent_map: Dict[str, str] = {}
        
        logger.info(
            "父文档检索器初始化完成",
            parent_chunk_size=self._parent_chunk_size,
            child_chunk_size=self._child_chunk_size,
            enabled=self._enabled
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档，建立父子层级关系
        
        Args:
            documents: 文档列表
            
        Returns:
            添加的文档ID列表
        """
        if not self._enabled:
            # 禁用时直接添加到向量存储
            return self._vectorstore.add_documents(documents)
        
        all_child_docs = []
        all_parent_ids = []
        
        for doc in documents:
            # 分割为父文档块
            parent_chunks = self._parent_splitter.split_documents([doc])
            
            for parent_idx, parent_chunk in enumerate(parent_chunks):
                # 生成父文档ID
                parent_id = f"{doc.metadata.get('doc_id', 'unknown')}_{parent_idx}"
                parent_chunk.metadata["parent_id"] = parent_id
                parent_chunk.metadata["is_parent"] = True
                
                # 存储父文档
                self._docstore.mset([(parent_id, parent_chunk)])
                all_parent_ids.append(parent_id)
                
                # 分割为子文档块
                child_chunks = self._child_splitter.split_documents([parent_chunk])
                
                child_ids = []
                for child_idx, child_chunk in enumerate(child_chunks):
                    # 生成子文档ID
                    child_id = f"{parent_id}_{child_idx}"
                    child_chunk.metadata["child_id"] = child_id
                    child_chunk.metadata["parent_id"] = parent_id
                    child_chunk.metadata["is_parent"] = False
                    
                    child_ids.append(child_id)
                    all_child_docs.append(child_chunk)
                
                # 建立映射关系
                self._parent_child_map[parent_id] = child_ids
                for child_id in child_ids:
                    self._child_parent_map[child_id] = parent_id
        
        # 将子文档添加到向量存储
        self._vectorstore.add_documents(all_child_docs)
        
        logger.info(
            "文档添加完成（父子层级）",
            parent_count=len(all_parent_ids),
            child_count=len(all_child_docs)
        )
        
        return all_parent_ids
    
    def retrieve(
        self,
        query: str,
        k: int = 4,
        expand_to_parent: bool = True
    ) -> List[Document]:
        """检索相关文档
        
        Args:
            query: 查询语句
            k: 返回结果数量
            expand_to_parent: 是否扩展到父文档
            
        Returns:
            相关文档列表（父文档或子文档）
        """
        if not self._enabled or not expand_to_parent:
            # 直接检索向量存储
            return self._vectorstore.similarity_search(query, k=k)
        
        # 检索子文档（使用更多候选）
        child_results = self._vectorstore.similarity_search(
            query, 
            k=k * 3  # 获取更多候选
        )
        
        # 获取唯一的父文档ID
        parent_ids = set()
        parent_scores: Dict[str, float] = {}
        
        for child in child_results:
            parent_id = child.metadata.get("parent_id")
            if parent_id and parent_id not in parent_ids:
                parent_ids.add(parent_id)
                # 使用第一个匹配的子文档分数作为父文档的近似分数
                if parent_id not in parent_scores:
                    parent_scores[parent_id] = 0.8  # 默认分数
        
        # 从存储中获取父文档
        parent_docs = []
        for parent_id in list(parent_ids)[:k]:
            parent_doc = self._docstore.mget([parent_id])
            if parent_doc and parent_doc[0]:
                doc = parent_doc[0]
                doc.metadata["retrieval_score"] = parent_scores.get(parent_id, 0.8)
                parent_docs.append(doc)
        
        logger.info(
            "父文档检索完成",
            child_candidates=len(child_results),
            parent_results=len(parent_docs)
        )
        
        return parent_docs
    
    def retrieve_with_children(
        self,
        query: str,
        k: int = 4
    ) -> Tuple[List[Document], Dict[str, List[Document]]]:
        """检索父文档并返回所有子文档
        
        Args:
            query: 查询语句
            k: 父文档数量
            
        Returns:
            (父文档列表, {父文档ID: 子文档列表})
        """
        parent_docs = self.retrieve(query, k=k, expand_to_parent=True)
        
        children_by_parent = {}
        for parent in parent_docs:
            parent_id = parent.metadata.get("parent_id")
            if parent_id and parent_id in self._parent_child_map:
                child_ids = self._parent_child_map[parent_id]
                # 从向量存储获取子文档
                children_by_parent[parent_id] = []
                # 这里可以进一步优化，从docstore或vectorstore获取子文档
        
        return parent_docs, children_by_parent
    
    def get_retriever(self, k: int = 4):
        """获取LangChain兼容的检索器"""
        from langchain.retrievers import VectorStoreRetriever
        
        class ParentDocumentVectorRetriever(VectorStoreRetriever):
            def __init__(self, parent_retriever, k):
                super().__init__(vectorstore=parent_retriever._vectorstore)
                self._parent_retriever = parent_retriever
                self._k = k
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._parent_retriever.retrieve(query, k=self._k)
        
        return ParentDocumentVectorRetriever(self, k)
```

### 4.3 RRF混合搜索与Reranking（增强核心改进）

混合搜索结合了语义搜索和关键词搜索的优势，使用Reciprocal Rank Fusion (RRF)算法合并结果，并支持Reranking重排序以进一步提升检索质量。

```python
# rag/hybrid_search_enhanced.py - 增强版混合搜索
from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document
import numpy as np
import structlog

logger = structlog.get_logger()


class RRFHybridSearch:
    """RRF混合搜索引擎
    
    使用Reciprocal Rank Fusion算法合并语义搜索和关键词搜索结果，
    并支持Reranking重排序。
    
    RRF公式: score(d) = Σ 1/(k + rank(d))
    其中k是平滑参数，通常设为60
    """
    
    def __init__(
        self,
        vectorstore,
        keyword_index=None,
        config: Dict[str, Any] = None
    ):
        """初始化RRF混合搜索
        
        Args:
            vectorstore: 向量存储
            keyword_index: 关键词索引（BM25）
            config: 混合搜索配置
        """
        self._vectorstore = vectorstore
        self._keyword_index = keyword_index
        
        config = config or {}
        hybrid_config = config.get("hybrid_search", {})
        
        # RRF参数
        self._rrf_k = hybrid_config.get("rrf_k", 60)
        
        # 权重配置（用户可配置）
        weights = hybrid_config.get("weights", {})
        self._semantic_weight = weights.get("semantic", 0.5)
        self._keyword_weight = weights.get("keyword", 0.5)
        
        # Reranking配置
        reranking_config = hybrid_config.get("reranking", {})
        self._reranking_enabled = reranking_config.get("enabled", True)
        self._reranking_model = reranking_config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._reranking_top_n = reranking_config.get("top_n", 10)
        
        # Reranker实例（延迟加载）
        self._reranker = None
        
        logger.info(
            "RRF混合搜索初始化",
            rrf_k=self._rrf_k,
            semantic_weight=self._semantic_weight,
            keyword_weight=self._keyword_weight,
            reranking_enabled=self._reranking_enabled
        )
    
    def set_weights(self, semantic_weight: float, keyword_weight: float):
        """设置搜索权重（用户可配置接口）
        
        Args:
            semantic_weight: 语义搜索权重 (0-1)
            keyword_weight: 关键词搜索权重 (0-1)
        """
        total = semantic_weight + keyword_weight
        self._semantic_weight = semantic_weight / total
        self._keyword_weight = keyword_weight / total
        logger.info("搜索权重已更新", semantic=self._semantic_weight, keyword=self._keyword_weight)
    
    def search(
        self,
        query: str,
        k: int = 10,
        semantic_k: int = 20,
        keyword_k: int = 20,
        enable_reranking: Optional[bool] = None
    ) -> List[Tuple[Document, float]]:
        """RRF混合搜索
        
        Args:
            query: 查询语句
            k: 最终返回结果数量
            semantic_k: 语义搜索候选数量
            keyword_k: 关键词搜索候选数量
            enable_reranking: 是否启用重排序（None使用配置值）
            
        Returns:
            排序后的 (Document, score) 列表
        """
        # 语义搜索
        semantic_results = self._semantic_search(query, k=semantic_k)
        
        # 关键词搜索
        keyword_results = self._keyword_search(query, k=keyword_k)
        
        # RRF合并
        merged = self._rrf_merge(semantic_results, keyword_results)
        
        # 排序
        sorted_results = sorted(merged.items(), key=lambda x: x[1][1], reverse=True)
        
        # Reranking重排序
        should_rerank = enable_reranking if enable_reranking is not None else self._reranking_enabled
        if should_rerank and len(sorted_results) > k:
            sorted_results = self._rerank(query, sorted_results, top_n=k)
        
        # 返回top-k
        final_results = []
        for doc_id, (doc, score) in sorted_results[:k]:
            final_results.append((doc, score))
        
        logger.info(
            "RRF混合搜索完成",
            query=query[:50],
            semantic_count=len(semantic_results),
            keyword_count=len(keyword_results),
            final_count=len(final_results)
        )
        
        return final_results
    
    def _semantic_search(self, query: str, k: int) -> List[Tuple[Document, int]]:
        """语义搜索
        
        Args:
            query: 查询语句
            k: 返回数量
            
        Returns:
            [(Document, rank)] 列表
        """
        results = self._vectorstore.similarity_search_with_score(query, k=k)
        
        if not results:
            return []
        
        # 返回带排名的结果
        return [(doc, rank + 1) for rank, (doc, _) in enumerate(results)]
    
    def _keyword_search(self, query: str, k: int) -> List[Tuple[Document, int]]:
        """关键词搜索（BM25）
        
        Args:
            query: 查询语句
            k: 返回数量
            
        Returns:
            [(Document, rank)] 列表
        """
        if not self._keyword_index:
            return []
        
        try:
            results = self._keyword_index.get_top_n(query.split(), n=k)
            
            if not results:
                return []
            
            # 返回带排名的结果
            return [(doc, rank + 1) for rank, (doc, _) in enumerate(results)]
        except Exception as e:
            logger.warning("关键词搜索失败", error=str(e))
            return []
    
    def _rrf_merge(
        self,
        semantic_results: List[Tuple[Document, int]],
        keyword_results: List[Tuple[Document, int]]
    ) -> Dict[str, Tuple[Document, float]]:
        """RRF合并算法
        
        RRF公式: score(d) = Σ 1/(k + rank(d))
        
        Args:
            semantic_results: 语义搜索结果 [(Document, rank)]
            keyword_results: 关键词搜索结果 [(Document, rank)]
            
        Returns:
            {doc_id: (Document, rrf_score)}
        """
        merged = {}
        
        # 处理语义搜索结果
        for doc, rank in semantic_results:
            doc_id = self._get_doc_id(doc)
            rrf_score = self._semantic_weight / (self._rrf_k + rank)
            
            if doc_id in merged:
                merged[doc_id] = (doc, merged[doc_id][1] + rrf_score)
            else:
                merged[doc_id] = (doc, rrf_score)
        
        # 处理关键词搜索结果
        for doc, rank in keyword_results:
            doc_id = self._get_doc_id(doc)
            rrf_score = self._keyword_weight / (self._rrf_k + rank)
            
            if doc_id in merged:
                merged[doc_id] = (merged[doc_id][0], merged[doc_id][1] + rrf_score)
            else:
                merged[doc_id] = (doc, rrf_score)
        
        return merged
    
    def _get_doc_id(self, doc: Document) -> str:
        """获取文档唯一标识"""
        return doc.metadata.get("id", doc.metadata.get("doc_id", doc.page_content[:50]))
    
    def _rerank(
        self,
        query: str,
        results: List[Tuple[str, Tuple[Document, float]]],
        top_n: int
    ) -> List[Tuple[str, Tuple[Document, float]]]:
        """使用Cross-Encoder重排序
        
        Args:
            query: 查询语句
            results: 初步排序结果
            top_n: 返回数量
            
        Returns:
            重排序后的结果
        """
        try:
            # 延迟加载Reranker
            if self._reranker is None:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self._reranking_model)
                logger.info("Reranker加载完成", model=self._reranking_model)
            
            # 准备重排序输入
            pairs = [(query, doc.page_content) for _, (doc, _) in results]
            
            # 获取重排序分数
            scores = self._reranker.predict(pairs)
            
            # 重新排序
            reranked = [
                (doc_id, (doc, float(score)))
                for score, (doc_id, (doc, _)) in zip(scores, results)
            ]
            reranked.sort(key=lambda x: x[1][1], reverse=True)
            
            logger.info("Reranking完成", input_count=len(results), output_count=len(reranked[:top_n]))
            
            return reranked[:top_n]
            
        except Exception as e:
            logger.warning("Reranking失败，使用原始排序", error=str(e))
            return results[:top_n]


class HybridSearchWrapper:
    """混合搜索包装器
    
    提供统一的搜索接口，支持多种搜索模式。
    """
    
    def __init__(
        self,
        vectorstore,
        keyword_index=None,
        config: Dict[str, Any] = None
    ):
        """初始化混合搜索包装器"""
        self._rrf_search = RRFHybridSearch(vectorstore, keyword_index, config)
        self._vectorstore = vectorstore
        self._keyword_index = keyword_index
    
    def search(
        self,
        query: str,
        search_type: str = "hybrid",  # semantic, keyword, hybrid
        k: int = 10,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """统一搜索接口
        
        Args:
            query: 查询语句
            search_type: 搜索类型 (semantic, keyword, hybrid)
            k: 返回数量
            **kwargs: 其他参数
            
        Returns:
            [(Document, score)] 列表
        """
        if search_type == "semantic":
            results = self._vectorstore.similarity_search_with_score(query, k=k)
            return [(doc, 1 - score) for doc, score in results]  # 转换为相似度分数
        
        elif search_type == "keyword":
            if not self._keyword_index:
                logger.warning("关键词索引不可用，回退到语义搜索")
                return self.search(query, search_type="semantic", k=k)
            
            results = self._keyword_index.get_top_n(query.split(), n=k)
            return [(doc, score / 100) for doc, score in results]  # 归一化分数
        
        else:  # hybrid
            return self._rrf_search.search(query, k=k, **kwargs)
```

### 4.4 检索质量评估机制（新增核心改进）

使用RAGAS框架进行检索质量评估，确保RAG系统的输出质量可衡量、可优化。

```python
# rag/evaluation.py - 检索质量评估
from typing import List, Dict, Any, Optional
from langchain.schema import Document
import structlog

logger = structlog.get_logger()


class RetrievalEvaluator:
    """检索质量评估器
    
    使用多种指标评估检索质量：
    1. 相关性分数阈值 - 过滤低相关性结果
    2. RAGAS评估框架 - 全面的RAG系统评估
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化评估器
        
        Args:
            config: 评估配置
        """
        self._config = config
        eval_config = config.get("evaluation", {})
        
        # 相关性阈值
        self._relevance_threshold = eval_config.get("relevance_threshold", 0.3)
        self._enabled = eval_config.get("enabled", True)
        
        # RAGAS配置
        ragas_config = eval_config.get("ragas", {})
        self._ragas_metrics = ragas_config.get("metrics", [
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "context_recall"
        ])
        self._sample_size = ragas_config.get("sample_size", 100)
        
        # RAGAS实例（延迟加载）
        self._ragas_evaluator = None
        
        # 评估历史记录
        self._evaluation_history: List[Dict[str, Any]] = []
        
        logger.info(
            "检索评估器初始化",
            relevance_threshold=self._relevance_threshold,
            ragas_metrics=self._ragas_metrics
        )
    
    def filter_by_relevance(
        self,
        results: List[Tuple[Document, float]],
        threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """基于相关性阈值过滤结果
        
        Args:
            results: 搜索结果 [(Document, score)]
            threshold: 阈值（None使用配置值）
            
        Returns:
            过滤后的结果
        """
        threshold = threshold or self._relevance_threshold
        
        filtered = [
            (doc, score) for doc, score in results
            if score >= threshold
        ]
        
        if len(filtered) < len(results):
            logger.info(
                "相关性过滤完成",
                input_count=len(results),
                output_count=len(filtered),
                threshold=threshold
            )
        
        return filtered
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Document],
        answer: Optional[str] = None,
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """评估单次检索质量
        
        Args:
            query: 用户查询
            retrieved_docs: 检索到的文档
            answer: 生成的答案（可选）
            ground_truth: 真实答案（可选）
            
        Returns:
            评估指标字典
        """
        if not self._enabled:
            return {}
        
        metrics = {}
        
        # 基础指标
        metrics["retrieved_count"] = len(retrieved_docs)
        metrics["avg_doc_length"] = sum(len(doc.page_content) for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
        
        # 检索覆盖率（如果提供了ground_truth）
        if ground_truth:
            coverage = self._calculate_coverage(retrieved_docs, ground_truth)
            metrics["context_coverage"] = coverage
        
        # RAGAS评估（如果提供了答案）
        if answer and self._ragas_metrics:
            ragas_scores = self._evaluate_with_ragas(
                query, retrieved_docs, answer, ground_truth
            )
            metrics.update(ragas_scores)
        
        # 记录评估历史
        self._evaluation_history.append({
            "query": query[:100],  # 截断长查询
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })
        
        return metrics
    
    def _calculate_coverage(
        self,
        retrieved_docs: List[Document],
        ground_truth: str
    ) -> float:
        """计算检索覆盖率
        
        Args:
            retrieved_docs: 检索到的文档
            ground_truth: 真实答案
            
        Returns:
            覆盖率分数 (0-1)
        """
        # 简单的关键词覆盖率计算
        truth_words = set(ground_truth.lower().split())
        truth_words = {w for w in truth_words if len(w) > 2}  # 过滤短词
        
        if not truth_words:
            return 0.0
        
        retrieved_text = " ".join(doc.page_content for doc in retrieved_docs).lower()
        covered = sum(1 for word in truth_words if word in retrieved_text)
        
        return covered / len(truth_words)
    
    def _evaluate_with_ragas(
        self,
        query: str,
        retrieved_docs: List[Document],
        answer: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """使用RAGAS框架评估
        
        Args:
            query: 用户查询
            retrieved_docs: 检索到的文档
            answer: 生成的答案
            ground_truth: 真实答案
            
        Returns:
            RAGAS评估指标
        """
        try:
            # 延迟加载RAGAS
            if self._ragas_evaluator is None:
                from ragas import evaluate
                from ragas.metrics import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                )
                self._ragas_evaluator = evaluate
                self._ragas_metrics_map = {
                    "faithfulness": faithfulness,
                    "answer_relevance": answer_relevancy,
                    "context_precision": context_precision,
                    "context_recall": context_recall
                }
                logger.info("RAGAS评估器加载完成")
            
            # 准备评估数据
            contexts = [doc.page_content for doc in retrieved_docs]
            
            # 构建评估输入
            eval_data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
            }
            
            if ground_truth:
                eval_data["ground_truth"] = [ground_truth]
            
            # 选择要计算的指标
            metrics_to_use = [
                self._ragas_metrics_map[m]
                for m in self._ragas_metrics
                if m in self._ragas_metrics_map
            ]
            
            # 执行评估
            result = self._ragas_evaluator(
                eval_data,
                metrics=metrics_to_use
            )
            
            # 提取分数
            scores = {}
            for metric in self._ragas_metrics:
                if metric in result:
                    scores[f"ragas_{metric}"] = float(result[metric])
            
            logger.info("RAGAS评估完成", scores=scores)
            return scores
            
        except ImportError:
            logger.warning("RAGAS未安装，跳过评估")
            return {}
        except Exception as e:
            logger.warning("RAGAS评估失败", error=str(e))
            return {}
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """获取评估历史摘要
        
        Returns:
            评估摘要统计
        """
        if not self._evaluation_history:
            return {"total_evaluations": 0}
        
        # 计算平均指标
        all_metrics = {}
        for record in self._evaluation_history:
            for key, value in record["metrics"].items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        summary = {
            "total_evaluations": len(self._evaluation_history),
            "average_metrics": {
                key: sum(values) / len(values)
                for key, values in all_metrics.items()
            }
        }
        
        return summary
    
    def get_quality_warning(
        self,
        results: List[Tuple[Document, float]],
        threshold: Optional[float] = None
    ) -> Optional[str]:
        """生成质量警告信息
        
        Args:
            results: 搜索结果
            threshold: 相关性阈值
            
        Returns:
            警告信息（如果需要）
        """
        threshold = threshold or self._relevance_threshold
        
        if not results:
            return "未找到相关内容"
        
        max_score = max(score for _, score in results)
        if max_score < threshold:
            return f"未找到高相关内容（最高相关性: {max_score:.2f}）"
        
        return None


# 集成到搜索流程
class EvaluatedSearchPipeline:
    """带评估的搜索管道"""
    
    def __init__(
        self,
        hybrid_search: RRFHybridSearch,
        evaluator: RetrievalEvaluator
    ):
        self._search = hybrid_search
        self._evaluator = evaluator
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter_low_relevance: bool = True
    ) -> Dict[str, Any]:
        """执行搜索并评估
        
        Args:
            query: 查询语句
            k: 返回数量
            filter_low_relevance: 是否过滤低相关性结果
            
        Returns:
            {
                "results": [(Document, score)],
                "warning": Optional[str],
                "metrics": Dict[str, float]
            }
        """
        # 执行搜索
        results = self._search.search(query, k=k * 2)  # 获取更多候选
        
        # 过滤低相关性结果
        if filter_low_relevance:
            results = self._evaluator.filter_by_relevance(results)
        
        results = results[:k]
        
        # 生成质量警告
        warning = self._evaluator.get_quality_warning(results)
        
        return {
            "results": results,
            "warning": warning,
            "metrics": {
                "total_count": len(results),
                "avg_score": sum(s for _, s in results) / len(results) if results else 0
            }
        }
```

### 4.5 完整RAG管道集成

将以上增强功能集成到完整的RAG管道中：

```python
# rag/pipeline_enhanced.py - 增强版RAG管道
from typing import List, Dict, Any, Optional
from langchain.schema import Document
import structlog

from models.factory import ModelFactory, ConfigManager
from rag.text_splitter import SmartTextSplitter
from rag.parent_document_retriever import ParentDocumentRetriever
from rag.hybrid_search_enhanced import RRFHybridSearch, HybridSearchWrapper
from rag.evaluation import RetrievalEvaluator, EvaluatedSearchPipeline

logger = structlog.get_logger()


class EnhancedRAGPipeline:
    """增强版RAG管道
    
    整合所有改进：
    - 智能文本分割策略矩阵
    - Parent Document Retriever
    - RRF混合搜索 + Reranking
    - 检索质量评估
    """
    
    def __init__(self, config_manager: ConfigManager):
        self._config = config_manager.get_config()
        self._config_manager = config_manager
        
        # 组件实例
        self._embedding_model = None
        self._llm_model = None
        self._vectorstore = None
        self._text_splitter = None
        self._parent_retriever = None
        self._hybrid_search = None
        self._evaluator = None
        self._search_pipeline = None
        
        self._initialize()
    
    def _initialize(self):
        """初始化所有组件"""
        logger.info("初始化增强版RAG管道")
        
        # 初始化嵌入模型
        embedding_config = self._config["models"]["embedding"]
        provider = embedding_config["provider"]
        provider_config = embedding_config["providers"].get(provider, {})
        self._embedding_model = ModelFactory.create_embedding_model(provider, provider_config)
        
        # 初始化智能文本分割器
        doc_config = self._config["models"]["document_processing"]
        self._text_splitter = SmartTextSplitter(doc_config)
        
        # 初始化向量存储
        vs_config = self._config["models"]["vectorstore"]
        self._vectorstore = self._init_vectorstore(vs_config)
        
        # 初始化父文档检索器
        if vs_config.get("parent_document", {}).get("enabled", True):
            self._parent_retriever = ParentDocumentRetriever(
                self._vectorstore,
                vs_config
            )
        
        # 初始化混合搜索
        self._hybrid_search = RRFHybridSearch(
            self._vectorstore,
            keyword_index=None,  # 可后续配置
            config=self._config
        )
        
        # 初始化评估器
        self._evaluator = RetrievalEvaluator(self._config)
        
        # 初始化搜索管道
        self._search_pipeline = EvaluatedSearchPipeline(
            self._hybrid_search,
            self._evaluator
        )
        
        logger.info("增强版RAG管道初始化完成")
    
    def add_documents(self, documents: List[Document]):
        """添加文档到索引
        
        Args:
            documents: 文档列表
        """
        # 智能分割
        chunks = self._text_splitter.split_documents(documents)
        
        # 添加到父文档检索器（如果启用）
        if self._parent_retriever:
            self._parent_retriever.add_documents(chunks)
        else:
            self._vectorstore.add_documents(chunks)
        
        logger.info("文档添加完成", chunks_count=len(chunks))
    
    def search(
        self,
        query: str,
        k: int = 10,
        search_type: str = "hybrid"
    ) -> Dict[str, Any]:
        """搜索
        
        Args:
            query: 查询语句
            k: 返回数量
            search_type: 搜索类型 (semantic, keyword, hybrid)
            
        Returns:
            搜索结果和评估信息
        """
        return self._search_pipeline.search(query, k=k)
    
    def ask(self, question: str) -> Dict[str, Any]:
        """问答
        
        Args:
            question: 用户问题
            
        Returns:
            答案和来源文档
        """
        # 检索相关文档
        search_result = self.search(question, k=4)
        docs = [doc for doc, _ in search_result["results"]]
        
        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 生成答案（需要LLM）
        if not self._llm_model:
            # 初始化LLM
            llm_config = self._config["models"]["llm"]
            provider = llm_config.get("provider", "ollama")
            provider_config = llm_config.get("providers", {}).get(provider, {})
            self._llm_model = ModelFactory.create_llm_model(provider, provider_config)
        
        prompt = f"""基于以下上下文回答问题。如果上下文中没有相关信息，请说明。

上下文：
{context}

问题：{question}

答案："""
        
        answer = self._llm_model.generate(prompt)
        
        # 评估检索质量
        metrics = self._evaluator.evaluate_retrieval(
            query=question,
            retrieved_docs=docs,
            answer=answer
        )
        
        return {
            "answer": answer,
            "source_documents": [
                {
                    "content": doc.page_content[:500],  # 截断长内容
                    "metadata": doc.metadata
                }
                for doc in docs
            ],
            "warning": search_result.get("warning"),
            "metrics": metrics
        }
```

---

## 五、数据模型设计

### 5.1 SQLite 数据库表结构

SQLite 作为元数据存储，保存文档的基本信息、标签、链接关系和审计日志。数据库设计遵循规范化原则，通过关联表实现多对多关系，同时保持查询效率。

```sql
-- 创建数据库表

-- 文档主表：存储文档的基本信息和元数据（增强版）
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT UNIQUE NOT NULL,        -- 文件绝对路径，保证唯一性
    filename TEXT NOT NULL,                -- 文件名（不含路径）
    file_type TEXT NOT NULL,               -- 文件类型：md、txt、pdf
    content_hash TEXT,                     -- 内容哈希值，用于检测变更
    title TEXT,                            -- 文档标题（从Frontmatter提取或文件名）
    summary TEXT,                          -- 文档摘要（可选）
    word_count INTEGER DEFAULT 0,          -- 文档字数
    author TEXT,                           -- 作者（新增）
    language TEXT,                         -- 语言（新增）
    headings TEXT,                         -- 标题结构JSON（新增）
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 标签表：存储用户定义的标签
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,            -- 标签名称，支持层级如 "开发/Python"
    color TEXT DEFAULT '#3b82f6',         -- 标签颜色
    parent_id INTEGER,                     -- 父标签 ID，支持层级标签
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_id) REFERENCES tags(id) ON DELETE SET NULL
);

-- 文档-标签关联表：多对多关系
CREATE TABLE IF NOT EXISTS doc_tags (
    doc_id INTEGER,
    tag_id INTEGER,
    PRIMARY KEY (doc_id, tag_id),
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

-- 文档链接表：存储 WikiLink 形式的双向链接（增强版）
CREATE TABLE IF NOT EXISTS doc_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_doc_id INTEGER,                -- 源文档 ID
    target_doc_id INTEGER,                -- 目标文档 ID
    link_text TEXT,                       -- 链接显示文本
    link_type TEXT DEFAULT 'wikilink',    -- 链接类型：wikilink, markdown, embed（新增）
    target_heading TEXT,                  -- 目标标题（如 [[文档名#标题]]）（新增）
    is_broken BOOLEAN DEFAULT FALSE,      -- 链接是否失效（新增）
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_doc_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (target_doc_id) REFERENCES documents(id) ON DELETE CASCADE,
    CONSTRAINT unique_link UNIQUE (source_doc_id, target_doc_id, link_text)
);

-- 审计日志表：记录关键操作
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,                 -- 操作类型：create, update, delete, search, ask
    entity_type TEXT,                     -- 实体类型：document, tag, config
    entity_id INTEGER,                    -- 实体 ID
    details TEXT,                         -- 操作详情（JSON格式）
    ip_address TEXT,                      -- IP 地址
    user_agent TEXT,                      -- 用户代理
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 配置表：存储用户配置
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 父子文档关系表（新增）
CREATE TABLE IF NOT EXISTS document_hierarchy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_chunk_id TEXT NOT NULL,        -- 父文档块ID
    child_chunk_id TEXT NOT NULL,         -- 子文档块ID
    doc_id INTEGER,                       -- 所属文档ID
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
    CONSTRAINT unique_hierarchy UNIQUE (parent_chunk_id, child_chunk_id)
);

-- 评估历史表（新增）
CREATE TABLE IF NOT EXISTS evaluation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    metrics TEXT,                         -- JSON格式的评估指标
    search_type TEXT,
    result_count INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 导入导出记录表（新增）
CREATE TABLE IF NOT EXISTS import_export_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL,              -- import, export
    format TEXT NOT NULL,                 -- obsidian, json, markdown
    file_count INTEGER,
    status TEXT,                          -- success, partial, failed
    details TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 索引优化查询性能
CREATE INDEX IF NOT EXISTS idx_documents_filepath ON documents(filepath);
CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents(file_type);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_author ON documents(author);
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
CREATE INDEX IF NOT EXISTS idx_tags_parent ON tags(parent_id);
CREATE INDEX IF NOT EXISTS idx_doc_links_source ON doc_links(source_doc_id);
CREATE INDEX IF NOT EXISTS idx_doc_links_target ON doc_links(target_doc_id);
CREATE INDEX IF NOT EXISTS idx_doc_links_broken ON doc_links(is_broken);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);

-- 全文搜索虚拟表（使用 FTS5）
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title,
    content,
    content='documents',
    content_rowid='id'
);

-- 触发器：自动更新全文搜索索引
CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, content) 
    VALUES (new.id, new.title, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, content) 
    VALUES ('delete', old.id, old.title, old.summary);
END;

CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, content) 
    VALUES ('delete', old.id, old.title, old.summary);
    INSERT INTO documents_fts(rowid, title, content) 
    VALUES (new.id, new.title, new.summary);
END;
```

---

## 六、安全性设计（增强版）

### 6.1 运行模式区分（核心改进）

根据用户使用场景，系统支持两种运行模式：本地单用户模式和局域网共享模式。不同模式下的安全策略有显著差异。

```python
# security/run_mode.py - 运行模式管理
from enum import Enum
from typing import Optional, List
import structlog

logger = structlog.get_logger()


class RunMode(Enum):
    """运行模式枚举"""
    LOCAL = "local"       # 本地单用户模式
    LAN = "lan"           # 局域网共享模式


class RunModeManager:
    """运行模式管理器
    
    根据运行模式自动调整安全策略：
    - LOCAL模式：简化认证，优化个人使用体验
    - LAN模式：完整认证，保护多用户访问安全
    """
    
    def __init__(self, config: dict):
        """初始化运行模式管理器
        
        Args:
            config: 运行模式配置
        """
        mode_str = config.get("mode", "local")
        self._mode = RunMode(mode_str)
        
        # 局域网模式配置
        self._allowed_ips = config.get("allowed_ips", [])
        self._allow_remote = config.get("allow_remote", False)
        
        logger.info(
            "运行模式初始化",
            mode=self._mode.value,
            allow_remote=self._allow_remote
        )
    
    @property
    def mode(self) -> RunMode:
        """获取当前运行模式"""
        return self._mode
    
    def is_local_mode(self) -> bool:
        """是否为本地模式"""
        return self._mode == RunMode.LOCAL
    
    def is_lan_mode(self) -> bool:
        """是否为局域网模式"""
        return self._mode == RunMode.LAN
    
    def should_enable_auth(self) -> bool:
        """是否应启用认证
        
        LOCAL模式：默认禁用认证
        LAN模式：默认启用认证
        """
        return self._mode == RunMode.LAN
    
    def get_rate_limit(self, config: dict) -> int:
        """获取速率限制
        
        Args:
            config: 速率限制配置
            
        Returns:
            每分钟请求数限制
        """
        if self.is_local_mode():
            # 本地模式：大幅放宽限制
            return config.get("requests_per_minute", 60) * 10
        else:
            # 局域网模式：正常限制
            return config.get("lan_requests_per_minute", 
                            config.get("requests_per_minute", 120))
    
    def check_ip_allowed(self, client_ip: str) -> bool:
        """检查IP是否允许访问
        
        Args:
            client_ip: 客户端IP地址
            
        Returns:
            是否允许访问
        """
        if self.is_local_mode():
            # 本地模式：只允许本地访问
            local_ips = ["127.0.0.1", "::1", "localhost"]
            if client_ip in local_ips:
                return True
            # 如果允许远程访问
            if self._allow_remote:
                return True
            return False
        
        # 局域网模式
        if not self._allowed_ips:
            # 未配置允许列表，允许所有局域网IP
            return self._is_lan_ip(client_ip)
        
        return client_ip in self._allowed_ips
    
    def _is_lan_ip(self, ip: str) -> bool:
        """检查是否为局域网IP"""
        # 简单判断：私有IP范围
        private_prefixes = [
            "10.",
            "172.16.", "172.17.", "172.18.", "172.19.",
            "172.20.", "172.21.", "172.22.", "172.23.",
            "172.24.", "172.25.", "172.26.", "172.27.",
            "172.28.", "172.29.", "172.30.", "172.31.",
            "192.168.",
            "127.", "::1"
        ]
        return any(ip.startswith(prefix) for prefix in private_prefixes)
    
    def get_security_headers(self) -> dict:
        """获取安全响应头
        
        Returns:
            安全头字典
        """
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
        }
        
        if self.is_lan_mode():
            headers["X-Frame-Options"] = "DENY"
            headers["Content-Security-Policy"] = "default-src 'self'"
        
        return headers


# 安全中间件适配
class AdaptiveSecurityMiddleware:
    """自适应安全中间件
    
    根据运行模式自动调整安全策略
    """
    
    def __init__(self, run_mode_manager: RunModeManager, config: dict):
        self._run_mode = run_mode_manager
        self._config = config
        
        # 速率限制器
        from collections import defaultdict
        from time import time
        self._requests = defaultdict(list)
        self._rate_limit = run_mode_manager.get_rate_limit(
            config.get("rate_limit", {})
        )
    
    async def __call__(self, request, call_next):
        """中间件处理逻辑"""
        client_ip = request.client.host if request.client else "unknown"
        
        # IP访问检查
        if not self._run_mode.check_ip_allowed(client_ip):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=403,
                content={"error": "访问被拒绝"}
            )
        
        # 速率限制检查
        if not self._check_rate_limit(client_ip):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"error": "请求过于频繁"}
            )
        
        # 执行请求
        response = await call_next(request)
        
        # 添加安全头
        for key, value in self._run_mode.get_security_headers().items():
            response.headers[key] = value
        
        return response
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """检查速率限制"""
        from time import time
        
        now = time()
        minute_ago = now - 60
        
        # 清理过期记录
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > minute_ago
        ]
        
        if len(self._requests[client_id]) >= self._rate_limit:
            return False
        
        self._requests[client_id].append(now)
        return True
```

### 6.2 认证实现（按需启用）

```python
# security/auth.py - 认证实现（条件启用）
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
from datetime import datetime, timedelta
import secrets
import structlog

from security.run_mode import RunModeManager

logger = structlog.get_logger()


class ConditionalAuthManager:
    """条件认证管理器
    
    根据运行模式决定是否启用认证：
    - LOCAL模式：跳过认证
    - LAN模式：启用完整认证
    """
    
    def __init__(
        self,
        run_mode_manager: RunModeManager,
        api_keys: list[str] = None
    ):
        self._run_mode = run_mode_manager
        self._api_keys = set(api_keys or [])
        
        # JWT配置（仅LAN模式使用）
        self._jwt_secret = secrets.token_urlsafe(32)
        self._jwt_algorithm = "HS256"
        self._jwt_expire_hours = 24
    
    async def optional_auth(
        self,
        api_key: Optional[str] = Depends(APIKeyHeader(name="X-API-Key", auto_error=False)),
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
    ) -> Optional[dict]:
        """可选认证（根据模式自动调整）
        
        LOCAL模式：返回None，跳过认证
        LAN模式：验证凭证
        """
        if not self._run_mode.should_enable_auth():
            # 本地模式：跳过认证
            return None
        
        # LAN模式：执行认证
        return await self._verify_credentials(api_key, credentials)
    
    async def _verify_credentials(
        self,
        api_key: Optional[str],
        credentials: Optional[HTTPAuthorizationCredentials]
    ) -> dict:
        """验证凭证"""
        # 优先检查API Key
        if api_key:
            if api_key in self._api_keys:
                return {"auth_type": "api_key", "key": api_key[:8] + "..."}
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的API Key"
            )
        
        # 检查JWT
        if credentials:
            try:
                payload = jwt.decode(
                    credentials.credentials,
                    self._jwt_secret,
                    algorithms=[self._jwt_algorithm]
                )
                return {"auth_type": "jwt", "user_id": payload.get("sub")}
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token已过期"
                )
            except jwt.InvalidTokenError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="无效的Token"
                )
        
        # 无凭证
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="需要认证"
        )
    
    def create_jwt_token(self, user_id: str, expires_hours: int = None) -> str:
        """创建JWT Token（仅LAN模式）"""
        if not self._run_mode.should_enable_auth():
            return ""  # 本地模式不需要Token
        
        expire = datetime.utcnow() + timedelta(hours=expires_hours or self._jwt_expire_hours)
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self._jwt_secret, algorithm=self._jwt_algorithm)
```

---

## 七、数据导入导出标准（新增核心改进）

### 7.1 Obsidian兼容导出

实现与Obsidian vault格式的双向兼容，降低用户迁移成本。

```python
# import_export/obsidian.py - Obsidian兼容导出
import os
import json
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import structlog

logger = structlog.get_logger()


class ObsidianExporter:
    """Obsidian格式导出器
    
    将LocalBrain数据导出为Obsidian兼容格式：
    - Markdown文件 + YAML Frontmatter
    - [[文档名]] 格式的双向链接
    - .obsidian 配置目录
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化导出器
        
        Args:
            config: 导出配置
        """
        obsidian_config = config.get("import_export", {}).get("obsidian", {})
        self._include_metadata = obsidian_config.get("include_metadata", True)
        self._link_format = obsidian_config.get("link_format", "wikilink")
        self._frontmatter_format = obsidian_config.get("frontmatter_format", "yaml")
    
    def export_vault(
        self,
        documents: List[Dict[str, Any]],
        output_dir: str,
        vault_name: str = "LocalBrain Export"
    ) -> Dict[str, Any]:
        """导出为Obsidian vault
        
        Args:
            documents: 文档列表
            output_dir: 输出目录
            vault_name: vault名称
            
        Returns:
            导出结果统计
        """
        vault_path = Path(output_dir) / vault_name
        vault_path.mkdir(parents=True, exist_ok=True)
        
        # 创建.obsidian配置目录
        obsidian_dir = vault_path / ".obsidian"
        obsidian_dir.mkdir(exist_ok=True)
        
        # 写入Obsidian配置
        self._write_obsidian_config(obsidian_dir)
        
        # 导出文档
        exported_count = 0
        failed_count = 0
        
        for doc in documents:
            try:
                self._export_document(doc, vault_path)
                exported_count += 1
            except Exception as e:
                logger.error("导出文档失败", doc_id=doc.get("id"), error=str(e))
                failed_count += 1
        
        # 写入导出日志
        self._write_export_log(vault_path, exported_count, failed_count)
        
        logger.info(
            "Obsidian vault导出完成",
            vault_path=str(vault_path),
            exported=exported_count,
            failed=failed_count
        )
        
        return {
            "vault_path": str(vault_path),
            "exported_count": exported_count,
            "failed_count": failed_count,
            "format": "obsidian"
        }
    
    def _write_obsidian_config(self, obsidian_dir: Path):
        """写入Obsidian配置文件"""
        # app.json - 基础配置
        app_config = {
            "legacyEditor": False,
            "promptDelete": False,
            "showLineNumber": True,
            "spellcheck": True,
            "tabSize": 2,
            "useTab": False
        }
        with open(obsidian_dir / "app.json", "w", encoding="utf-8") as f:
            json.dump(app_config, f, indent=2)
        
        # appearance.json - 外观配置
        appearance_config = {
            "baseFontSize": 16,
            "cssTheme": "",
            "theme": "system"
        }
        with open(obsidian_dir / "appearance.json", "w", encoding="utf-8") as f:
            json.dump(appearance_config, f, indent=2)
        
        # 核心插件配置
        core_plugins = {
            "file-explorer": True,
            "outline": True,
            "backlink": True,
            "graph-view": True,
            "search": True,
            "tag-pane": True
        }
        with open(obsidian_dir / "core-plugins.json", "w", encoding="utf-8") as f:
            json.dump(core_plugins, f, indent=2)
    
    def _export_document(self, doc: Dict[str, Any], vault_path: Path):
        """导出单个文档"""
        filename = self._sanitize_filename(doc.get("title", doc.get("filename", "untitled")))
        filepath = vault_path / f"{filename}.md"
        
        # 构建Frontmatter
        frontmatter = self._build_frontmatter(doc)
        
        # 转换链接格式
        content = self._convert_links(doc.get("content", ""))
        
        # 组装完整内容
        full_content = f"---\n{frontmatter}\n---\n\n{content}"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content)
    
    def _build_frontmatter(self, doc: Dict[str, Any]) -> str:
        """构建YAML Frontmatter"""
        if not self._include_metadata:
            return ""
        
        metadata = {
            "id": doc.get("id"),
            "title": doc.get("title"),
            "created": doc.get("created_at"),
            "modified": doc.get("updated_at"),
            "tags": doc.get("tags", []),
            "source": "LocalBrain"
        }
        
        # 移除空值
        metadata = {k: v for k, v in metadata.items() if v}
        
        # 转换为YAML格式
        lines = []
        for key, value in metadata.items():
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    def _convert_links(self, content: str) -> str:
        """转换链接格式为[[文档名]]格式"""
        import re
        
        # 假设原始链接格式为 [文本](路径)
        # 转换为 [[路径|文本]] 或 [[路径]]
        def replace_link(match):
            text = match.group(1)
            path = match.group(2)
            # 提取文件名（不含扩展名）
            name = Path(path).stem
            if text == name:
                return f"[[{name}]]"
            return f"[[{name}|{text}]]"
        
        # 匹配Markdown链接
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        converted = re.sub(pattern, replace_link, content)
        
        return converted
    
    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名"""
        # 移除不允许的字符
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename.strip()
    
    def _write_export_log(self, vault_path: Path, exported: int, failed: int):
        """写入导出日志"""
        log_content = f"""# LocalBrain Export Log

- Export Date: {datetime.now().isoformat()}
- Total Documents: {exported + failed}
- Successfully Exported: {exported}
- Failed: {failed}
- Format: Obsidian Vault

## Notes

This vault was exported from LocalBrain.
You can open it directly in Obsidian.
"""
        with open(vault_path / "Export Log.md", "w", encoding="utf-8") as f:
            f.write(log_content)


class ObsidianImporter:
    """Obsidian格式导入器
    
    从Obsidian vault导入文档到LocalBrain
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
    
    def import_vault(
        self,
        vault_path: str,
        document_handler  # 处理导入文档的回调
    ) -> Dict[str, Any]:
        """导入Obsidian vault
        
        Args:
            vault_path: vault目录路径
            document_handler: 文档处理回调函数
            
        Returns:
            导入结果统计
        """
        vault_dir = Path(vault_path)
        if not vault_dir.exists():
            raise ValueError(f"Vault目录不存在: {vault_path}")
        
        imported_count = 0
        failed_count = 0
        
        # 遍历所有Markdown文件
        for md_file in vault_dir.rglob("*.md"):
            # 跳过.obsidian目录
            if ".obsidian" in str(md_file):
                continue
            
            try:
                doc = self._parse_markdown_file(md_file, vault_dir)
                document_handler(doc)
                imported_count += 1
            except Exception as e:
                logger.error("导入文档失败", file=str(md_file), error=str(e))
                failed_count += 1
        
        logger.info(
            "Obsidian vault导入完成",
            vault_path=vault_path,
            imported=imported_count,
            failed=failed_count
        )
        
        return {
            "imported_count": imported_count,
            "failed_count": failed_count,
            "format": "obsidian"
        }
    
    def _parse_markdown_file(self, filepath: Path, vault_root: Path) -> Dict[str, Any]:
        """解析Markdown文件"""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 解析Frontmatter
        metadata = {}
        body = content
        
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter_text = parts[1].strip()
                metadata = self._parse_frontmatter(frontmatter_text)
                body = parts[2].strip()
        
        # 提取标题
        title = metadata.get("title", filepath.stem)
        
        # 提取标签
        tags = metadata.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        
        # 转换链接
        converted_body = self._convert_links_to_localbrain(body)
        
        return {
            "title": title,
            "content": converted_body,
            "tags": tags,
            "metadata": metadata,
            "source_path": str(filepath.relative_to(vault_root)),
            "file_type": "md"
        }
    
    def _parse_frontmatter(self, text: str) -> Dict[str, Any]:
        """解析YAML Frontmatter"""
        import yaml
        try:
            return yaml.safe_load(text) or {}
        except:
            return {}
    
    def _convert_links_to_localbrain(self, content: str) -> str:
        """将[[链接]]转换回标准格式"""
        import re
        
        def replace_wikilink(match):
            full = match.group(1)
            if "|" in full:
                name, text = full.split("|", 1)
                return f"[{text}]({name}.md)"
            return f"[{full}]({full}.md)"
        
        pattern = r'\[\[([^\]]+)\]\]'
        return re.sub(pattern, replace_wikilink, content)


class ImportExportManager:
    """导入导出管理器
    
    统一管理各种格式的导入导出
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._obsidian_exporter = ObsidianExporter(config)
        self._obsidian_importer = ObsidianImporter(config)
    
    def export_to(
        self,
        format: str,
        documents: List[Dict[str, Any]],
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """导出到指定格式
        
        Args:
            format: 导出格式 (obsidian, json, markdown)
            documents: 文档列表
            output_path: 输出路径
            **kwargs: 其他参数
            
        Returns:
            导出结果
        """
        if format == "obsidian":
            return self._obsidian_exporter.export_vault(
                documents, output_path, kwargs.get("vault_name", "LocalBrain Export")
            )
        elif format == "json":
            return self._export_json(documents, output_path)
        elif format == "markdown":
            return self._export_markdown(documents, output_path)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def import_from(
        self,
        format: str,
        source_path: str,
        document_handler,
        **kwargs
    ) -> Dict[str, Any]:
        """从指定格式导入
        
        Args:
            format: 导入格式 (obsidian, notion, logseq)
            source_path: 源路径
            document_handler: 文档处理回调
            **kwargs: 其他参数
            
        Returns:
            导入结果
        """
        if format == "obsidian":
            return self._obsidian_importer.import_vault(source_path, document_handler)
        elif format == "notion":
            return self._import_notion(source_path, document_handler)
        elif format == "logseq":
            return self._import_logseq(source_path, document_handler)
        else:
            raise ValueError(f"不支持的导入格式: {format}")
    
    def _export_json(self, documents: List[Dict], output_path: str) -> Dict[str, Any]:
        """导出为JSON格式"""
        output_file = Path(output_path) / "localbrain_export.json"
        
        export_data = {
            "version": "1.0",
            "export_date": datetime.now().isoformat(),
            "documents": documents
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return {
            "output_path": str(output_file),
            "exported_count": len(documents),
            "format": "json"
        }
    
    def _export_markdown(self, documents: List[Dict], output_path: str) -> Dict[str, Any]:
        """导出为纯Markdown文件夹"""
        output_dir = Path(output_path) / "markdown_export"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for doc in documents:
            filename = f"{doc.get('title', 'untitled')}.md"
            filepath = output_dir / filename
            
            content = doc.get("content", "")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        
        return {
            "output_path": str(output_dir),
            "exported_count": len(documents),
            "format": "markdown"
        }
    
    def _import_notion(self, source_path: str, document_handler) -> Dict[str, Any]:
        """从Notion导入（待实现）"""
        # Notion导出通常是ZIP文件，需要解压处理
        raise NotImplementedError("Notion导入功能开发中")
    
    def _import_logseq(self, source_path: str, document_handler) -> Dict[str, Any]:
        """从Logseq导入"""
        # Logseq格式与Obsidian类似，可以复用部分逻辑
        return self._obsidian_importer.import_vault(source_path, document_handler)
```

---

## 八、API 接口设计

### 8.1 后端 API 架构

后端基于 FastAPI 构建，提供完整的 RESTful API 接口。API 设计遵循资源导向原则，每个资源对应独立的路由和操作。

```python
# api/routes.py - API 路由定义（增强版）
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import structlog

from database.models import Document, Tag, DocLink, AuditLog
from database.database import get_db, SessionLocal
from rag.pipeline_enhanced import EnhancedRAGPipeline
from security.run_mode import RunModeManager, RunMode
from security.auth import ConditionalAuthManager
from import_export.obsidian import ImportExportManager

logger = structlog.get_logger()
router = APIRouter()
rag_pipeline = None
run_mode_manager = None
auth_manager = None
import_export_manager = None

# ==================== 响应模型 ====================

class DocumentResponse(BaseModel):
    id: int
    filepath: str
    filename: str
    file_type: str
    title: Optional[str]
    summary: Optional[str]
    word_count: int
    author: Optional[str] = None
    language: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    tags: List[str] = []

    class Config:
        from_attributes = True

class SearchResultResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    query: str
    search_type: str
    took_ms: float
    warning: Optional[str] = None
    metrics: Optional[dict] = None

class QAResponse(BaseModel):
    answer: str
    source_documents: List[dict]
    warning: Optional[str] = None
    metrics: Optional[dict] = None

class ModelStatusResponse(BaseModel):
    embedding_model: str
    llm_model: Optional[str]
    embedding_available: bool
    llm_available: bool
    run_mode: str

class ExportRequest(BaseModel):
    format: str = Field(default="obsidian", description="导出格式: obsidian, json, markdown")
    output_path: str = Field(default="./exports", description="输出路径")
    vault_name: Optional[str] = Field(default="LocalBrain Export", description="Vault名称")

class ExportResponse(BaseModel):
    output_path: str
    exported_count: int
    failed_count: int = 0
    format: str

class ImportRequest(BaseModel):
    format: str = Field(default="obsidian", description="导入格式: obsidian, notion, logseq")
    source_path: str

class ImportResponse(BaseModel):
    imported_count: int
    failed_count: int
    format: str

class RunModeResponse(BaseModel):
    current_mode: str
    auth_enabled: bool
    allow_remote: bool

class SearchConfigRequest(BaseModel):
    semantic_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

# ==================== 运行模式接口 ====================

@router.get("/run-mode", response_model=RunModeResponse)
async def get_run_mode():
    """获取当前运行模式信息"""
    return RunModeResponse(
        current_mode=run_mode_manager.mode.value,
        auth_enabled=run_mode_manager.should_enable_auth(),
        allow_remote=run_mode_manager._allow_remote
    )

# ==================== 健康检查接口 ====================

@router.get("/health")
async def health_check():
    """健康检查端点"""
    health_status = {
        "status": "healthy",
        "version": "3.0.0",
        "run_mode": run_mode_manager.mode.value,
        "database": "ok",
        "vectorstore": "ok",
        "models": {
            "embedding": "ok",
            "llm": "unknown"
        }
    }

    # 检查数据库
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    return health_status

# ==================== 文档管理接口 ====================

@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 20,
    tag: Optional[str] = None,
    file_type: Optional[str] = None,
    db = Depends(get_db)
):
    """获取文档列表"""
    query = db.query(Document)

    if tag:
        query = query.join(Document.tags).filter(Tag.name == tag)
    if file_type:
        query = query.filter(Document.file_type == file_type)

    documents = query.order_by(Document.updated_at.desc()).offset(skip).limit(limit).all()

    logger.info("获取文档列表", skip=skip, limit=limit, count=len(documents))
    return documents

@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: int, db = Depends(get_db)):
    """获取单个文档详情"""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    return doc

# ==================== 搜索接口（增强版） ====================

@router.get("/search", response_model=SearchResultResponse)
async def search_documents(
    q: str,
    search_type: str = "hybrid",
    limit: int = 10,
    db = Depends(get_db)
):
    """搜索文档（带质量评估）"""
    import time
    start_time = time.time()

    # 执行增强搜索
    result = rag_pipeline.search(q, k=limit, search_type=search_type)

    # 转换结果
    documents = []
    for doc, score in result.get("results", []):
        doc_id = doc.metadata.get("doc_id")
        if doc_id:
            db_doc = db.query(Document).filter(Document.id == doc_id).first()
            if db_doc:
                documents.append(db_doc)

    took_ms = (time.time() - start_time) * 1000

    logger.info("搜索文档", query=q[:50], search_type=search_type, results=len(documents))

    return SearchResultResponse(
        documents=documents[:limit],
        total=len(documents),
        query=q,
        search_type=search_type,
        took_ms=round(took_ms, 2),
        warning=result.get("warning"),
        metrics=result.get("metrics")
    )

# ==================== 搜索配置接口（新增） ====================

@router.post("/search/config")
async def update_search_config(config: SearchConfigRequest):
    """更新搜索权重配置（用户可配置接口）"""
    rag_pipeline._hybrid_search.set_weights(
        config.semantic_weight,
        config.keyword_weight
    )
    rag_pipeline._evaluator._relevance_threshold = config.relevance_threshold
    
    logger.info("搜索配置已更新", config=config.dict())
    return {"message": "搜索配置已更新", "config": config.dict()}

# ==================== 问答接口（增强版） ====================

@router.post("/qa", response_model=QAResponse)
async def ask_question(question: str):
    """问答接口（带质量评估）"""
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG 管道未初始化")

    result = rag_pipeline.ask(question)

    logger.info("问答请求", question=question[:50])
    return QAResponse(
        answer=result["answer"],
        source_documents=result["source_documents"],
        warning=result.get("warning"),
        metrics=result.get("metrics")
    )

# ==================== 导入导出接口（新增） ====================

@router.post("/export", response_model=ExportResponse)
async def export_documents(request: ExportRequest, db = Depends(get_db)):
    """导出文档到指定格式"""
    # 获取所有文档
    documents = db.query(Document).all()
    
    doc_list = []
    for doc in documents:
        doc_list.append({
            "id": doc.id,
            "title": doc.title,
            "filename": doc.filename,
            "content": doc.summary or "",  # 实际应获取完整内容
            "tags": [t.name for t in doc.tags],
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "updated_at": doc.updated_at.isoformat() if doc.updated_at else None
        })
    
    result = import_export_manager.export_to(
        format=request.format,
        documents=doc_list,
        output_path=request.output_path,
        vault_name=request.vault_name
    )
    
    logger.info("文档导出完成", format=request.format, count=result["exported_count"])
    
    return ExportResponse(**result)

@router.post("/import", response_model=ImportResponse)
async def import_documents(
    request: ImportRequest,
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """从指定格式导入文档"""
    def handle_imported_doc(doc):
        """处理导入的文档"""
        # 创建或更新文档记录
        existing = db.query(Document).filter(
            Document.title == doc["title"]
        ).first()
        
        if not existing:
            new_doc = Document(
                filename=f"{doc['title']}.md",
                filepath=doc.get("source_path", ""),
                file_type="md",
                title=doc["title"],
                summary=doc["content"][:500] if doc.get("content") else ""
            )
            db.add(new_doc)
            db.commit()
            
            # 添加到向量存储
            rag_pipeline.add_texts([doc["content"]], metadatas=[{
                "doc_id": new_doc.id,
                "title": doc["title"],
                "tags": ",".join(doc.get("tags", []))
            }])
    
    result = import_export_manager.import_from(
        format=request.format,
        source_path=request.source_path,
        document_handler=handle_imported_doc
    )
    
    logger.info("文档导入完成", format=request.format, count=result["imported_count"])
    
    return ImportResponse(**result)

# ==================== 评估统计接口（新增） ====================

@router.get("/evaluation/summary")
async def get_evaluation_summary():
    """获取检索质量评估摘要"""
    if not rag_pipeline or not rag_pipeline._evaluator:
        return {"total_evaluations": 0}
    
    return rag_pipeline._evaluator.get_evaluation_summary()
```

---

## 九、前端设计

### 9.1 界面布局

前端采用 React 构建单页应用，界面设计借鉴现代笔记应用的最佳实践，采用三栏布局提供高效的信息浏览和操作体验。

左侧边栏宽度约 250 像素，可折叠。包含应用 Logo 和名称、文件夹树形导航、标签云、快速添加按钮和设置入口。文件夹树支持展开折叠操作，点击文件夹可快速定位文档。标签云以标签云形式展示所有标签，点击标签可过滤相关文档。

中间主内容区是核心工作区域，支持多种视图模式。文档列表视图以卡片或列表形式展示文档，包含标题、摘要、更新时间等信息。文档查看器展示文档详细内容，支持 Markdown 渲染和代码高亮。搜索结果视图展示搜索返回的文档列表，每条结果包含相关性评分和匹配片段预览。问答视图展示对话界面，显示问题和 AI 回答以及引用的文档来源。

右侧边栏宽度约 300 像素，可切换显示。包含文档元数据面板（标题、创建时间、字数等）、目录导航（自动从文档标题生成）、标签管理（添加移除标签）、双向链接面板（当前文档的入链和出链）。

### 9.2 核心组件

```typescript
// types/index.ts - 类型定义
export interface Document {
  id: number;
  filepath: string;
  filename: string;
  file_type: 'md' | 'txt' | 'pdf';
  title: string;
  summary?: string;
  word_count: number;
  author?: string;
  language?: string;
  tags: Tag[];
  created_at: string;
  updated_at: string;
}

export interface Tag {
  id: number;
  name: string;
  color: string;
  parent_id?: number;
  document_count: number;
}

export interface SearchResult {
  documents: Document[];
  total: number;
  query: string;
  search_type: string;
  took_ms: number;
  warning?: string;
  metrics?: EvaluationMetrics;
}

export interface QAAnswer {
  answer: string;
  source_documents: {
    content: string;
    metadata: Record<string, any>;
  }[];
  warning?: string;
  metrics?: EvaluationMetrics;
}

export interface EvaluationMetrics {
  retrieved_count?: number;
  avg_doc_length?: number;
  ragas_faithfulness?: number;
  ragas_answer_relevance?: number;
  ragas_context_precision?: number;
}

export interface RunMode {
  current_mode: 'local' | 'lan';
  auth_enabled: boolean;
  allow_remote: boolean;
}

export interface SearchConfig {
  semantic_weight: number;
  keyword_weight: number;
  relevance_threshold: number;
}

// components/SearchConfigPanel.tsx - 搜索配置面板（新增）
import { useState, useEffect } from 'react';
import { Slider, Button, Card } from '@/components/ui';
import { Settings, Save, RotateCcw } from 'lucide-react';

export function SearchConfigPanel({ onSave }: { onSave: (config: SearchConfig) => void }) {
  const [config, setConfig] = useState<SearchConfig>({
    semantic_weight: 0.5,
    keyword_weight: 0.5,
    relevance_threshold: 0.3
  });
  
  const [hasChanges, setHasChanges] = useState(false);
  
  const handleSemanticChange = (value: number) => {
    setConfig(prev => ({
      ...prev,
      semantic_weight: value,
      keyword_weight: 1 - value
    }));
    setHasChanges(true);
  };
  
  const handleThresholdChange = (value: number) => {
    setConfig(prev => ({
      ...prev,
      relevance_threshold: value
    }));
    setHasChanges(true);
  };
  
  const handleReset = () => {
    setConfig({
      semantic_weight: 0.5,
      keyword_weight: 0.5,
      relevance_threshold: 0.3
    });
    setHasChanges(false);
  };
  
  const handleSave = async () => {
    await fetch('/api/search/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    onSave(config);
    setHasChanges(false);
  };
  
  return (
    <Card className="p-4">
      <div className="flex items-center gap-2 mb-4">
        <Settings className="w-5 h-5" />
        <h3 className="font-semibold">搜索配置</h3>
      </div>
      
      <div className="space-y-6">
        {/* 语义/关键词权重 */}
        <div>
          <label className="text-sm text-slate-600 mb-2 block">
            语义搜索权重: {Math.round(config.semantic_weight * 100)}%
          </label>
          <Slider
            value={[config.semantic_weight]}
            min={0}
            max={1}
            step={0.1}
            onValueChange={([value]) => handleSemanticChange(value)}
          />
          <p className="text-xs text-slate-500 mt-1">
            关键词搜索权重: {Math.round(config.keyword_weight * 100)}%
          </p>
        </div>
        
        {/* 相关性阈值 */}
        <div>
          <label className="text-sm text-slate-600 mb-2 block">
            最低相关性阈值: {config.relevance_threshold.toFixed(2)}
          </label>
          <Slider
            value={[config.relevance_threshold]}
            min={0}
            max={1}
            step={0.05}
            onValueChange={([value]) => handleThresholdChange(value)}
          />
          <p className="text-xs text-slate-500 mt-1">
            低于此阈值的结果将被过滤
          </p>
        </div>
        
        {/* 操作按钮 */}
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleReset}
            disabled={!hasChanges}
          >
            <RotateCcw className="w-4 h-4 mr-1" />
            重置
          </Button>
          <Button
            size="sm"
            onClick={handleSave}
            disabled={!hasChanges}
          >
            <Save className="w-4 h-4 mr-1" />
            保存
          </Button>
        </div>
      </div>
    </Card>
  );
}

// components/EvaluationMetrics.tsx - 评估指标展示（新增）
import { Card } from '@/components/ui';
import { AlertCircle, CheckCircle, Info } from 'lucide-react';

export function EvaluationMetricsPanel({ metrics, warning }: { 
  metrics?: EvaluationMetrics;
  warning?: string;
}) {
  if (!metrics && !warning) return null;
  
  return (
    <Card className="p-3 bg-slate-50 dark:bg-slate-800">
      {/* 警告信息 */}
      {warning && (
        <div className="flex items-center gap-2 text-amber-600 dark:text-amber-400 mb-2">
          <AlertCircle className="w-4 h-4" />
          <span className="text-sm">{warning}</span>
        </div>
      )}
      
      {/* 评估指标 */}
      {metrics && (
        <div className="grid grid-cols-2 gap-2 text-xs">
          {metrics.ragas_faithfulness !== undefined && (
            <div className="flex items-center gap-1">
              <CheckCircle className="w-3 h-3 text-green-500" />
              <span>忠实度: {(metrics.ragas_faithfulness * 100).toFixed(0)}%</span>
            </div>
          )}
          {metrics.ragas_answer_relevance !== undefined && (
            <div className="flex items-center gap-1">
              <CheckCircle className="w-3 h-3 text-green-500" />
              <span>相关性: {(metrics.ragas_answer_relevance * 100).toFixed(0)}%</span>
            </div>
          )}
          {metrics.retrieved_count !== undefined && (
            <div className="flex items-center gap-1">
              <Info className="w-3 h-3 text-blue-500" />
              <span>检索数量: {metrics.retrieved_count}</span>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

// components/ImportExportPanel.tsx - 导入导出面板（新增）
import { useState } from 'react';
import { Button, Select, Input, Card } from '@/components/ui';
import { Download, Upload, FileDown, FileUp } from 'lucide-react';

export function ImportExportPanel() {
  const [exportFormat, setExportFormat] = useState('obsidian');
  const [importFormat, setImportFormat] = useState('obsidian');
  const [exportPath, setExportPath] = useState('./exports');
  const [importPath, setImportPath] = useState('');
  const [isExporting, setIsExporting] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  
  const handleExport = async () => {
    setIsExporting(true);
    try {
      const response = await fetch('/api/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          format: exportFormat,
          output_path: exportPath
        })
      });
      const result = await response.json();
      alert(`导出成功！共导出 ${result.exported_count} 个文档`);
    } catch (error) {
      alert('导出失败：' + error.message);
    }
    setIsExporting(false);
  };
  
  const handleImport = async () => {
    if (!importPath) {
      alert('请输入导入路径');
      return;
    }
    setIsImporting(true);
    try {
      const response = await fetch('/api/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          format: importFormat,
          source_path: importPath
        })
      });
      const result = await response.json();
      alert(`导入成功！共导入 ${result.imported_count} 个文档`);
    } catch (error) {
      alert('导入失败：' + error.message);
    }
    setIsImporting(false);
  };
  
  return (
    <div className="space-y-4">
      {/* 导出 */}
      <Card className="p-4">
        <div className="flex items-center gap-2 mb-3">
          <FileDown className="w-5 h-5" />
          <h3 className="font-semibold">导出文档</h3>
        </div>
        <div className="space-y-3">
          <div>
            <label className="text-sm text-slate-600 mb-1 block">导出格式</label>
            <Select value={exportFormat} onValueChange={setExportFormat}>
              <option value="obsidian">Obsidian Vault</option>
              <option value="json">JSON</option>
              <option value="markdown">纯Markdown</option>
            </Select>
          </div>
          <div>
            <label className="text-sm text-slate-600 mb-1 block">输出路径</label>
            <Input
              value={exportPath}
              onChange={(e) => setExportPath(e.target.value)}
              placeholder="./exports"
            />
          </div>
          <Button
            className="w-full"
            onClick={handleExport}
            disabled={isExporting}
          >
            <Download className="w-4 h-4 mr-2" />
            {isExporting ? '导出中...' : '开始导出'}
          </Button>
        </div>
      </Card>
      
      {/* 导入 */}
      <Card className="p-4">
        <div className="flex items-center gap-2 mb-3">
          <FileUp className="w-5 h-5" />
          <h3 className="font-semibold">导入文档</h3>
        </div>
        <div className="space-y-3">
          <div>
            <label className="text-sm text-slate-600 mb-1 block">导入格式</label>
            <Select value={importFormat} onValueChange={setImportFormat}>
              <option value="obsidian">Obsidian Vault</option>
              <option value="logseq">Logseq</option>
              <option value="notion">Notion (开发中)</option>
            </Select>
          </div>
          <div>
            <label className="text-sm text-slate-600 mb-1 block">源路径</label>
            <Input
              value={importPath}
              onChange={(e) => setImportPath(e.target.value)}
              placeholder="/path/to/vault"
            />
          </div>
          <Button
            className="w-full"
            onClick={handleImport}
            disabled={isImporting || !importPath}
          >
            <Upload className="w-4 h-4 mr-2" />
            {isImporting ? '导入中...' : '开始导入'}
          </Button>
        </div>
      </Card>
    </div>
  );
}
```

---

## 十、测试体系

### 10.1 测试策略概述

系统采用多层次测试体系，确保代码质量和功能正确性。测试类型包括单元测试、集成测试、端到端测试和性能测试，各层次测试协同配合，形成完整的质量保障体系。

| 测试类型 | 工具/框架 | 覆盖范围 | 目标覆盖率 |
|----------|-----------|----------|------------|
| 单元测试 | pytest + pytest-asyncio | 核心业务逻辑、工具函数 | >80% |
| 集成测试 | FastAPI TestClient | API端点、数据库操作 | >70% |
| 端到端测试 | Playwright | 用户关键路径 | 核心流程100% |
| 性能测试 | Locust | 并发处理、响应时间 | 关键接口 |

### 10.2 RAG管道测试（新增）

```python
# tests/test_rag_enhanced.py - RAG管道增强测试
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from rag.text_splitter import SmartTextSplitter, SplittingStrategy
from rag.parent_document_retriever import ParentDocumentRetriever
from rag.hybrid_search_enhanced import RRFHybridSearch
from rag.evaluation import RetrievalEvaluator


class TestSmartTextSplitter:
    """智能文本分割器测试"""
    
    @pytest.fixture
    def splitter_config(self):
        return {
            "splitting_strategies": {
                "markdown": {
                    "strategy": "markdown_header",
                    "chunk_size": 500,
                    "chunk_overlap": 50
                },
                "pdf": {
                    "strategy": "semantic",
                    "chunk_size": 800
                },
                "txt": {
                    "strategy": "recursive",
                    "chunk_size": 500
                }
            }
        }
    
    @pytest.fixture
    def splitter(self, splitter_config):
        return SmartTextSplitter(splitter_config)
    
    def test_markdown_splitting_with_headers(self, splitter):
        """测试Markdown按标题分割"""
        doc = Document(
            page_content="# 标题1\n\n内容1\n\n## 标题2\n\n内容2",
            metadata={"file_type": "md"}
        )
        
        chunks = splitter.split_documents([doc])
        
        assert len(chunks) > 0
        # 验证保留了标题结构
        assert any("标题1" in chunk.page_content for chunk in chunks)
    
    def test_pdf_splitting(self, splitter):
        """测试PDF分割"""
        doc = Document(
            page_content="PDF内容" * 100,
            metadata={"file_type": "pdf"}
        )
        
        chunks = splitter.split_documents([doc])
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.page_content) <= 1000  # chunk_size + overlap
    
    def test_fallback_to_default(self, splitter):
        """测试未知类型回退到默认分割"""
        doc = Document(
            page_content="普通文本内容",
            metadata={"file_type": "unknown"}
        )
        
        chunks = splitter.split_documents([doc])
        
        assert len(chunks) > 0


class TestParentDocumentRetriever:
    """父文档检索器测试"""
    
    @pytest.fixture
    def mock_vectorstore(self):
        return MagicMock()
    
    @pytest.fixture
    def retriever_config(self):
        return {
            "parent_document": {
                "enabled": True,
                "parent_chunk_size": 2000,
                "child_chunk_size": 400
            }
        }
    
    def test_add_documents_creates_hierarchy(self, mock_vectorstore, retriever_config):
        """测试添加文档建立父子层级"""
        retriever = ParentDocumentRetriever(mock_vectorstore, retriever_config)
        
        doc = Document(
            page_content="长文档内容" * 500,
            metadata={"doc_id": "test-1"}
        )
        
        parent_ids = retriever.add_documents([doc])
        
        assert len(parent_ids) > 0
        # 验证父子映射建立
        assert len(retriever._parent_child_map) > 0
    
    def test_retrieve_expands_to_parent(self, mock_vectorstore, retriever_config):
        """测试检索扩展到父文档"""
        retriever = ParentDocumentRetriever(mock_vectorstore, retriever_config)
        
        # 模拟向量存储返回子文档
        child_doc = Document(
            page_content="子文档内容",
            metadata={"parent_id": "parent-1"}
        )
        mock_vectorstore.similarity_search.return_value = [child_doc]
        
        # 模拟父文档存储
        parent_doc = Document(
            page_content="父文档完整内容",
            metadata={"parent_id": "parent-1", "is_parent": True}
        )
        retriever._docstore.mset([("parent-1", parent_doc)])
        
        results = retriever.retrieve("测试查询")
        
        # 应返回父文档
        assert len(results) > 0


class TestRRFHybridSearch:
    """RRF混合搜索测试"""
    
    @pytest.fixture
    def mock_vectorstore(self):
        vs = MagicMock()
        vs.similarity_search_with_score.return_value = [
            (Document(page_content="文档1", metadata={"id": "1"}), 0.1),
            (Document(page_content="文档2", metadata={"id": "2"}), 0.2),
        ]
        return vs
    
    @pytest.fixture
    def rrf_search(self, mock_vectorstore):
        return RRFHybridSearch(
            mock_vectorstore,
            keyword_index=None,
            config={"hybrid_search": {"rrf_k": 60}}
        )
    
    def test_rrf_formula(self, rrf_search):
        """测试RRF公式计算"""
        # 手动验证RRF分数
        # score = 1 / (k + rank)
        # rank 1: 1/61 ≈ 0.0164
        # rank 2: 1/62 ≈ 0.0161
        
        results = rrf_search.search("测试查询", k=2)
        
        assert len(results) > 0
        # 验证结果按分数降序排列
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_weight_adjustment(self, rrf_search):
        """测试权重调整"""
        rrf_search.set_weights(0.7, 0.3)
        
        assert rrf_search._semantic_weight == 0.7
        assert rrf_search._keyword_weight == 0.3


class TestRetrievalEvaluator:
    """检索质量评估测试"""
    
    @pytest.fixture
    def evaluator_config(self):
        return {
            "evaluation": {
                "enabled": True,
                "relevance_threshold": 0.3
            }
        }
    
    @pytest.fixture
    def evaluator(self, evaluator_config):
        return RetrievalEvaluator(evaluator_config)
    
    def test_filter_by_relevance(self, evaluator):
        """测试相关性过滤"""
        results = [
            (Document(page_content="高相关"), 0.8),
            (Document(page_content="低相关"), 0.2),
            (Document(page_content="中相关"), 0.5),
        ]
        
        filtered = evaluator.filter_by_relevance(results, threshold=0.3)
        
        assert len(filtered) == 2
        assert all(score >= 0.3 for _, score in filtered)
    
    def test_quality_warning(self, evaluator):
        """测试质量警告"""
        # 低相关性结果
        low_results = [(Document(page_content="低"), 0.1)]
        warning = evaluator.get_quality_warning(low_results)
        assert warning is not None
        assert "未找到高相关内容" in warning
        
        # 高相关性结果
        high_results = [(Document(page_content="高"), 0.8)]
        warning = evaluator.get_quality_warning(high_results)
        assert warning is None
    
    def test_coverage_calculation(self, evaluator):
        """测试覆盖率计算"""
        docs = [
            Document(page_content="这是关于Python编程的内容"),
            Document(page_content="Java也是一种编程语言"),
        ]
        ground_truth = "Python是一种流行的编程语言"
        
        coverage = evaluator._calculate_coverage(docs, ground_truth)
        
        assert coverage > 0  # 应该有一定的覆盖率
```

---

## 十一、部署方案

### 11.1 Docker Compose 部署

项目采用 Docker Compose 进行容器化部署，实现开发环境和生产环境的一致性。

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./config.yaml:/app/config.yaml
      - ./logs:/app/logs
      - ./backups:/app/backups
      - ./exports:/app/exports
    environment:
      - PYTHONUNBUFFERED=1
      - LOCALBRAIN_ENV=production
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - backend
    restart: unless-stopped

# 数据卷
volumes:
  data:
  logs:
  backups:
  exports:
```

### 11.2 本地运行

对于本地开发环境，可以直接运行各个组件。

```bash
# 1. 安装依赖
cd backend && pip install -r requirements.txt
cd frontend && npm install

# 2. 配置环境变量
export LOCALBRAIN_ENV=development

# 3. 启动后端
cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 4. 启动前端
cd frontend && npm run dev

# 5. 运行测试
pytest tests/

# 6. 运行数据库迁移
alembic upgrade head
```

---

## 十二、项目目录结构

```
localbrain/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes.py          # API 路由
│   │   │   ├── exceptions.py      # 异常处理
│   │   │   └── middleware.py      # 中间件
│   │   ├── core/
│   │   │   ├── config.py           # 配置管理
│   │   │   ├── logging_config.py   # 日志配置
│   │   │   ├── cache.py            # 缓存系统
│   │   │   └── backup.py           # 备份系统
│   │   ├── models/
│   │   │   ├── base.py             # 抽象基类
│   │   │   ├── providers.py        # 模型实现
│   │   │   └── factory.py          # 模型工厂
│   │   ├── rag/
│   │   │   ├── pipeline.py         # RAG 管道基础
│   │   │   ├── pipeline_enhanced.py # 增强版RAG管道
│   │   │   ├── text_splitter.py    # 智能文本分割器
│   │   │   ├── parent_document_retriever.py # 父文档检索器
│   │   │   ├── hybrid_search_enhanced.py # RRF混合搜索
│   │   │   ├── evaluation.py       # 检索质量评估
│   │   │   └── file_watcher.py     # 文件监控
│   │   ├── database/
│   │   │   ├── models.py            # ORM 模型
│   │   │   └── database.py         # 数据库连接
│   │   ├── security/
│   │   │   ├── run_mode.py          # 运行模式管理
│   │   │   ├── auth.py              # 认证授权
│   │   │   └── validation.py        # 输入验证
│   │   ├── import_export/
│   │   │   ├── obsidian.py          # Obsidian导入导出
│   │   │   ├── notion.py            # Notion导入（待实现）
│   │   │   └── manager.py           # 导入导出管理器
│   │   └── main.py                 # 应用入口
│   ├── tests/
│   │   ├── unit/                    # 单元测试
│   │   ├── integration/             # 集成测试
│   │   ├── e2e/                     # 端到端测试
│   │   └── test_rag_enhanced.py     # RAG增强测试
│   ├── alembic/                     # 数据库迁移
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── components/             # React 组件
│   │   │   ├── SearchConfigPanel.tsx  # 搜索配置面板
│   │   │   ├── EvaluationMetrics.tsx  # 评估指标展示
│   │   │   └── ImportExportPanel.tsx  # 导入导出面板
│   │   ├── pages/                  # 页面组件
│   │   ├── hooks/                  # 自定义 Hooks
│   │   ├── api/                    # API 调用
│   │   ├── types/                  # 类型定义
│   │   ├── i18n/                   # 国际化资源
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts
│   └── Dockerfile
│
├── config.yaml                      # 主配置文件
├── docker-compose.yml
├── alembic.ini                      # Alembic 配置
└── README.md
```

---

## 十三、实施阶段建议

基于上述方案的重要性和依赖关系，建议分三个阶段实施：

### 第一阶段：核心功能完善（4-6周）

- 文本分割策略矩阵（智能分割器实现）
- RRF混合搜索 + Reranking
- 运行模式区分（本地/局域网）
- 测试体系搭建（单元测试、集成测试框架）
- 错误处理机制（自定义异常、全局异常处理器）
- 增量更新机制（文件监控、基于哈希的增量索引）
- 数据备份恢复（自动备份、一键恢复）
- 日志系统（structlog结构化日志）

### 第二阶段：检索增强（4-6周）

- Parent Document Retriever（父文档检索器）
- 检索质量评估（RAGAS框架集成）
- 数据导入导出（Obsidian兼容）
- PDF处理增强（OCR支持、表格提取）
- 全文搜索优化（SQLite FTS5、jieba中文分词）
- WebSocket实时通信（流式输出、进度推送）
- 健康检查监控（Prometheus指标）

### 第三阶段：用户体验优化（灵活安排）

- 前端搜索配置界面
- 评估指标可视化
- 导入导出向导
- 主题国际化（深色模式、多语言）
- 性能优化（前端懒加载、后端异步优化）
- 文档编写（用户手册、API文档）

---

## 十四、总结

本增强版V3实施方案在原有基础上进行了全面的核心改进。系统设计遵循模块化原则，各组件之间松耦合，便于后续功能扩展和维护。

核心改进点包括：

1. **文本分割策略矩阵**：针对Markdown、PDF、代码等不同文档类型采用差异化分割策略，显著提升检索质量。

2. **Parent Document Retriever**：检索小块匹配精确相关性，返回大块提供完整上下文，解决"检索精确但上下文丢失"的经典问题。

3. **RRF + Reranking**：使用业界标准的Reciprocal Rank Fusion算法合并混合搜索结果，并支持Cross-Encoder重排序进一步提升质量。

4. **检索质量评估机制**：集成RAGAS评估框架，提供忠实度、答案相关性、上下文精度等多维度评估指标。

5. **安全模式区分**：本地单用户模式简化认证优化体验，局域网共享模式启用完整认证保护安全。

6. **数据导入导出标准**：支持Obsidian vault格式双向兼容，降低用户迁移成本。

系统架构清晰，技术选型合理，具备良好的可扩展性和可维护性，适合作为生产级个人知识库系统的基础框架。
