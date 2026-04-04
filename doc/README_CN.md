# LocalBrain

English | [中文](./README_CN.md)

## 项目简介

LocalBrain 是一个**本地个人知识库管理系统**，具备 AI 智能问答功能。它允许你使用自然语言组织、搜索和与文档交互，所有功能完全在本地运行。

## 功能特性

### 📄 文档管理
- **多格式支持**：导入 PDF、Markdown 和纯文本文件
- **自动处理**：文档自动解析、分块和索引
- **分类组织**：将文档组织到自定义分类中
- **状态追踪**：监控处理状态（待处理、处理中、已完成、失败）

### 🔍 智能搜索
- **语义搜索**：使用自然语言查询查找相关内容
- **关键词搜索**：传统文本搜索
- **混合搜索**：结合语义和关键词方法获得更好的结果

### 🤖 AI 智能问答
- **RAG（检索增强生成）**：针对文档内容提问
- **来源追溯**：回答包含源文档引用
- **多 LLM 支持**：兼容多种语言模型

### 🔧 灵活的模型配置
- **LLM 提供商**：支持 LM Studio、Ollama、OpenAI、Anthropic 和自定义端点
- **嵌入模型**：可配置嵌入服务（LM Studio、HuggingFace）
- **本地优先**：使用本地模型完全离线运行

### 📁 导入与导出
- **导入**：支持 Obsidian、Notion 和 Logseq 格式
- **导出**：导出为 Obsidian、JSON 和 Markdown 格式

## 技术栈

| 组件 | 技术 |
|------|------|
| **后端** | FastAPI, LangChain, ChromaDB, SQLAlchemy |
| **前端** | React 19, TypeScript, Vite, Tailwind CSS |
| **AI/ML** | LangChain, Sentence Transformers, 多种 LLM 提供商 |
| **数据库** | SQLite (通过 SQLAlchemy) |
| **向量存储** | ChromaDB |

## 项目结构

```
LocalBrain/
├── backend/                 # FastAPI 后端
│   └── app/
│       ├── api/             # API 路由处理
│       │   ├── documents.py # 文档管理
│       │   ├── search.py    # 搜索接口
│       │   ├── qa.py        # 问答接口
│       │   ├── categories.py
│       │   ├── settings.py
│       │   └── models.py
│       ├── core/            # 核心配置
│       │   ├── config.py
│       │   ├── config_manager.py
│       │   └── logging_config.py
│       ├── models/          # 数据库模型和模式
│       ├── rag/             # RAG 实现
│       │   ├── document_loader.py
│       │   └── vector_store.py
│       └── services/        # 业务逻辑服务
│           ├── database.py
│           ├── embedding_service.py
│           └── llm_service.py
├── frontend/                # React 前端
│   └── src/
│       ├── components/      # React 组件
│       │   ├── ui/          # UI 基础组件
│       │   ├── Sidebar.tsx
│       │   ├── MainContent.tsx
│       │   ├── QADialog.tsx
│       │   └── SettingsDialog.tsx
│       └── types/           # TypeScript 类型定义
├── config.yaml              # 主配置文件
├── launcher.py              # 应用启动器
└── run.bat                  # Windows 启动脚本
```

## 快速开始

### 环境要求
- Python 3.10+
- Node.js 18+
- 运行中的 LLM 服务（LM Studio、Ollama 等）

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone <repository-url>
   cd LocalBrain
   ```

2. **后端设置**
   ```bash
   cd backend
   
   # 创建虚拟环境
   uv venv
   
   # 激活虚拟环境
   .venv\Scripts\activate  # Windows
   
   # 安装依赖
   uv pip install -r requirements.txt
   ```

3. **前端设置**
   ```bash
   cd frontend
   npm install
   ```

4. **配置**
   
   编辑 `config.yaml` 配置 LLM 和嵌入模型设置。

### 运行应用

**方式一：使用启动器（推荐）**
```bash
python launcher.py
```

**方式二：手动启动**
```bash
# 终端 1 - 后端
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 终端 2 - 前端
cd frontend
npm run dev
```

访问地址：`http://localhost:5173`

## 配置说明

主配置文件为 `config.yaml`，主要配置项如下：

### LLM 配置
```yaml
models:
  llm:
    provider: lmstudio  # 可选：ollama, openai, anthropic, custom
    providers:
      lmstudio:
        base_url: http://localhost:1234/v1
        model_name: your-model-name
```

### 嵌入模型配置
```yaml
models:
  embedding:
    provider: lmstudio
    providers:
      lmstudio:
        base_url: http://localhost:1234/v1
        model_name: text-embedding-bge-m3
        dimension: 1024
```

### 文档处理配置
```yaml
models:
  document_processing:
    chunk_size: 500
    chunk_overlap: 50
    supported_formats:
      - md
      - txt
      - pdf
```

## API 接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/documents` | GET, POST | 获取文档列表或上传文档 |
| `/api/documents/{id}` | GET, DELETE | 获取或删除单个文档 |
| `/api/search` | GET | 搜索文档 |
| `/api/qa` | POST | 智能问答 |
| `/api/categories` | GET, POST | 管理分类 |
| `/api/settings` | GET, PUT | 应用设置 |
| `/api/models` | GET | 获取可用模型列表 |

## 许可证

本项目采用 [LICENSE](../LICENSE) 文件中指定的许可条款。
