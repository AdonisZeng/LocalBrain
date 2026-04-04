# LocalBrain

[中文](./doc/README_CN.md) | English

## Overview

LocalBrain is a **local personal knowledge base management system** with AI-powered Q&A capabilities. It allows you to organize, search, and interact with your documents using natural language, all running entirely on your local machine.

## Features

### 📄 Document Management
- **Multi-format Support**: Import PDF, Markdown, and plain text files
- **Automatic Processing**: Documents are automatically parsed, chunked, and indexed
- **Category Organization**: Organize documents into customizable categories
- **Status Tracking**: Monitor processing status (pending, processing, completed, failed)

### 🔍 Intelligent Search
- **Semantic Search**: Find relevant content using natural language queries
- **Keyword Search**: Traditional text-based search
- **Hybrid Search**: Combine semantic and keyword approaches for better results

### 🤖 AI-Powered Q&A
- **RAG (Retrieval-Augmented Generation)**: Ask questions about your documents
- **Source Attribution**: Answers include references to source documents
- **Multiple LLM Support**: Compatible with various language models

### 🔧 Flexible Model Configuration
- **LLM Providers**: Support for LM Studio, Ollama, OpenAI, Anthropic, and custom endpoints
- **Embedding Models**: Configurable embedding services (LM Studio, HuggingFace)
- **Local-First**: Run completely offline with local models

### 📁 Import & Export
- **Import**: Support for Obsidian, Notion, and Logseq formats
- **Export**: Export to Obsidian, JSON, and Markdown formats

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | FastAPI, LangChain, ChromaDB, SQLAlchemy |
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS |
| **AI/ML** | LangChain, Sentence Transformers, Multiple LLM providers |
| **Database** | SQLite (via SQLAlchemy) |
| **Vector Store** | ChromaDB |

## Project Structure

```
LocalBrain/
├── backend/                 # FastAPI backend
│   └── app/
│       ├── api/             # API route handlers
│       │   ├── documents.py # Document management
│       │   ├── search.py    # Search endpoints
│       │   ├── qa.py        # Q&A endpoints
│       │   ├── categories.py
│       │   ├── settings.py
│       │   └── models.py
│       ├── core/            # Core configuration
│       │   ├── config.py
│       │   ├── config_manager.py
│       │   └── logging_config.py
│       ├── models/          # Database models & schemas
│       ├── rag/             # RAG implementation
│       │   ├── document_loader.py
│       │   └── vector_store.py
│       └── services/        # Business logic services
│           ├── database.py
│           ├── embedding_service.py
│           └── llm_service.py
├── frontend/                # React frontend
│   └── src/
│       ├── components/      # React components
│       │   ├── ui/          # UI primitives
│       │   ├── Sidebar.tsx
│       │   ├── MainContent.tsx
│       │   ├── QADialog.tsx
│       │   └── SettingsDialog.tsx
│       └── types/           # TypeScript types
├── config.yaml              # Main configuration file
├── launcher.py              # Application launcher
└── run.bat                  # Windows startup script
```

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- A running LLM service (LM Studio, Ollama, etc.)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd LocalBrain
   ```

2. **Backend Setup**
   ```bash
   cd backend
   
   # Create virtual environment
   uv venv
   
   # Activate virtual environment
   .venv\Scripts\activate  # Windows
   
   # Install dependencies
   uv pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

4. **Configuration**
   
   Edit `config.yaml` to configure your LLM and embedding model settings.

### Running the Application

**Option 1: Using the launcher (recommended)**
```bash
python launcher.py
```

**Option 2: Manual startup**
```bash
# Terminal 1 - Backend
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

Access the application at `http://localhost:5173`

## Configuration

The main configuration file is `config.yaml`. Key settings include:

### LLM Configuration
```yaml
models:
  llm:
    provider: lmstudio  # or ollama, openai, anthropic, custom
    providers:
      lmstudio:
        base_url: http://localhost:1234/v1
        model_name: your-model-name
```

### Embedding Configuration
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

### Document Processing
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

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/documents` | GET, POST | List or upload documents |
| `/api/documents/{id}` | GET, DELETE | Get or delete a document |
| `/api/search` | GET | Search documents |
| `/api/qa` | POST | Ask questions |
| `/api/categories` | GET, POST | Manage categories |
| `/api/settings` | GET, PUT | Application settings |
| `/api/models` | GET | List available models |

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
