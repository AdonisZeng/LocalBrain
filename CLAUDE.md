# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LocalBrain is a **local personal knowledge base management system** with AI-powered Q&A (RAG) capabilities. Documents are indexed locally using ChromaDB vector store, with retrieval augmented generation for answering questions.

## Tech Stack

- **Backend**: FastAPI, LangChain, ChromaDB, SQLAlchemy (SQLite)
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS v4, Radix UI
- **Python**: D:\Software\uv\envs\trae_cn\Scripts\python.exe

## Running the Application

```bash
# Option 1: Use the launcher (recommended)
python launcher.py

# Option 2: Manual startup
# Terminal 1 - Backend (from project root)
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend (from project root)
cd frontend && npm run dev
```

Access frontend at `http://localhost:5173`, API at `http://localhost:8000`.

## Backend Architecture

### Configuration
- Main config: `config.yaml` at project root
- Pydantic models in `backend/app/core/config.py` define the schema
- `ConfigManager` in `config_manager.py` handles runtime config updates
- Config changes can be reloaded without restart via the settings API

### Core Services (singleton pattern)
- `get_llm_service()` - LLM provider abstraction (LM Studio, Ollama, OpenAI, Anthropic, Custom)
- `get_embedding_service()` - Embedding provider abstraction (HuggingFace, LM Studio, Ollama, OpenAI)
- `get_vector_store()` - ChromaDB vector store with parent-child document retrieval

### RAG Pipeline
1. `DocumentLoader` (`rag/document_loader.py`) - Loads PDFs, Markdown, TXT, DOCX
2. `TextSplitter` - Chunks documents by type (markdown headers, PDF semantic, recursive text)
3. `VectorStore` - Stores embeddings in ChromaDB with parent-child hierarchy
4. `ask_question` (`api/qa.py`) - Retrieves relevant docs, builds context prompt, calls LLM

### API Routes
| Route | File | Purpose |
|-------|------|---------|
| `/api/documents` | `api/documents.py` | Upload, list, delete documents |
| `/api/search` | `api/search.py` | Semantic/keyword/hybrid search |
| `/api/qa` | `api/qa.py` | RAG-powered question answering |
| `/api/categories` | `api/categories.py` | Document categorization |
| `/api/settings` | `api/settings.py` | Runtime config reload |
| `/api/models` | `api/models.py` | List available LLM/embedding models |

### Database Models
SQLite at `./data/localbrain.db`:
- `Document` - file_path (unique), title, status, category_id
- `Category` - name (unique), color
- `Link` - wikilinks between documents

### Data Directories
- `./data/` - SQLite DB, ChromaDB vector store, imported documents
- `./logs/` - Structured JSON logs via structlog

## Frontend Architecture

- **State management**: Local React state with useState/useCallback
- **API layer**: `src/lib/api.ts` - typed API client for all endpoints
- **Chat history**: Stored in localStorage (not backend)
- **Components**: Sidebar, MainContent, QADialog, SettingsDialog
- **UI primitives**: Radix UI dialogs, Tailwind CSS v4 styling

## Key Implementation Details

### Parent Document Retrieval
VectorStore implements parent-child chunk hierarchy:
- Child chunks (~400 chars) stored for precise retrieval
- Parent chunks (~2000 chars) retrieved and used as context
- Reduces token usage while maintaining document coherence

### Compression
After similarity search, results are filtered by:
- Score threshold (default 0.5)
- Max context chars (default 4000)
- Preserves only the most relevant passages

### Hybrid Search
Configurable RRF (Reciprocal Rank Fusion) combining:
- Semantic search (vector similarity)
- Keyword search (BM25)
- Weights configurable in `config.yaml`

### Rate Limiting
Uses slowapi with configurable limits:
- Default: 60 requests/minute for local, 120 for LAN
- Configurable via `config.yaml` security section
