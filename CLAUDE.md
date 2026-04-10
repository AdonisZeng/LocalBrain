# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LocalBrain is a local personal knowledge base system with AI-powered Q&A. It runs fully locally — FastAPI backend on port 8000, React 19 + Vite frontend on port 5173, SQLite for metadata, ChromaDB for vector storage.

## Commands

### Running the Application

```bash
# Recommended: auto-manages both servers and opens browser
python launcher.py

# Manual: backend only
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Manual: frontend only
cd frontend
npm run dev
```

### Frontend

```bash
cd frontend
npm install          # Install dependencies
npm run dev          # Dev server (port 5173)
npm run build        # TypeScript check + Vite bundle
npm run lint         # ESLint
npm run preview      # Preview production build
```

### Backend

```bash
cd backend
# Python interpreter: D:\Development\Python\Nexus\.venv\Scripts\python.exe
uv pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

No test framework is currently configured.

## Architecture

### High-Level Structure

```
launcher.py              # Orchestrates both servers, monitors browser connection
config.yaml              # Single source of truth for all runtime config
backend/app/
  main.py                # FastAPI app, CORS, rate limiting, router registration
  api/                   # 6 route modules (documents, search, qa, categories, settings, models)
  core/                  # Config loading, interfaces, EventBus for hot-reload
  models/                # SQLAlchemy ORM (database.py) + Pydantic schemas (schemas.py)
  providers/             # Pluggable LLM / embedding / vectorstore implementations
  rag/                   # Document loading, chunking, vector store wrapper
  services/              # Business logic; BaseModelService handles hot-reload lifecycle
frontend/src/
  App.tsx                # Main state: documents, categories, chat history (localStorage)
  components/            # Sidebar, MainContent, QADialog, SettingsDialog
  lib/api.ts             # All REST API calls
```

### Provider Registry Pattern

All AI backend integrations use a decorator-based registry:

```python
@register_llm_provider("ollama")
class OllamaProvider(BaseLLMProvider): ...
```

Supported LLM providers: `openai`, `ollama`, `lmstudio`, `anthropic`, `custom`  
Supported embedding providers: `huggingface`, `openai`, `ollama`, `lmstudio`, `custom`  
Vector store: ChromaDB only (currently)

All provider interfaces are defined in `backend/app/core/interfaces.py`.

### Configuration Hot-Reload

`config.yaml` is the live config. When settings are changed via the `/api/settings` endpoints:
1. The API writes to `config.yaml`
2. `EventBus` emits a typed event (`LLM_CONFIG_CHANGED`, `EMBEDDING_CONFIG_CHANGED`, etc.)
3. `BaseModelService` subclasses (`LLMService`, `EmbeddingService`) subscribe to events and reset their instances
4. Next call to `get_instance()` triggers lazy re-initialization with new config

### RAG Pipeline Flow

`/api/qa` POST → `LLMService.get_instance()` + `EmbeddingService.get_instance()` → `VectorStore` retrieval (semantic/keyword/hybrid with optional RRF reranking) → LangChain chain → streamed response

Document ingestion: upload → `DocumentLoader` (PDF/MD/TXT/DOCX) → format-specific `TextSplitter` → embed → persist to ChromaDB + SQLite status update.

### Data Layer

- **SQLite** (`localbrain.db`): `documents`, `categories`, `links` tables via SQLAlchemy ORM
- **ChromaDB** (`data/chroma_db/`): Vector embeddings with persistence
- **Config**: `config.yaml` (YAML, runtime-mutable via API)
- **Documents**: stored under `data/documents/`

### Frontend State

`App.tsx` manages all global state. Chat history is persisted to `localStorage`. The frontend proxies API requests to `http://localhost:8000` via Vite dev config.
