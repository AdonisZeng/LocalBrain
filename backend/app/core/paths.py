"""
Path resolution utility for LocalBrain.

Handles both development mode (running from source) and
frozen mode (running as packaged .exe).

All path access should go through this module to ensure
correct behavior in both environments.
"""

import sys
import os
from pathlib import Path
from typing import Optional


def is_frozen() -> bool:
    """Check if running as a frozen (packaged) executable."""
    return getattr(sys, 'frozen', False)


def get_app_dir() -> Path:
    """
    Get the application directory.

    - In frozen mode: directory containing the .exe
    - In dev mode: the project root (parent of backend/app)
    """
    if is_frozen():
        # Running as packaged exe - exe is in the app dir
        return Path(sys.executable).parent
    else:
        # Running in development - project root is 4 levels up from this file
        # backend/app/core/paths.py -> backend/app/core -> backend/app -> backend -> project root
        return Path(__file__).parent.parent.parent.parent


def get_data_dir() -> Path:
    """Get the data directory, creating it if it doesn't exist."""
    data_dir = get_app_dir() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_logs_dir() -> Path:
    """Get the logs directory, creating it if it doesn't exist."""
    logs_dir = get_app_dir() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_config_path() -> Path:
    """Get the path to config.yaml."""
    return get_app_dir() / "config.yaml"


def get_chroma_db_dir() -> Path:
    """Get the ChromaDB data directory."""
    chroma_dir = get_data_dir() / "chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir


def get_documents_dir() -> Path:
    """Get the documents storage directory."""
    docs_dir = get_data_dir() / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    return docs_dir


def get_database_path() -> Path:
    """Get the SQLite database path."""
    return get_data_dir() / "localbrain.db"


def get_config_backups_dir() -> Path:
    """Get the config backups directory."""
    backups_dir = get_data_dir() / "config_backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    return backups_dir


def get_frontend_dist_dir() -> Optional[Path]:
    """
    Get the frontend dist directory (for serving static files in production).

    In dev mode returns None (frontend runs on separate dev server).
    In frozen mode returns the bundled frontend dist.
    """
    if is_frozen():
        app_dir = get_app_dir()
        possible_dirs = [
            app_dir / "_internal" / "frontend_dist",
            app_dir / "_internal" / "frontend" / "dist",
            app_dir / "frontend_dist",
            app_dir / "frontend" / "dist",
        ]
        for dist_dir in possible_dirs:
            if dist_dir.exists():
                return dist_dir
    return None
