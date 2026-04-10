import mimetypes
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

import logging
import sys
import webbrowser
from pathlib import Path
from threading import Timer
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api import documents, search, qa, categories, settings, models
from app.services.database import db
from app.models.database import Base
from app.core.logging_config import setup_logging, get_logger
from app import __version__
from app.core.config import get_config
from app.core.paths import get_logs_dir, get_frontend_dist_dir, get_app_dir

# Ensure data and logs directories exist before anything else
logs_dir = get_logs_dir()

setup_logging(log_level="INFO")
logger = get_logger("main")

db.create_tables()
logger.info("Database tables initialized")

# Check if frontend dist is available (production mode)
frontend_dist = get_frontend_dist_dir()
if frontend_dist:
    logger.info(f"Frontend dist found at {frontend_dist}, will serve static files")
else:
    logger.info("Frontend dist not found, API-only mode")


def create_app() -> FastAPI:
    app = FastAPI(
        title="LocalBrain API",
        description="本地知识库管理系统 API",
        version=__version__,
    )

    # Read rate limit config
    config = get_config()
    rate_limit_config = config.security.rate_limit
    requests_per_minute = rate_limit_config.get("requests_per_minute", 60)

    # Create limiter
    limiter = Limiter(key_func=get_remote_address, default_limits=[f"{requests_per_minute}/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS origins
    if frontend_dist:
        # In production (packaged), only allow the app itself
        origins = [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
    else:
        # In development, allow Vite dev server
        origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "http://localhost:5175",
            "http://127.0.0.1:5175",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
    app.include_router(search.router, prefix="/api/search", tags=["search"])
    app.include_router(qa.router, prefix="/api/qa", tags=["qa"])
    app.include_router(categories.router, prefix="/api/categories", tags=["categories"])
    app.include_router(settings.router, prefix="/api/settings", tags=["settings"])
    app.include_router(models.router, prefix="/api/models", tags=["models"])

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    # Serve frontend static files in production mode
    if frontend_dist and frontend_dist.exists():
        # Serve index.html for the root
        @app.get("/")
        async def serve_index():
            return FileResponse(str(frontend_dist / "index.html"))

        # Mount assets folder at /assets
        assets_dir = frontend_dist / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir), html=False), name="assets")

        # Serve other static files (favicon, icons) at root
        @app.get("/favicon.svg")
        async def serve_favicon():
            favicon_path = frontend_dist / "favicon.svg"
            if favicon_path.exists():
                return FileResponse(str(favicon_path))
            return {"error": "Not found"}

        @app.get("/icons.svg")
        async def serve_icons():
            icons_path = frontend_dist / "icons.svg"
            if icons_path.exists():
                return FileResponse(str(icons_path))
            return {"error": "Not found"}

        # Catch-all for SPA routing - serve index.html for non-API routes
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            # Skip API routes
            if full_path.startswith("api/"):
                return JSONResponse({"error": "Not found"}, status_code=404)
            # Serve index.html for SPA routing
            return FileResponse(str(frontend_dist / "index.html"))

    else:
        # API-only mode (development)
        @app.get("/")
        async def root():
            return {"message": "LocalBrain API", "version": __version__}

    return app


app = create_app()


def open_browser():
    """Open the browser after a short delay."""
    webbrowser.open("http://localhost:8000")


if __name__ == "__main__":
    import uvicorn

    # Open browser after server starts
    Timer(1.5, open_browser).start()

    uvicorn.run(app, host="0.0.0.0", port=8000)
