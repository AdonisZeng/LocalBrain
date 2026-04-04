from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api import documents, search, qa, categories, settings, models
from app.services.database import db
from app.models.database import Base
from app.core.logging_config import setup_logging, get_logger
from app.core.config import get_config

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

setup_logging(log_dir=str(LOGS_DIR), log_level="INFO")
logger = get_logger("main")

db.create_tables()
logger.info("Database tables initialized")
logger.info(f"Application starting - data_dir={DATA_DIR}, logs_dir={LOGS_DIR}")


def create_app() -> FastAPI:
    app = FastAPI(
        title="LocalBrain API",
        description="本地知识库管理系统 API",
        version="1.0.0",
    )

    # 读取限流配置
    config = get_config()
    rate_limit_config = config.security.rate_limit
    requests_per_minute = rate_limit_config.get("requests_per_minute", 60)

    # 创建限流器
    limiter = Limiter(key_func=get_remote_address, default_limits=[f"{requests_per_minute}/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "http://localhost:5175",
            "http://127.0.0.1:5175",
        ],
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

    @app.get("/")
    async def root():
        return {"message": "LocalBrain API", "version": "1.0.0"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
