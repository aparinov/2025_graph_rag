# -*- coding: utf-8 -*-
"""Main entry point for the Graph RAG application."""

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api import router as api_router

app = FastAPI(title="MedGraph RAG")
app.include_router(api_router)

# Serve frontend static files
_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend" / "dist"

if _FRONTEND_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        """SPA fallback – serve index.html for any non-API route."""
        file = _FRONTEND_DIR / full_path
        if file.is_file():
            return FileResponse(file)
        return FileResponse(_FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
