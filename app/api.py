# -*- coding: utf-8 -*-
"""FastAPI REST API endpoints."""

import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import COLLECTION_NAME_PATTERN, UPLOAD_DIR
from app.dependencies import get_qdrant_manager, get_document_service, get_qa_service

router = APIRouter(prefix="/api")


# ── Request / Response models ──────────────────────────────────────────


class CreateCollectionRequest(BaseModel):
    name: str


class ChatRequest(BaseModel):
    question: str
    collections: List[str]


class DocumentOut(BaseModel):
    collection: str
    name: str
    status: str
    created_at: str
    chunks: int
    entities: int
    error_message: str


# ── Helpers ────────────────────────────────────────────────────────────


def _strip_legacy_marker(name: str) -> str:
    return name.rstrip(" *")


def _format_status(doc: dict) -> str:
    status = doc.get("status", "unknown")
    display = {
        "queued": "queued",
        "processing": "processing",
        "completed": "completed",
        "error": "error",
        "unknown": "unknown",
    }.get(status, status)
    return display


# ── Collections ────────────────────────────────────────────────────────


@router.get("/collections")
def list_collections():
    qdrant = get_qdrant_manager()
    collections = qdrant.get_collections()
    return [
        {
            "name": c["name"],
            "type": c["type"],
            "doc_count": c["doc_count"],
        }
        for c in collections
    ]


@router.post("/collections")
def create_collection(body: CreateCollectionRequest):
    name = body.name.strip().lower()

    if not name:
        raise HTTPException(400, "Collection name is required.")

    if not re.match(COLLECTION_NAME_PATTERN, name):
        raise HTTPException(
            400,
            "Name must be 3-50 chars: lowercase letters, digits, underscores only.",
        )

    qdrant = get_qdrant_manager()
    existing = qdrant.get_collections()
    if any(c["name"] == name for c in existing):
        raise HTTPException(409, f"Collection '{name}' already exists.")

    qdrant.create_collection(name)
    return {"status": "ok", "name": name}


@router.delete("/collections/{name}")
def delete_collection(name: str):
    qdrant = get_qdrant_manager()
    clean = _strip_legacy_marker(name)
    collection_type = qdrant.get_collection_type(clean)

    try:
        if collection_type == "legacy":
            qdrant.delete_session(clean)
        else:
            qdrant.delete_collection(clean)
    except Exception as e:
        raise HTTPException(500, f"Failed to delete collection: {e}")

    return {"status": "ok"}


# ── Documents ──────────────────────────────────────────────────────────


@router.get("/collections/{name}/documents")
def list_documents(name: str):
    qdrant = get_qdrant_manager()
    clean = _strip_legacy_marker(name)
    docs = qdrant.get_documents(collection_name=clean)

    result = []
    for doc in docs:
        created = doc.get("created_at", "")
        if created:
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                created = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                created = created[:16] if len(created) > 16 else created

        result.append(
            DocumentOut(
                collection=clean,
                name=doc["name"],
                status=_format_status(doc),
                created_at=created,
                chunks=doc.get("chunks", 0),
                entities=doc.get("entities", 0),
                error_message=doc.get("error_message", ""),
            )
        )
    return result


# ── Upload ─────────────────────────────────────────────────────────────


@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    collection: str = Form(...),
):
    qdrant = get_qdrant_manager()
    doc_service = get_document_service()
    clean = _strip_legacy_marker(collection)
    collection_type = qdrant.get_collection_type(clean)

    # Save files to persistent uploads directory
    collection_dir = Path(UPLOAD_DIR) / clean
    collection_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []

    for f in files:
        dest = collection_dir / f.filename
        with open(dest, "wb") as buf:
            shutil.copyfileobj(f.file, buf)
        saved_paths.append(str(dest))

        document_name = os.path.splitext(f.filename)[0]
        if collection_type == "legacy":
            qdrant.set_document_status(document_name, "queued", session_name=clean)
        else:
            qdrant.set_document_status(document_name, "queued", collection_name=clean)

    if collection_type == "legacy":
        job_id = doc_service.enqueue_upload(saved_paths, session_name=clean)
    else:
        job_id = doc_service.enqueue_upload(saved_paths, collection_name=clean)

    return {"status": "ok", "job_id": job_id, "file_count": len(saved_paths)}


# ── File serving ──────────────────────────────────────────────────────


@router.get("/files/{collection}/{filename:path}")
def serve_file(collection: str, filename: str):
    """Serve an uploaded file for viewing/download."""
    file_path = Path(UPLOAD_DIR) / collection / filename
    if not file_path.is_file():
        raise HTTPException(404, "File not found.")
    return FileResponse(file_path, filename=filename)


# ── Migration ──────────────────────────────────────────────────────


@router.post("/migrate-files")
async def migrate_files(
    files: List[UploadFile] = File(...),
    collection: str = Form(...),
):
    """Update file_name metadata in Qdrant for existing documents without re-indexing."""
    qdrant = get_qdrant_manager()
    clean = _strip_legacy_marker(collection)

    # Save files and build mappings
    collection_dir = Path(UPLOAD_DIR) / clean
    collection_dir.mkdir(parents=True, exist_ok=True)

    file_mappings: dict[str, str] = {}  # document_name -> file_name
    saved_files: dict[str, Path] = {}   # document_name -> file path on disk

    for f in files:
        document_name = os.path.splitext(f.filename)[0]
        dest = collection_dir / f.filename
        with open(dest, "wb") as buf:
            shutil.copyfileobj(f.file, buf)
        file_mappings[document_name] = f.filename
        saved_files[document_name] = dest

    # Update file_name in Qdrant payloads
    matched = qdrant.migrate_file_names(clean, file_mappings)
    matched_set = set(matched)

    # Delete files that had no matching documents
    unmatched_deleted = []
    for doc_name, path in saved_files.items():
        if doc_name not in matched_set:
            path.unlink(missing_ok=True)
            unmatched_deleted.append(file_mappings[doc_name])

    return {
        "matched": [file_mappings[m] for m in matched],
        "unmatched_deleted": unmatched_deleted,
    }


# ── Chat ───────────────────────────────────────────────────────────────


@router.post("/chat")
def chat(body: ChatRequest):
    if not body.collections:
        raise HTTPException(400, "Select at least one collection.")
    if not body.question.strip():
        raise HTTPException(400, "Question must not be empty.")

    qa = get_qa_service()
    clean = [_strip_legacy_marker(c) for c in body.collections]
    result = qa.answer_question_with_sources(body.question, clean)
    return {"answer": result["answer"], "sources": result["sources"]}
