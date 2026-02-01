# -*- coding: utf-8 -*-
"""Document processing service."""

import os
import queue
import threading
import uuid
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import CharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP, OPENAI_INGEST_MODEL
from app.clients.llm_client import get_llm
from app.services.entity_service import EntityService
from app.utils.text_utils import normalize_whitespace, build_entity_id


# Upload queue and worker flag
_UPLOAD_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue()
_UPLOAD_WORKER_STARTED = False


class DocumentService:
    """Service for document processing and upload management."""

    def __init__(self, qdrant_manager):
        """Initialize document service."""
        self.qdrant = qdrant_manager
        self.entity_service = EntityService()

    def process_document(
        self,
        file_path: str,
        session_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process PDF or Markdown document and store in Qdrant.

        Args:
            file_path: Path to the document file
            session_name: Legacy session name (for backward compatibility)
            collection_name: Collection name (new paradigm)

        Returns:
            Dict with status, document name, and processing results
        """
        document_name = os.path.splitext(os.path.basename(file_path))[0]

        # Determine identifier for entity ID generation
        identifier = collection_name or session_name or "default"

        # Detect file type and use appropriate loader
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".pdf":
            loader = UnstructuredPDFLoader(file_path)
        elif file_extension in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            return {
                "status": "error",
                "document": document_name,
                "message": f"Неподдерживаемый формат файла: {file_extension}. Используйте .pdf или .md",
            }

        print(f"[DEBUG] Loading {file_extension} file: {file_path}")
        docs = loader.load()
        print(
            f"[DEBUG] Loaded {len(docs)} document(s) with total {sum(len(d.page_content) for d in docs)} characters"
        )

        splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)
        chunk_texts = [normalize_whitespace(c.page_content) for c in chunks]
        print(f"[DEBUG] Split into {len(chunks)} chunks")

        ingest_llm = get_llm(OPENAI_INGEST_MODEL, temperature=0.0)

        # Extract entities from chunks
        raw_entities_by_chunk = self.entity_service.extract_entities_parallel(
            chunk_texts, ingest_llm
        )
        entities_by_chunk: List[List[Dict[str, Any]]] = []

        for chunk_entities in raw_entities_by_chunk:
            prepared_chunk: List[Dict[str, Any]] = []
            for entity in chunk_entities:
                entity_id = build_entity_id(identifier, entity["name"], entity["type"])
                prepared = {
                    "id": entity_id,
                    "name": entity["name"],
                    "type": entity["type"],
                    "properties": dict(entity.get("properties", {})),
                }
                prepared_chunk.append(prepared)
            entities_by_chunk.append(prepared_chunk)

        # Store in Qdrant
        self.qdrant.add_chunks(
            chunk_texts,
            session_name or "default",
            document_name,
            entities_by_chunk,
            collection_name=collection_name,
        )

        total_entities = sum(len(e) for e in entities_by_chunk)

        # Update status to completed
        self.qdrant.set_document_status(
            document_name,
            "completed",
            session_name=session_name,
            collection_name=collection_name,
            chunks=len(chunks),
            entities=total_entities,
        )

        print(
            f"[DEBUG] Stored {len(chunks)} chunks with {total_entities} entities in Qdrant"
        )

        return {
            "status": "success",
            "document": document_name,
            "chunks": len(chunks),
            "entities": total_entities,
            "message": f"Готово: {document_name}",
        }

    def _ensure_upload_worker(self) -> None:
        """Ensure upload worker thread is running."""
        global _UPLOAD_WORKER_STARTED
        if _UPLOAD_WORKER_STARTED:
            return
        _UPLOAD_WORKER_STARTED = True

        def _worker() -> None:
            while True:
                job = _UPLOAD_QUEUE.get()
                session_name = job.get("session_name")
                collection_name = job.get("collection_name")
                files = job["files"]

                for file_path in files:
                    document_name = os.path.splitext(os.path.basename(file_path))[0]
                    try:
                        if not os.path.exists(file_path):
                            print(f"[WARNING] Файл не найден: {file_path}")
                            self.qdrant.set_document_status(
                                document_name,
                                "error",
                                session_name=session_name,
                                collection_name=collection_name,
                                error_message=f"Файл не найден: {file_path}",
                            )
                            continue

                        # Update status to processing
                        self.qdrant.set_document_status(
                            document_name,
                            "processing",
                            session_name=session_name,
                            collection_name=collection_name,
                        )

                        result = self.process_document(
                            file_path,
                            session_name=session_name,
                            collection_name=collection_name,
                        )

                        if result["status"] == "error":
                            print(f"[ERROR] {result['message']}")
                            self.qdrant.set_document_status(
                                document_name,
                                "error",
                                session_name=session_name,
                                collection_name=collection_name,
                                error_message=result["message"],
                            )
                        elif result["status"] == "success":
                            print(f"[SUCCESS] {result['message']}")
                            # Status already updated in process_document
                    except Exception as exc:
                        error_msg = f"Ошибка при обработке {file_path}: {exc}"
                        print(f"[ERROR] {error_msg}")
                        self.qdrant.set_document_status(
                            document_name,
                            "error",
                            session_name=session_name,
                            collection_name=collection_name,
                            error_message=str(exc),
                        )

                _UPLOAD_QUEUE.task_done()

        threading.Thread(target=_worker, name="upload-worker", daemon=True).start()

    def enqueue_upload(
        self,
        files: List[str],
        session_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> str:
        """Enqueue files for background upload.

        Args:
            files: List of file paths to upload
            session_name: Legacy session name (for backward compatibility)
            collection_name: Collection name (new paradigm)

        Returns:
            Job ID for tracking
        """
        self._ensure_upload_worker()
        job_id = str(uuid.uuid4())
        _UPLOAD_QUEUE.put({
            "job_id": job_id,
            "session_name": session_name,
            "collection_name": collection_name,
            "files": files,
        })
        return job_id
