# -*- coding: utf-8 -*-
"""Application dependencies and shared instances."""

from qdrant_manager import QdrantManager
from app.config import QDRANT_URL
from app.services.document_service import DocumentService
from app.services.qa_service import QAService


# Initialize shared instances
qdrant_manager = QdrantManager(url=QDRANT_URL)
document_service = DocumentService(qdrant_manager)
qa_service = QAService(qdrant_manager)


def get_qdrant_manager() -> QdrantManager:
    """Get Qdrant manager instance."""
    return qdrant_manager


def get_document_service() -> DocumentService:
    """Get document service instance."""
    return document_service


def get_qa_service() -> QAService:
    """Get QA service instance."""
    return qa_service
