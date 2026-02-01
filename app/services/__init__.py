# -*- coding: utf-8 -*-
"""Service modules."""

from app.services.document_service import DocumentService
from app.services.entity_service import EntityService
from app.services.relation_service import RelationService
from app.services.qa_service import QAService

__all__ = ["DocumentService", "EntityService", "RelationService", "QAService"]
