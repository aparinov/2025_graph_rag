# -*- coding: utf-8 -*-

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone


class QdrantManager:
    def __init__(self, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=url)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.collection_name = "medical_documents"
        self.metadata_collection = "document_metadata"
        self._ensure_collection()
        self._ensure_metadata_collection()

    def _ensure_collection(self):
        """Create collection if not exists"""
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    def _ensure_metadata_collection(self):
        """Create metadata collection for document status tracking"""
        collections = self.client.get_collections().collections
        if not any(c.name == self.metadata_collection for c in collections):
            # Metadata collection doesn't need vectors, just use a dummy vector
            self.client.create_collection(
                collection_name=self.metadata_collection,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )

    def add_chunks(
        self,
        chunks: List[str],
        session_name: str,
        document_name: str,
        entities_by_chunk: List[List[Dict]],
    ):
        """Store chunks with embeddings and metadata"""
        points = []

        for i, (chunk, entities) in enumerate(zip(chunks, entities_by_chunk)):
            vector = self.embeddings.embed_query(chunk)

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "session_name": session_name,
                    "document_name": document_name,
                    "chunk_id": i,
                    "entities": [
                        {"name": e["name"], "type": e["type"]} for e in entities
                    ],
                    "entity_count": len(entities),
                },
            )
            points.append(point)

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, session_name: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks within session"""
        query_vector = self.embeddings.embed_query(query)

        search_filter = Filter(
            must=[
                FieldCondition(
                    key="session_name", match=MatchValue(value=session_name)
                )
            ]
        )

        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=k,
                with_payload=True,
                with_vectors=False,
            )
        elif hasattr(self.client, "query_points"):
            try:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=k,
                    with_payload=True,
                    with_vectors=False,
                    query_filter=search_filter,
                )
            except TypeError:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=k,
                    with_payload=True,
                    with_vectors=False,
                    filter=search_filter,
                )
            results = response.points if hasattr(response, "points") else response
        elif hasattr(self.client, "search_points"):
            try:
                response = self.client.search_points(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=k,
                    with_payload=True,
                    with_vectors=False,
                    query_filter=search_filter,
                )
            except TypeError:
                response = self.client.search_points(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=k,
                    with_payload=True,
                    with_vectors=False,
                    filter=search_filter,
                )
            results = response.points if hasattr(response, "points") else response
        else:
            raise AttributeError(
                "QdrantClient has no search/query_points/search_points method"
            )

        return [
            {
                "text": hit.payload["text"],
                "document_name": hit.payload["document_name"],
                "chunk_id": hit.payload["chunk_id"],
                "entities": hit.payload.get("entities", []),
                "score": hit.score,
            }
            for hit in results
        ]

    def get_sessions(self) -> List[str]:
        """Get all unique session names"""
        sessions = set()
        offset = None

        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=["session_name"],
                with_vectors=False,
            )

            for point in results:
                sessions.add(point.payload["session_name"])

            if next_offset is None:
                break
            offset = next_offset

        return sorted(list(sessions))

    def get_documents(self, session_name: str) -> List[Dict[str, Any]]:
        """Get documents in session with stats and status"""
        offset = None
        documents = []

        # Get all metadata entries for this session
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.metadata_collection,
                limit=100,
                offset=offset,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="session_name", match=MatchValue(value=session_name)
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=False,
            )

            for point in results:
                documents.append({
                    "name": point.payload["document_name"],
                    "status": point.payload.get("status", "unknown"),
                    "chunks": point.payload.get("chunks", 0),
                    "entities": point.payload.get("entities", 0),
                    "created_at": point.payload.get("created_at", ""),
                    "updated_at": point.payload.get("updated_at", ""),
                    "error_message": point.payload.get("error_message", ""),
                })

            if next_offset is None:
                break
            offset = next_offset

        # Sort by created_at descending (newest first)
        documents.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return documents

    def delete_session(self, session_name: str):
        """Delete all points in a session"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="session_name", match=MatchValue(value=session_name)
                    )
                ]
            ),
        )
        # Also delete metadata
        self.client.delete(
            collection_name=self.metadata_collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="session_name", match=MatchValue(value=session_name)
                    )
                ]
            ),
        )

    def _get_document_id(self, session_name: str, document_name: str) -> str:
        """Generate unique UUID for document metadata based on session and document name"""
        # Use UUID5 (namespace-based) to generate deterministic UUID from session+doc name
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # DNS namespace
        unique_string = f"{session_name}::{document_name}"
        return str(uuid.uuid5(namespace, unique_string))

    def set_document_status(
        self,
        session_name: str,
        document_name: str,
        status: str,
        chunks: int = 0,
        entities: int = 0,
        error_message: str = "",
    ):
        """Create or update document metadata with status

        Status values: "queued", "processing", "completed", "error"
        """
        doc_id = self._get_document_id(session_name, document_name)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Get existing metadata if any
        try:
            existing = self.client.retrieve(
                collection_name=self.metadata_collection,
                ids=[doc_id],
            )
            created_at = existing[0].payload.get("created_at", timestamp) if existing else timestamp
        except:
            created_at = timestamp

        point = PointStruct(
            id=doc_id,
            vector=[0.0],  # Dummy vector
            payload={
                "session_name": session_name,
                "document_name": document_name,
                "status": status,
                "chunks": chunks,
                "entities": entities,
                "created_at": created_at,
                "updated_at": timestamp,
                "error_message": error_message,
            },
        )

        self.client.upsert(collection_name=self.metadata_collection, points=[point])

    def get_document_metadata(
        self, session_name: str, document_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document"""
        doc_id = self._get_document_id(session_name, document_name)

        try:
            results = self.client.retrieve(
                collection_name=self.metadata_collection,
                ids=[doc_id],
            )
            if results:
                return results[0].payload
        except:
            pass

        return None
