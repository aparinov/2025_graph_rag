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
        self.collection_name = "medical_documents"  # Legacy collection
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

    def create_collection(self, collection_name: str) -> None:
        """Create a new collection for storing documents"""
        collections = self.client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    def get_collections(self) -> List[Dict[str, Any]]:
        """Get all collections (both legacy sessions and new collections)

        Returns:
            List of dicts with keys: name, type (legacy/collection), doc_count
        """
        all_qdrant_collections = self.client.get_collections().collections
        collection_names = {
            c.name for c in all_qdrant_collections
            if c.name != self.metadata_collection
        }

        result = []

        # Get legacy sessions from medical_documents collection
        if self.collection_name in collection_names:
            legacy_sessions = self._get_legacy_sessions()
            for session in legacy_sessions:
                doc_count = self._count_documents(self.collection_name, session_name=session)
                result.append({
                    "name": session,
                    "type": "legacy",
                    "doc_count": doc_count,
                })

        # Get new collections
        for coll_name in collection_names:
            if coll_name != self.collection_name:  # Skip medical_documents
                doc_count = self._count_documents(coll_name)
                result.append({
                    "name": coll_name,
                    "type": "collection",
                    "doc_count": doc_count,
                })

        return sorted(result, key=lambda x: x["name"])

    def _get_legacy_sessions(self) -> List[str]:
        """Get all unique session names from legacy collection"""
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

    def _count_documents(self, collection_name: str, session_name: Optional[str] = None) -> int:
        """Count unique documents in a collection or legacy session"""
        offset = None
        documents = set()

        filter_condition = None
        if session_name:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="session_name", match=MatchValue(value=session_name)
                    )
                ]
            )

        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.metadata_collection,
                limit=100,
                offset=offset,
                scroll_filter=filter_condition,
                with_payload=["document_name", "collection_name", "session_name"],
                with_vectors=False,
            )

            for point in results:
                # For legacy, match by session_name
                if session_name and point.payload.get("session_name") == session_name:
                    documents.add(point.payload["document_name"])
                # For new collections, match by collection_name
                elif not session_name and point.payload.get("collection_name") == collection_name:
                    documents.add(point.payload["document_name"])

            if next_offset is None:
                break
            offset = next_offset

        return len(documents)

    def get_collection_type(self, collection_name: str) -> str:
        """Determine if collection is legacy session or new collection

        Returns:
            "legacy" if it's a session in medical_documents, "collection" otherwise
        """
        legacy_sessions = self._get_legacy_sessions()
        return "legacy" if collection_name in legacy_sessions else "collection"

    def add_chunks(
        self,
        chunks: List[str],
        session_name: str,
        document_name: str,
        entities_by_chunk: List[List[Dict]],
        collection_name: Optional[str] = None,
        file_name: Optional[str] = None,
    ):
        """Store chunks with embeddings and metadata

        Args:
            chunks: List of text chunks
            session_name: Legacy session name (for backward compatibility)
            document_name: Name of the document
            entities_by_chunk: Entities extracted from each chunk
            collection_name: Target collection (None = use legacy medical_documents)
            file_name: Original filename with extension (for download links)
        """
        # Determine target collection
        target_collection = collection_name or self.collection_name

        # Ensure collection exists for new collections
        if collection_name and collection_name != self.collection_name:
            self.create_collection(collection_name)

        points = []

        for i, (chunk, entities) in enumerate(zip(chunks, entities_by_chunk)):
            vector = self.embeddings.embed_query(chunk)

            payload = {
                "text": chunk,
                "document_name": document_name,
                "chunk_id": i,
                "entities": [
                    {"name": e["name"], "type": e["type"]} for e in entities
                ],
                "entity_count": len(entities),
            }

            if file_name:
                payload["file_name"] = file_name

            # Add session_name for legacy collections
            if target_collection == self.collection_name:
                payload["session_name"] = session_name
            else:
                payload["collection_name"] = collection_name

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
            points.append(point)

        self.client.upsert(collection_name=target_collection, points=points)

    def search(self, query: str, session_name: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks within session (legacy method for backward compatibility)"""
        return self.search_multi_collection(query, [session_name], k)

    def search_multi_collection(
        self, query: str, collection_names: List[str], k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks across multiple collections/sessions

        Args:
            query: Search query
            collection_names: List of collection names (can be legacy sessions or new collections)
            k: Number of results per collection

        Returns:
            List of search results with collection name and document citations
        """
        query_vector = self.embeddings.embed_query(query)
        all_results = []

        for coll_name in collection_names:
            collection_type = self.get_collection_type(coll_name)

            if collection_type == "legacy":
                # Search in medical_documents with session filter
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="session_name", match=MatchValue(value=coll_name)
                        )
                    ]
                )
                target_collection = self.collection_name
            else:
                # Search in new collection (no filter needed)
                search_filter = None
                target_collection = coll_name

            # Perform search with appropriate API method
            if hasattr(self.client, "search"):
                results = self.client.search(
                    collection_name=target_collection,
                    query_vector=query_vector,
                    query_filter=search_filter,
                    limit=k,
                    with_payload=True,
                    with_vectors=False,
                )
            elif hasattr(self.client, "query_points"):
                try:
                    response = self.client.query_points(
                        collection_name=target_collection,
                        query=query_vector,
                        limit=k,
                        with_payload=True,
                        with_vectors=False,
                        query_filter=search_filter,
                    )
                except TypeError:
                    response = self.client.query_points(
                        collection_name=target_collection,
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
                        collection_name=target_collection,
                        query_vector=query_vector,
                        limit=k,
                        with_payload=True,
                        with_vectors=False,
                        query_filter=search_filter,
                    )
                except TypeError:
                    response = self.client.search_points(
                        collection_name=target_collection,
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

            # Add collection name to each result
            for hit in results:
                all_results.append({
                    "text": hit.payload["text"],
                    "document_name": hit.payload["document_name"],
                    "collection_name": coll_name,
                    "chunk_id": hit.payload["chunk_id"],
                    "entities": hit.payload.get("entities", []),
                    "score": hit.score,
                    "file_name": hit.payload.get("file_name", ""),
                })

        # Sort all results by score and return top k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:k * len(collection_names)]

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

    def get_documents(
        self, session_name: Optional[str] = None, collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get documents in session or collection with stats and status

        Args:
            session_name: Legacy session name (for backward compatibility)
            collection_name: Collection name

        Returns:
            List of document metadata
        """
        # Determine which field to filter by
        filter_key = None
        filter_value = None

        if collection_name:
            collection_type = self.get_collection_type(collection_name)
            if collection_type == "legacy":
                filter_key = "session_name"
                filter_value = collection_name
            else:
                filter_key = "collection_name"
                filter_value = collection_name
        elif session_name:
            # Backward compatibility
            filter_key = "session_name"
            filter_value = session_name

        if not filter_key or not filter_value:
            return []

        offset = None
        documents = []

        # Get all metadata entries
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.metadata_collection,
                limit=100,
                offset=offset,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key=filter_key, match=MatchValue(value=filter_value)
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
        """Delete all points in a legacy session"""
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

    def delete_collection(self, collection_name: str):
        """Delete a collection and its metadata

        Args:
            collection_name: Name of collection to delete
        """
        collection_type = self.get_collection_type(collection_name)

        if collection_type == "legacy":
            # Use delete_session for legacy collections
            self.delete_session(collection_name)
        else:
            # Delete the entire collection
            try:
                self.client.delete_collection(collection_name=collection_name)
            except Exception as e:
                print(f"[WARNING] Failed to delete collection {collection_name}: {e}")

            # Delete metadata
            self.client.delete(
                collection_name=self.metadata_collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="collection_name", match=MatchValue(value=collection_name)
                        )
                    ]
                ),
            )

    def _get_document_id(
        self,
        document_name: str,
        session_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> str:
        """Generate unique UUID for document metadata

        Args:
            document_name: Name of the document
            session_name: Legacy session name (for backward compatibility)
            collection_name: Collection name

        Returns:
            Deterministic UUID based on collection/session and document name
        """
        # Use UUID5 (namespace-based) to generate deterministic UUID
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # DNS namespace
        identifier = collection_name or session_name or "default"
        unique_string = f"{identifier}::{document_name}"
        return str(uuid.uuid5(namespace, unique_string))

    def set_document_status(
        self,
        document_name: str,
        status: str,
        session_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        chunks: int = 0,
        entities: int = 0,
        error_message: str = "",
    ):
        """Create or update document metadata with status

        Args:
            document_name: Name of the document
            status: Status value ("queued", "processing", "completed", "error")
            session_name: Legacy session name (for backward compatibility)
            collection_name: Collection name
            chunks: Number of chunks
            entities: Number of entities
            error_message: Error message if status is "error"
        """
        doc_id = self._get_document_id(document_name, session_name, collection_name)
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

        payload = {
            "document_name": document_name,
            "status": status,
            "chunks": chunks,
            "entities": entities,
            "created_at": created_at,
            "updated_at": timestamp,
            "error_message": error_message,
        }

        # Add session_name or collection_name
        if session_name:
            payload["session_name"] = session_name
        if collection_name:
            payload["collection_name"] = collection_name

        point = PointStruct(
            id=doc_id,
            vector=[0.0],  # Dummy vector
            payload=payload,
        )

        self.client.upsert(collection_name=self.metadata_collection, points=[point])

    def get_document_metadata(
        self,
        document_name: str,
        session_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document

        Args:
            document_name: Name of the document
            session_name: Legacy session name (for backward compatibility)
            collection_name: Collection name

        Returns:
            Document metadata or None
        """
        doc_id = self._get_document_id(document_name, session_name, collection_name)

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
