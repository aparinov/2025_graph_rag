# -*- coding: utf-8 -*-
"""Application configuration and environment variables."""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Proxy Configuration
PROXY_URL = (
    os.getenv("PROXY_URL") or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
)

# Set proxy environment variables for OpenAI client
if PROXY_URL:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL
    # Exclude local services from proxy
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,qdrant,neo4j"
    print(f"[DEBUG] Proxy configured: {PROXY_URL}")
    print(f"[DEBUG] NO_PROXY: {os.environ['NO_PROXY']}")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "2000"))

# Processing Configuration
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# OpenAI Models
OPENAI_INGEST_MODEL = os.getenv("OPENAI_INGEST_MODEL", "gpt-4o-mini")
OPENAI_QA_MODEL = os.getenv("OPENAI_QA_MODEL", "gpt-4o-mini")

# SSL Configuration
SSL_CERT_FILE = os.getenv("SSL_CERT_FILE") or True

# Collection Configuration
DEFAULT_COLLECTION = "medical_documents"  # Legacy collection for backward compatibility
COLLECTION_NAME_PATTERN = r"^[a-z0-9_]{3,50}$"  # Collection name validation pattern

# Upload storage
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads"))
