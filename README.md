# Graph RAG Application

Medical knowledge graph and RAG system for processing pharmaceutical instructions and medical documents.

## Project Structure

```
2025_graph_rag/
├── app/                      # Main application package
│   ├── clients/              # Client management
│   │   ├── http_client.py    # HTTP client configuration
│   │   └── llm_client.py     # LLM client management
│   ├── config.py             # Application configuration
│   ├── dependencies.py       # Shared dependencies
│   ├── models/               # Data models
│   │   └── prompts.py        # LLM prompt templates
│   ├── services/             # Business logic services
│   │   ├── document_service.py   # Document processing
│   │   ├── entity_service.py     # Entity extraction
│   │   ├── qa_service.py         # Q&A functionality
│   │   └── relation_service.py   # Relation extraction
│   ├── ui/                   # User interface
│   │   └── gradio_app.py     # Gradio web interface
│   └── utils/                # Utility functions
│       └── text_utils.py     # Text processing utilities
├── qdrant_manager.py         # Qdrant vector database manager
├── graph_builder.py          # Graph visualization builder
├── main.py                   # Application entry point
├── docker-compose.yml        # Docker services configuration
├── Dockerfile                # Application Docker image
└── pyproject.toml           # Project dependencies
```

## Setup

### 1. Start Qdrant Vector Database

```bash
docker compose up -d
```

### 2. Install uv Package Manager

**Linux/Mac:**
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

More info: https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

### 3. Install Dependencies

```bash
uv sync
```

### 4. Configure Environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Proxy configuration
# PROXY_URL=http://your-proxy:port

# Optional: Qdrant configuration
# QDRANT_URL=http://localhost:6333

# Optional: Processing configuration
# CHUNK_SIZE=800
# CHUNK_OVERLAP=150
# MAX_WORKERS=4
```

### 5. Run the Application

```bash
python3 main.py
```

The Gradio interface will be available at `http://localhost:7860`

## Docker Deployment

To run the entire stack with Docker:

```bash
docker compose up -d
```

This will start:
- Qdrant vector database (ports 6333, 6334)
- Gradio application (port 7860)

## Features

- **Document Processing**: Upload and process PDF and Markdown medical documents
- **Entity Extraction**: Automatically extract medical entities (diseases, drugs, symptoms, etc.)
- **Relation Extraction**: Identify relationships between medical entities
- **Vector Search**: Semantic search using Qdrant vector database
- **Knowledge Graph**: Interactive visualization of extracted entities and relations
- **Q&A System**: Answer questions based on processed documents
- **Session Management**: Organize documents into isolated sessions