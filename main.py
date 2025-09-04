import os
import asyncio
import json

import gradio as gr
from dotenv import load_dotenv
from langchain_community.llms.gigachat import GigaChat
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# It's a good practice to handle potential import errors for optional dependencies
try:
    import json_repair
except ImportError:
    raise ImportError(
        "The 'json_repair' library is required for GigaChat compatibility. "
        "Please install it with 'pip install json-repair'"
    )


load_dotenv(override=True)

HOST = os.getenv("NEO4J_HOST", "localhost")
PORT = os.getenv("NEO4J_PORT", "7687")
print(f"neo4j://{HOST}:{PORT}")

# === Neo4j Setup ===
graph = Neo4jGraph(
    url=f"neo4j://{HOST}:{PORT}",
    username="neo4j",
    password=os.getenv("NEO4J_PASSWORD"),
    # Let's disable schema refresh on init and handle it manually
    refresh_schema=False,
    enhanced_schema=False,  # We will generate our own schema info for the prompt
)

# === Global document name storage ===
current_document_name = None

# ... (all your other imports remain the same)


def ingest_with_iterative_extraction(
    chunks: list[Document], llm, source_document_name: str
):
    """
    Processes documents by first extracting all entities, then iterating through
    pairs of entities to find relationships. This is more stable for LLMs that
    struggle with complex, single-pass JSON generation.
    """

    # 1. First Pass: Extract all entities from each chunk
    # CORRECTED: Escaped curly braces in the example JSON to prevent templating errors.
    entity_extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert in extracting named entities from text. "
                "Extract all entities, their types, and any properties you can find. "
                "Respond with a JSON list of objects, where each object has 'id', 'type', and 'properties' keys. "
                "Example: [{{'id': 'Elon Musk', 'type': 'Person', 'properties': {{'title': 'CEO'}}}}, {{'id': 'SpaceX', 'type': 'Company', 'properties': {{}} }}]",
            ),
            ("human", "Extract entities from the following text: \n\n{text}"),
        ]
    )
    entity_chain = entity_extraction_prompt | llm | StrOutputParser()

    all_chunk_entities = []
    print("--- Starting Entity Extraction Pass ---")
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} for entities...")
        try:
            response = entity_chain.invoke({"text": chunk.page_content})
            entities = json_repair.loads(response)
            all_chunk_entities.append(entities)
            # Natively insert nodes as we find them
            for entity in entities:
                # Ensure 'id' exists before proceeding
                if "id" in entity and entity["id"]:
                    graph.query(
                        "MERGE (n:`"
                        + entity.get("type", "Unknown")
                        + "` {id: $id}) SET n += $properties",
                        params={
                            "id": entity["id"],
                            "properties": entity.get("properties", {}),
                        },
                    )
        except Exception as e:
            print(f"  Error processing chunk {i+1} for entities: {e}")
            all_chunk_entities.append([])  # Add empty list on failure
    print("--- Entity Extraction Complete ---")

    # 2. Second Pass: Extract relationships between entities found in each chunk
    # CORRECTED: Escaped curly braces in the example JSON to prevent templating errors.
    relationship_extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert in finding relationships between entities in text. "
                "Given a text and two entities, determine if a relationship exists between them. "
                "If it does, describe the relationship type and any properties. "
                "Respond with a single JSON object with 'type' and 'properties' keys, or an empty JSON object {{}} if no relationship exists. "
                "Example: {{'type': 'CEO_OF', 'properties': {{'start_date': '2002'}}}}",
            ),
            (
                "human",
                "Text: {text}\n\nEntity 1: {entity1}\nEntity 2: {entity2}\n\nWhat is the relationship between Entity 1 and Entity 2?",
            ),
        ]
    )
    relationship_chain = relationship_extraction_prompt | llm | StrOutputParser()

    print("--- Starting Relationship Extraction Pass ---")
    for i, chunk_entities in enumerate(all_chunk_entities):
        if len(chunk_entities) < 2:
            continue

        print(f"Processing chunk {i+1}/{len(chunks)} for relationships...")
        # Iterate through all unique pairs of entities in the chunk
        for j in range(len(chunk_entities)):
            for k in range(j + 1, len(chunk_entities)):
                entity1 = chunk_entities[j]
                entity2 = chunk_entities[k]

                # Basic check to ensure entities are valid dictionaries with 'id'
                if not all(
                    isinstance(e, dict) and "id" in e for e in [entity1, entity2]
                ):
                    continue

                try:
                    response = relationship_chain.invoke(
                        {
                            "text": chunks[i].page_content,
                            "entity1": entity1["id"],
                            "entity2": entity2["id"],
                        }
                    )
                    relationship = json_repair.loads(response)

                    if relationship and relationship.get("type"):
                        # Natively insert the relationship
                        graph.query(
                            """
                            MATCH (a {id: $source_id}), (b {id: $target_id})
                            MERGE (a)-[r:`"""
                            + relationship["type"]
                            + """`]->(b)
                            SET r += $properties
                            """,
                            params={
                                "source_id": entity1["id"],
                                "target_id": entity2["id"],
                                "properties": relationship.get("properties", {}),
                            },
                        )
                except Exception as e:
                    print(
                        f"  Error processing relationship between '{entity1.get('id', 'N/A')}' and '{entity2.get('id', 'N/A')}': {e}"
                    )

    # Finally, link all nodes to the source document
    graph.query(
        """
        MATCH (n) WHERE NOT n:Document
        MERGE (s:Document {id: $doc_name})
        MERGE (n)-[:APPEARS_IN]->(s)
    """,
        params={"doc_name": source_document_name},
    )

    print("--- Relationship Extraction Complete ---")


def upload_pdfs(files, llm_provider):
    global current_document_name
    if not files:
        return "❌ No file uploaded."

    pdf_path = files
    current_document_name = os.path.splitext(os.path.basename(pdf_path))[0]

    loader = UnstructuredPDFLoader(pdf_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    chunks = splitter.split_documents(docs)

    # 🧠 Select the LLM
    llm = create_llm_client(llm_provider)
    print(llm)

    # ### MODIFIED ### - Use the new iterative ingestion function
    ingest_with_iterative_extraction(chunks, llm, current_document_name)

    # Manually refresh schema after ingestion
    graph.refresh_schema()

    return f"✅ Ingested & extracted graph from '{current_document_name}' ({len(chunks)} chunks)"


def create_llm_client(provider: str, for_cypher: bool = False):
    """
    Creates an LLM client. If for_cypher is True, it will prioritize
    the most powerful model available (OpenAI).
    """
    if for_cypher and os.getenv("OPENAI_API_KEY"):
        # Always use the best model for Cypher generation if available
        return ChatOpenAI(
            model_name="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
        )

    if provider == "OpenAI":
        return ChatOpenAI(
            model_name="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider == "GigaChat":
        return GigaChat(
            credentials=os.getenv("GIGACHAT_API_KEY"),
            model="GigaChat-Pro",
            temperature=0,
            verify_ssl_certs=False,
        )
    elif provider == "Yandex":
        raise NotImplementedError("Yandex client not implemented.")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def predict_with_rag(message, history, provider):
    history = history or []
    history.append({"role": "user", "content": message})

    # ### MODIFIED ### - Implement the Hybrid LLM approach
    # Use the best available model for Cypher generation
    cypher_llm = create_llm_client(provider, for_cypher=True)
    # Use the user-selected model for the final answer
    qa_llm = create_llm_client(provider)

    # print(f"Using {cypher_llm.model_name} for Cypher generation.")
    # print(f"Using {qa_llm.model_name or provider} for QA.")

    qa_chain = GraphCypherQAChain.from_llm(
        cypher_llm=cypher_llm,
        qa_llm=qa_llm,
        graph=graph,
        top_k=20,
        verbose=True,
        return_direct=False,
        allow_dangerous_requests=True,
    )

    result = qa_chain.invoke({"query": message})
    assistant_reply = result.get("result", "Sorry, I couldn't find an answer.")

    history.append({"role": "assistant", "content": assistant_reply})
    return history, history


def clear_neo4j():
    graph.query("MATCH (n) DETACH DELETE n")
    # No need for the second query, DETACH DELETE handles it.
    return "✅ All records cleared from Neo4j."


with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.Markdown("### Upload a PDF to begin")
        llm_provider = gr.Dropdown(
            choices=["OpenAI", "GigaChat", "Yandex"],
            value="OpenAI",
            label="LLM Provider",
        )
        upload_btn = gr.UploadButton("Upload PDF", file_types=[".pdf"])
        upload_status = gr.Textbox(label="Status", interactive=False)
        upload_btn.upload(upload_pdfs, [upload_btn, llm_provider], upload_status)

        clear_btn = gr.Button("Clear Neo4j DB", variant="stop")
        clear_status = gr.Textbox(label="DB Status", interactive=False)
        clear_btn.click(fn=clear_neo4j, inputs=None, outputs=clear_status)

    chatbot = gr.Chatbot(type="messages")
    user_input = gr.Textbox(
        placeholder="Ask about the document…", label="Your Question"
    )
    ask_btn = gr.Button("Ask")
    ask_btn.click(
        predict_with_rag,
        inputs=[user_input, chatbot, llm_provider],
        outputs=[chatbot, chatbot],
    )

    demo.load(lambda: "✅ Ready to ingest and query!", None, upload_status)

demo.launch(share=True)
