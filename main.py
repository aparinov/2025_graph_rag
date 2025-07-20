import os
import asyncio

import gradio as gr
from dotenv import load_dotenv
from langchain_community.llms import GigaChat
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_text_splitters import CharacterTextSplitter

load_dotenv(override=True)

HOST = os.getenv("NEO4J_HOST", "localhost")
PORT = os.getenv("NEO4J_PORT", "7687")
print(f"neo4j://{HOST}:{PORT}")
# === Neo4j Setup ===
graph = Neo4jGraph(
    url=f"neo4j://{HOST}:{PORT}",
    username="neo4j",
    password=os.getenv("NEO4J_PASSWORD"),
    refresh_schema=True,
    enhanced_schema=True,
)

# === Global document name storage ===
current_document_name = None


def upload_pdfs(files, llm_provider):
    global current_document_name
    if not files:
        return "❌ No file uploaded."

    pdf_path = files
    current_document_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Load & chunk text
    loader = UnstructuredPDFLoader(pdf_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    chunks = splitter.split_documents(docs)

    # 🧠 Use LLMGraphTransformer to extract entities & relations
    llm = create_llm_client(llm_provider)
    print(llm)
    transformer = LLMGraphTransformer(
        llm=llm,
        strict_mode=False,  # allow any entities/relations
        node_properties=True,
        relationship_properties=True,
    )
    graph_docs = transformer.convert_to_graph_documents(chunks)

    # Store to Neo4j
    graph.add_graph_documents(graph_docs, include_source=True)

    graph.refresh_schema()
    return f"✅ Ingested & extracted graph from '{current_document_name}' ({len(chunks)} chunks)"


def create_llm_client(provider: str):
    """
    Placeholder for LLM client creation.
    """
    if provider == "OpenAI":
        # return OpenAI client
        return ChatOpenAI(
            model_name="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider == "GigaChat":
        # return GigaChat client (placeholder)
        return GigaChat(
            credentials=os.getenv("GIGACHAT_API_KEY"),
            model="GigaChat-2",
            temperature=0,
            verify_ssl_certs=False,
        )
    elif provider == "Yandex":
        # return Yandex client (placeholder)
        return "YandexClient()"
    else:
        raise ValueError(f"Unknown provider: {provider}")


def predict_with_rag(message, history, provider):
    history = history or []
    history.append({"role": "user", "content": message})

    llm = create_llm_client(provider)
    # If using a placeholder string, handle accordingly
    if isinstance(llm, str):
        assistant_reply = f"[{provider}]: This is a placeholder response."
    else:
        qa_chain = GraphCypherQAChain.from_llm(
            cypher_llm=llm,
            qa_llm=llm,
            graph=graph,
            top_k=20,
            verbose=True,
            return_direct=False,
            allow_dangerous_requests=True,
            validate_cypher=True,
        )
        result = qa_chain.invoke(message)
        if isinstance(result, dict):
            if "result" in result:
                assistant_reply = str(result["result"])
            else:
                assistant_reply = str(result)
        else:
            assistant_reply = str(result)

    history.append({"role": "assistant", "content": assistant_reply})
    return history, history


def clear_neo4j():
    graph.query("MATCH (n) DETACH DELETE n")
    graph.query("MATCH (n) SET n = {}")
    return "✅ All records cleared from Neo4j."


# === Gradio UI ===
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
