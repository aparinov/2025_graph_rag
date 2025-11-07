#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import our improved pipeline
from pipelines.improved_pipeline import ImprovedPipeline

load_dotenv(override=True)

# ----------------------------
# Configuration
# ----------------------------
NEO4J_HOST = os.getenv("NEO4J_HOST", "localhost")
NEO4J_PORT = os.getenv("NEO4J_PORT", "7687")
NEO4J_URL = f"neo4j://{NEO4J_HOST}:{NEO4J_PORT}"
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ----------------------------
# Multi-Document Graph Manager
# ----------------------------
class MultiDocumentGraphManager:
    """
    Handles multiple documents in Neo4j by:
    1. Each document gets a unique Document node
    2. Entities are shared across documents if they have the same name+type
    3. Context-aware querying allows filtering by specific documents
    """
    
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_URL,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            refresh_schema=False
        )
        self.pipeline = ImprovedPipeline()
        self._setup_constraints()
    
    def _setup_constraints(self):
        """Setup database constraints and indexes for better performance"""
        try:
            # Create unique constraints
            self.graph.query("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            self.graph.query("CREATE CONSTRAINT document_name IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE")
            
            # Create indexes for faster lookups - fix syntax
            self.graph.query("CREATE INDEX entity_name_type IF NOT EXISTS FOR (e:Entity) ON (e.name, e.type)")
            self.graph.query("CREATE INDEX document_created IF NOT EXISTS FOR (d:Document) ON (d.created_at)")
            
        except Exception as e:
            print(f"⚠️ Constraint/index setup: {e}")
    
    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """Process a new document and integrate it into the multi-document graph"""
        document_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Check if document already exists
        existing = self.graph.query(
            "MATCH (d:Document {name: $name}) RETURN d.created_at as created",
            params={"name": document_name}
        )
        
        if existing:
            return {
                "status": "exists",
                "message": f"Document '{document_name}' already processed at {existing[0]['created']}",
                "document": document_name
            }
        
        # Process with improved pipeline
        result = self.pipeline.process_pdf(pdf_path)
        
        # Add metadata to document node
        self.graph.query(
            """
            MATCH (d:Document {name: $name})
            SET d.created_at = $timestamp,
                d.file_path = $path,
                d.processed_version = $version
            """,
            params={
                "name": document_name,
                "timestamp": datetime.now().isoformat(),
                "path": pdf_path,
                "version": "improved_v1"
            }
        )
        
        # Merge entities across documents (entities with same name+type are linked)
        self._merge_cross_document_entities()
        
        result["status"] = "success"
        result["message"] = f"Successfully processed '{document_name}'"
        return result
    
    def _merge_cross_document_entities(self):
        """Merge entities that appear across multiple documents"""
        # Find potential duplicates across documents
        merge_query = """
        MATCH (e1), (e2)
        WHERE e1.name = e2.name 
        AND e1.type = e2.type 
        AND id(e1) < id(e2)
        AND NOT (e1)-[:MERGED_WITH]-(e2)
        WITH e1, e2, 
             COUNT {(e1)-[:APPEARS_IN]->()} as e1_docs,
             COUNT {(e2)-[:APPEARS_IN]->()} as e2_docs
        WHERE e1_docs > 0 AND e2_docs > 0
        MERGE (e1)-[:MERGED_WITH]-(e2)
        SET e1.cross_document = true,
            e2.cross_document = true,
            e1.total_appearances = e1_docs + e2_docs
        RETURN count(*) as merged_pairs
        """
        
        result = self.graph.query(merge_query)
        if result and result[0]["merged_pairs"] > 0:
            print(f"🔗 Merged {result[0]['merged_pairs']} cross-document entity pairs")
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """Get list of all processed documents"""
        query = """
        MATCH (d:Document)
        OPTIONAL MATCH (d)<-[:APPEARS_IN]-(e)
        RETURN d.name as name, 
               d.created_at as created,
               d.chunks as chunks,
               count(DISTINCT e) as entities
        ORDER BY d.created_at DESC
        """
        
        results = self.graph.query(query)
        return [dict(result) for result in results]
    
    def delete_document(self, document_name: str) -> Dict[str, str]:
        """Delete a document and its orphaned entities"""
        # First, delete entities that only appear in this document
        orphan_query = """
        MATCH (e)-[:APPEARS_IN]->(d:Document {name: $name})
        WHERE COUNT {(e)-[:APPEARS_IN]->()} = 1
        DETACH DELETE e
        """
        
        # Then delete the document itself
        doc_query = """
        MATCH (d:Document {name: $name})
        DETACH DELETE d
        """
        
        self.graph.query(orphan_query, params={"name": document_name})
        result = self.graph.query(doc_query, params={"name": document_name})
        
        return {
            "status": "success",
            "message": f"Deleted document '{document_name}' and orphaned entities"
        }
    
    def clear_all(self) -> Dict[str, str]:
        """Clear entire database"""
        self.graph.query("MATCH (n) DETACH DELETE n")
        return {"status": "success", "message": "All data cleared"}

# ----------------------------
# Document-Aware QA System
# ----------------------------
class DocumentAwareQA:
    """
    QA system that can answer questions about specific documents or across all documents
    """
    
    def __init__(self, graph_manager: MultiDocumentGraphManager):
        self.graph_manager = graph_manager
        self.graph = graph_manager.graph
        
        # Create different LLMs for different tasks
        self.cypher_llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.qa_llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Custom prompt for multi-document awareness
        self.qa_chain = None
        self._setup_qa_chain()
    
    def _setup_qa_chain(self):
        """Setup QA chain with custom prompts for multi-document context"""
        
        # Use default prompts - the custom prompts were causing issues
        self.qa_chain = GraphCypherQAChain.from_llm(
            cypher_llm=self.cypher_llm,
            qa_llm=self.qa_llm,
            graph=self.graph,
            top_k=20,
            verbose=True,
            return_direct=False,
            allow_dangerous_requests=True
        )
    
    def ask_question(self, question: str, document_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Ask a question, optionally filtered to a specific document
        """
        start_time = time.time()
        
        # Modify question if document filter is specified
        if document_filter and document_filter != "All Documents":
            enhanced_question = f"In the document '{document_filter}': {question}"
        else:
            enhanced_question = question
        
        try:
            # Refresh schema before querying
            self.graph.refresh_schema()
            
            result = self.qa_chain.invoke({"query": enhanced_question})
            
            answer = result.get("result", "I couldn't find relevant information to answer your question.")
            cypher_query = result.get("cypher_query", "")
            
            # Get document context
            doc_context = self._get_answer_context(cypher_query, document_filter)
            
            response_time = time.time() - start_time
            
            return {
                "answer": answer,
                "cypher_query": cypher_query,
                "document_context": doc_context,
                "response_time": round(response_time, 2),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "cypher_query": "",
                "document_context": [],
                "response_time": 0,
                "status": "error"
            }
    
    def _get_answer_context(self, cypher_query: str, document_filter: Optional[str]) -> List[str]:
        """Get information about which documents contributed to the answer"""
        try:
            if document_filter and document_filter != "All Documents":
                context_query = """
                MATCH (d:Document {name: $doc_name})
                RETURN d.name as document, d.created_at as created
                """
                result = self.graph.query(context_query, params={"doc_name": document_filter})
            else:
                context_query = """
                MATCH (d:Document)
                RETURN d.name as document, d.created_at as created
                ORDER BY d.created_at DESC
                LIMIT 5
                """
                result = self.graph.query(context_query)
            
            return [f"{r['document']} (processed: {r['created'][:10]})" for r in result]
            
        except Exception:
            return ["Context information unavailable"]

# ----------------------------
# Gradio Interface
# ----------------------------
class GraphRAGApp:
    def __init__(self):
        self.graph_manager = MultiDocumentGraphManager()
        self.qa_system = DocumentAwareQA(self.graph_manager)
        
    def upload_document(self, file, progress=gr.Progress()):
        """Handle document upload and processing"""
        if not file:
            return "❌ No file uploaded", self._get_document_list(), "All Documents"
        
        progress(0.1, desc="Starting document processing...")
        
        try:
            # Process the document
            progress(0.3, desc="Extracting entities and relations...")
            result = self.graph_manager.process_document(file.name)
            
            progress(0.8, desc="Finalizing...")
            
            if result["status"] == "exists":
                message = f"⚠️ {result['message']}"
            else:
                message = f"✅ {result['message']}\n"
                message += f"📊 Processed {result['chunks']} chunks, "
                message += f"extracted {result['entities']} entities, "
                message += f"{result['relations']} relations in {result['processing_time']}s"
            
            progress(1.0, desc="Complete!")
            
            # Update document list
            doc_list = self._get_document_list()
            doc_choices = ["All Documents"] + [doc["name"] for doc in doc_list]
            
            return message, doc_list, gr.Dropdown(choices=doc_choices, value="All Documents")
            
        except Exception as e:
            return f"❌ Error processing document: {str(e)}", self._get_document_list(), "All Documents"
    
    def ask_question(self, question: str, document_filter: str, history):
        """Handle question answering"""
        if not question.strip():
            return history, ""
        
        # Add user message to history
        history = history or []
        history.append({"role": "user", "content": question})
        
        # Get answer
        result = self.qa_system.ask_question(question, document_filter)
        
        # Format answer with context
        answer = result["answer"]
        if result["document_context"]:
            answer += f"\n\n📚 **Sources:** {', '.join(result['document_context'])}"
        
        if result["response_time"]:
            answer += f"\n⏱️ *Response time: {result['response_time']}s*"
        
        # Add assistant response to history
        history.append({"role": "assistant", "content": answer})
        
        return history, ""
    
    def delete_document(self, doc_name: str):
        """Delete a document"""
        if not doc_name or doc_name == "All Documents":
            return "❌ Please select a valid document to delete", self._get_document_list()
        
        result = self.graph_manager.delete_document(doc_name)
        doc_list = self._get_document_list()
        
        return f"✅ {result['message']}", doc_list
    
    def clear_database(self):
        """Clear entire database"""
        result = self.graph_manager.clear_all()
        return f"✅ {result['message']}", [], "All Documents"
    
    def _get_document_list(self):
        """Get formatted document list for display"""
        docs = self.graph_manager.get_documents()
        return docs

# ----------------------------
# Gradio UI
# ----------------------------
def create_app():
    app = GraphRAGApp()
    
    with gr.Blocks(title="Graph RAG - Multi-Document Knowledge Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 📚 Graph RAG - Multi-Document Knowledge Assistant
        
        Upload PDF documents to build a knowledge graph and ask questions across all your documents!
        """)
        
        with gr.Row():
            # Left column - Document management
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Document Management")
                
                file_upload = gr.File(
                    file_types=[".pdf"], 
                    label="Upload PDF Document"
                )
                
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                    value="Ready to upload documents..."
                )
                
                gr.Markdown("### 📋 Document Library")
                document_list = gr.Dataframe(
                    headers=["Document", "Created", "Chunks", "Entities"],
                    datatype=["str", "str", "number", "number"],
                    label="Processed Documents",
                    interactive=False,
                    value=app._get_document_list()
                )
                
                with gr.Row():
                    delete_btn = gr.Button("🗑️ Delete Selected", variant="stop", size="sm")
                    clear_btn = gr.Button("💥 Clear All", variant="stop", size="sm")
                
                delete_doc_name = gr.Textbox(
                    label="Document name to delete",
                    placeholder="Enter exact document name...",
                    visible=True
                )
            
            # Right column - Chat interface
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Ask Questions")
                
                document_filter = gr.Dropdown(
                    choices=["All Documents"],
                    value="All Documents",
                    label="Search in Document",
                    info="Select a specific document or search across all"
                )
                
                chatbot = gr.Chatbot(
                    type="messages",
                    height=400,
                    label="Knowledge Assistant"
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        placeholder="Ask about your documents...",
                        label="Your Question",
                        scale=4
                    )
                    ask_btn = gr.Button("Ask", variant="primary", scale=1)
        
        # Event handlers
        file_upload.upload(
            fn=app.upload_document,
            inputs=[file_upload],
            outputs=[upload_status, document_list, document_filter]
        )
        
        ask_btn.click(
            fn=app.ask_question,
            inputs=[question_input, document_filter, chatbot],
            outputs=[chatbot, question_input]
        )
        
        question_input.submit(
            fn=app.ask_question,
            inputs=[question_input, document_filter, chatbot],
            outputs=[chatbot, question_input]
        )
        
        delete_btn.click(
            fn=app.delete_document,
            inputs=[delete_doc_name],
            outputs=[upload_status, document_list]
        )
        
        clear_btn.click(
            fn=app.clear_database,
            inputs=[],
            outputs=[upload_status, document_list, document_filter]
        )
        
        # Initialize
        demo.load(
            fn=lambda: ("🚀 Ready to process documents!", app._get_document_list()),
            outputs=[upload_status, document_list]
        )
    
    return demo

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
