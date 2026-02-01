# -*- coding: utf-8 -*-
"""Main entry point for the Graph RAG application."""

from app.ui.gradio_app import create_app

if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=True)
