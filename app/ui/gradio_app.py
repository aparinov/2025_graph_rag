# -*- coding: utf-8 -*-
"""Gradio UI application."""

import os
import re
from typing import List, Dict, Any
from datetime import datetime

import gradio as gr

from app.dependencies import get_qdrant_manager, get_document_service, get_qa_service
from app.config import COLLECTION_NAME_PATTERN


def get_collections() -> List[Dict[str, Any]]:
    """Get all available collections (both legacy sessions and new collections)."""
    qdrant = get_qdrant_manager()
    return qdrant.get_collections()


def get_collection_choices() -> List[str]:
    """Get collection names for dropdown choices."""
    collections = get_collections()
    choices = []
    for coll in collections:
        name = coll["name"]
        if coll["type"] == "legacy":
            name += " *"  # Mark legacy sessions
        choices.append(name)
    return choices


def strip_legacy_marker(collection_name: str) -> str:
    """Remove legacy marker (*) from collection name."""
    return collection_name.rstrip(" *")


def get_documents_for_collections(collection_names: List[str]) -> List[List[Any]]:
    """Get documents for selected collections."""
    if not collection_names:
        return []

    qdrant = get_qdrant_manager()
    all_docs = []

    for coll_name in collection_names:
        clean_name = strip_legacy_marker(coll_name)
        docs = qdrant.get_documents(collection_name=clean_name)

        for doc in docs:
            # Format created_at to be more readable
            created = doc.get("created_at", "")
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    created = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    created = created[:16] if len(created) > 16 else created

            status = doc.get("status", "unknown")
            # Add emoji for status
            status_display = {
                "queued": "🕒 Queued",
                "processing": "⏳ Processing",
                "completed": "✅ Completed",
                "error": "❌ Error",
                "unknown": "❓ Unknown",
            }.get(status, status)

            # Add error message to status if present
            if status == "error" and doc.get("error_message"):
                status_display += (
                    f" ({doc['error_message'][:50]}...)"
                    if len(doc.get("error_message", "")) > 50
                    else f" ({doc.get('error_message', '')})"
                )

            all_docs.append(
                [
                    coll_name,  # Collection name
                    doc["name"],  # Document name
                    status_display,
                    created,
                    doc.get("chunks", 0),
                    doc.get("entities", 0),
                ]
            )

    return all_docs


def validate_collection_name(name: str) -> bool:
    """Validate collection name format."""
    return bool(re.match(COLLECTION_NAME_PATTERN, name))


def create_collection_ui(collection_name: str):
    """Create a new collection."""
    name = collection_name.strip().lower()

    if not name:
        return (
            "❌ Укажите название коллекции.",
            gr.Dropdown(choices=get_collection_choices()),
        )

    if not validate_collection_name(name):
        return (
            "❌ Название должно содержать только латинские буквы, цифры и подчеркивания (3-50 символов).",
            gr.Dropdown(choices=get_collection_choices()),
        )

    # Check if collection already exists
    existing = get_collections()
    if any(c["name"] == name for c in existing):
        return (
            f"❌ Коллекция '{name}' уже существует.",
            gr.Dropdown(choices=get_collection_choices()),
        )

    # Create the collection
    qdrant = get_qdrant_manager()
    try:
        qdrant.create_collection(name)
        return (
            f"✅ Коллекция '{name}' создана.",
            gr.Dropdown(choices=get_collection_choices(), value=name),
        )
    except Exception as e:
        return (
            f"❌ Ошибка создания коллекции: {e}",
            gr.Dropdown(choices=get_collection_choices()),
        )


def on_collection_change(selected_collections: List[str]):
    """Handle collection selection change."""
    if not selected_collections:
        return [], "⚠️ Выберите хотя бы одну коллекцию."

    clean_names = [strip_legacy_marker(c) for c in selected_collections]
    status = f"Выбрано коллекций: {len(clean_names)} ({', '.join(clean_names)})"
    return get_documents_for_collections(selected_collections), status


def refresh_documents(selected_collections: List[str]):
    """Refresh document list."""
    if not selected_collections:
        return [], "⚠️ Выберите хотя бы одну коллекцию."

    clean_names = [strip_legacy_marker(c) for c in selected_collections]
    status = f"Выбрано коллекций: {len(clean_names)}"
    return get_documents_for_collections(selected_collections), status


def upload_files_ui(files, upload_collection: str):
    """Upload and process PDF or Markdown documents."""
    qdrant = get_qdrant_manager()
    doc_service = get_document_service()

    if not upload_collection:
        return "❌ Выберите коллекцию для загрузки.", []

    if not files:
        return "❌ Файлы не выбраны.", []

    clean_collection = strip_legacy_marker(upload_collection)

    file_list = files if isinstance(files, list) else [files]
    file_paths: List[str] = []
    for file_obj in file_list:
        file_path = getattr(file_obj, "name", None) or str(file_obj)
        file_paths.append(file_path)

        # Set initial status as queued for each document
        document_name = os.path.splitext(os.path.basename(file_path))[0]

        # Determine if this is a legacy collection
        collection_type = qdrant.get_collection_type(clean_collection)
        if collection_type == "legacy":
            qdrant.set_document_status(
                document_name, "queued", session_name=clean_collection
            )
        else:
            qdrant.set_document_status(
                document_name, "queued", collection_name=clean_collection
            )

    # Enqueue upload
    if collection_type == "legacy":
        job_id = doc_service.enqueue_upload(file_paths, session_name=clean_collection)
    else:
        job_id = doc_service.enqueue_upload(file_paths, collection_name=clean_collection)

    return (
        f"🕒 Добавлено в очередь: {len(file_paths)} файл(ов). "
        f"ID задания: {job_id}. Обработка продолжается в фоне. Используйте кнопку 'Обновить' для проверки статуса.",
        get_documents_for_collections([upload_collection]),
    )


def delete_collection_ui(collection_name: str, confirmed: bool):
    """Delete a collection and all its data."""
    qdrant = get_qdrant_manager()

    if not collection_name:
        return (
            "❌ Выберите коллекцию для удаления.",
            gr.Dropdown(choices=get_collection_choices()),
            [],
            False,
        )

    if not confirmed:
        return (
            "⚠️ Подтвердите удаление коллекции.",
            gr.Dropdown(choices=get_collection_choices()),
            [],
            False,
        )

    clean_name = strip_legacy_marker(collection_name)
    collection_type = qdrant.get_collection_type(clean_name)

    try:
        if collection_type == "legacy":
            # Legacy sessions can be deleted
            qdrant.delete_session(clean_name)
        else:
            # New collections can be deleted
            qdrant.delete_collection(clean_name)

        remaining = get_collection_choices()

        return (
            f"✅ Коллекция '{clean_name}' удалена.",
            gr.Dropdown(choices=remaining),
            [],
            False,
        )
    except Exception as e:
        return (
            f"❌ Ошибка удаления: {e}",
            gr.Dropdown(choices=get_collection_choices()),
            [],
            False,
        )


def chat(
    message: str,
    history: List[Dict[str, str]],
    selected_collections: List[str],
):
    """Handle chat messages."""
    qa_service = get_qa_service()

    if not selected_collections:
        return history, ""

    if not message.strip():
        return history, ""

    clean_collections = [strip_legacy_marker(c) for c in selected_collections]

    history = history or []
    history.append({"role": "user", "content": message})

    # Get answer
    answer = qa_service.answer_question(message, clean_collections)
    history.append({"role": "assistant", "content": answer})

    return history, ""


def create_app() -> gr.Blocks:
    """Create and configure Gradio application."""
    # Initialize with existing collections
    initial_collections = get_collection_choices()
    initial_selection = initial_collections[:1] if initial_collections else []
    initial_docs = get_documents_for_collections(initial_selection) if initial_selection else []
    initial_status = (
        f"Выбрано коллекций: {len(initial_selection)}"
        if initial_selection
        else "Создайте коллекцию."
    )

    with gr.Blocks() as demo:
        with gr.Sidebar():
            gr.Markdown("### 📚 Коллекции")

            # Multi-select for querying
            collection_selector = gr.Dropdown(
                choices=initial_collections,
                value=initial_selection,
                label="Выбрать коллекции для поиска",
                multiselect=True,
            )

            collection_status = gr.Textbox(
                label="Статус",
                value=initial_status,
                interactive=False,
            )

            gr.Markdown("### ➕ Создать коллекцию")
            new_collection_name = gr.Textbox(
                label="Название",
                placeholder="например: clinical_en",
            )
            create_collection_btn = gr.Button("Создать")
            create_status = gr.Textbox(label="Статус создания", interactive=False)

            gr.Markdown("### 📤 Загрузка документов")
            upload_collection = gr.Dropdown(
                choices=initial_collections,
                label="Коллекция для загрузки (одна)",
                value=initial_selection[0] if initial_selection else None,
            )
            upload_files = gr.File(
                file_count="multiple",
                file_types=[".pdf", ".md", ".markdown"],
                label="PDF и Markdown файлы",
            )
            upload_btn = gr.Button("Загрузить и обработать")
            upload_status = gr.Textbox(label="Статус загрузки", interactive=False)

            gr.Markdown("### 🗑️ Удалить коллекцию")
            delete_confirm = gr.Checkbox(label="Подтвердить удаление", value=False)
            delete_collection_btn = gr.Button("Удалить коллекцию", variant="stop")
            delete_status = gr.Textbox(label="Статус удаления", interactive=False)

        with gr.Column():
            gr.Markdown("### Вопросы по документам")
            chatbot = gr.Chatbot(type="messages", height=500)
            user_input = gr.Textbox(
                placeholder="Спросите о документах...", label="Ваш вопрос"
            )
            ask_btn = gr.Button("Спросить")

            gr.Markdown("### Документы в коллекциях")
            refresh_docs_btn = gr.Button("Обновить")
            document_table = gr.Dataframe(
                headers=["Коллекция", "Документ", "Статус", "Создан", "Чанков", "Сущностей"],
                datatype=["str", "str", "str", "str", "number", "number"],
                interactive=False,
                value=initial_docs,
            )

        # Event handlers
        create_collection_btn.click(
            fn=create_collection_ui,
            inputs=[new_collection_name],
            outputs=[create_status, collection_selector],
        ).then(
            fn=lambda: gr.Dropdown(choices=get_collection_choices()),
            inputs=None,
            outputs=[upload_collection],
        )

        collection_selector.change(
            fn=on_collection_change,
            inputs=[collection_selector],
            outputs=[document_table, collection_status],
        )

        upload_btn.click(
            fn=upload_files_ui,
            inputs=[upload_files, upload_collection],
            outputs=[upload_status, document_table],
        )

        refresh_docs_btn.click(
            fn=refresh_documents,
            inputs=[collection_selector],
            outputs=[document_table, collection_status],
        )

        delete_collection_btn.click(
            fn=delete_collection_ui,
            inputs=[upload_collection, delete_confirm],
            outputs=[delete_status, upload_collection, document_table, delete_confirm],
        ).then(
            fn=lambda: gr.Dropdown(choices=get_collection_choices()),
            inputs=None,
            outputs=[collection_selector],
        )

        ask_btn.click(
            fn=chat,
            inputs=[user_input, chatbot, collection_selector],
            outputs=[chatbot, user_input],
        )

        user_input.submit(
            fn=chat,
            inputs=[user_input, chatbot, collection_selector],
            outputs=[chatbot, user_input],
        )

        def init_ui():
            collections = get_collection_choices()
            default_selection = collections[:1] if collections else []
            status = (
                f"Выбрано коллекций: {len(default_selection)}"
                if default_selection
                else "Создайте коллекцию."
            )
            docs = get_documents_for_collections(default_selection) if default_selection else []
            return (
                gr.update(choices=collections, value=default_selection),
                gr.update(choices=collections, value=default_selection[0] if default_selection else None),
                docs,
                status,
            )

        demo.load(
            fn=init_ui,
            inputs=None,
            outputs=[collection_selector, upload_collection, document_table, collection_status],
        )

    return demo
