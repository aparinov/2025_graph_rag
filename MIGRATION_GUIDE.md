# Migration Guide: Session-Based to Collection-Based Architecture

## Overview

This guide explains the migration from a session-based paradigm to a collection-based architecture, completed on 2026-02-01.

## What Changed?

### Before (Session-Based)
- Single Qdrant collection: `medical_documents`
- Data isolated by `session_name` field
- One session at a time for querying
- UI: Session selector (single select)

### After (Collection-Based)
- Multiple Qdrant collections (one per topic/project)
- Legacy sessions preserved as "virtual collections" in `medical_documents`
- Multi-collection querying
- UI: Collection multi-selector with legacy marker `*`

## Key Features

1. **Multi-Collection Support**: Create and manage multiple collections
2. **Multi-Collection Querying**: Search across multiple collections simultaneously
3. **Enhanced Citations**: Answers include collection name and document name
4. **Backward Compatibility**: All existing sessions work as legacy collections
5. **Zero Migration**: No data migration required, existing data works immediately

## Verification

Run the verification script to ensure everything is set up correctly:

```bash
python3 verify_migration.py
```

Expected output:
```
============================================================
Collection-Based Migration Verification
============================================================
Testing imports...
  ✓ QdrantManager imported
  ✓ DocumentService imported
  ✓ QAService imported
  ✓ Gradio UI imported

Testing method signatures...
  ✓ create_collection has correct signature
  ✓ get_collections has correct signature
  ✓ get_collection_type has correct signature
  ✓ search_multi_collection has correct signature
  ✓ delete_collection has correct signature

Testing backward compatibility...
  ✓ search() maintains session_name parameter
  ✓ add_chunks() has optional collection_name

Testing configuration...
  ✓ DEFAULT_COLLECTION = medical_documents
  ✓ COLLECTION_NAME_PATTERN = ^[a-z0-9_]{3,50}$

============================================================
Summary
============================================================
Tests passed: 4/4

✅ All verification tests passed!
```

## Usage Guide

### Starting the Application

1. Start Qdrant (if not running):
```bash
docker-compose up -d qdrant
```

2. Start the application:
```bash
python3 main.py
```

3. Open browser: http://localhost:7860

### Working with Collections

#### Viewing Legacy Sessions

Legacy sessions (from before migration) appear with a `*` marker:
- Example: `minzdrav *` (legacy session)
- These are virtual collections stored in the `medical_documents` collection
- Fully functional for querying and uploading

#### Creating a New Collection

1. In the sidebar, find "➕ Создать коллекцию"
2. Enter a collection name (e.g., `clinical_en`)
   - Must be 3-50 characters
   - Only lowercase letters, numbers, and underscores
   - Valid: `clinical_en`, `drug_database`, `test_2025`
   - Invalid: `Clinical-EN`, `Test Collection`, `ab`
3. Click "Создать"
4. Collection appears in both dropdowns

#### Uploading Documents

1. Select target collection from "Коллекция для загрузки" dropdown
   - Can upload to both legacy sessions and new collections
2. Choose PDF or Markdown files
3. Click "Загрузить и обработать"
4. Documents process in background
5. Use "Обновить" button to check status

#### Querying Multiple Collections

1. In "Выбрать коллекции для поиска", select multiple collections
   - Can mix legacy sessions and new collections
   - Example: Select both `minzdrav *` and `clinical_en`
2. Ask your question in the chat
3. Answer will cite sources from all collections:
   ```
   [Collection: minzdrav, Document: протокол.pdf, Chunk 5]
   Согласно протоколу Минздрава...

   [Collection: clinical_en, Document: diabetes.md, Chunk 2]
   The treatment protocol includes...
   ```

#### Deleting Collections

1. Select collection from "Коллекция для загрузки" dropdown
2. Check "Подтвердить удаление" checkbox
3. Click "Удалить коллекцию"
4. **Note**: Legacy sessions can be deleted (removes data from medical_documents)
5. **Note**: New collections are fully deleted (collection + metadata)

### Document Table

The document table now shows:
- **Коллекция**: Which collection the document belongs to
- **Документ**: Document name
- **Статус**: Processing status (Queued, Processing, Completed, Error)
- **Создан**: Creation timestamp
- **Чанков**: Number of text chunks
- **Сущностей**: Number of extracted entities

## API Changes (for Developers)

### QdrantManager

**New Methods:**
```python
# Create a new collection
qdrant.create_collection("clinical_en")

# Get all collections (legacy + new)
collections = qdrant.get_collections()
# Returns: [{"name": "minzdrav", "type": "legacy", "doc_count": 5}, ...]

# Check if collection is legacy or new
coll_type = qdrant.get_collection_type("minzdrav")  # Returns: "legacy"

# Search across multiple collections
results = qdrant.search_multi_collection(
    query="diabetes treatment",
    collection_names=["minzdrav", "clinical_en"],
    k=5
)

# Delete collection
qdrant.delete_collection("clinical_en")
```

**Modified Methods:**
```python
# add_chunks now supports collection_name
qdrant.add_chunks(
    chunks,
    session_name="legacy_session",  # For legacy
    document_name="doc.pdf",
    entities_by_chunk=entities,
    collection_name=None  # Or specify new collection
)

# get_documents now supports collection_name
docs = qdrant.get_documents(collection_name="clinical_en")

# set_document_status supports both paradigms
qdrant.set_document_status(
    document_name="doc.pdf",
    status="completed",
    session_name="legacy",  # For legacy
    collection_name=None,   # Or for new collection
    chunks=10,
    entities=25
)
```

### DocumentService

```python
# Process document for specific collection
doc_service.process_document(
    file_path="doc.pdf",
    session_name=None,           # For legacy
    collection_name="clinical_en"  # For new collection
)

# Enqueue upload
doc_service.enqueue_upload(
    files=["doc1.pdf", "doc2.md"],
    session_name=None,           # For legacy
    collection_name="clinical_en"  # For new collection
)
```

### QAService

```python
# Answer question across multiple collections
answer, graph_html = qa_service.answer_question(
    question="What is the treatment?",
    collection_names=["minzdrav", "clinical_en"]
)

# Fast answer without graph
answer = qa_service.answer_question_fast(
    question="What is the treatment?",
    collection_names=["minzdrav", "clinical_en"]
)

# Generate graph only
graph_html = qa_service.generate_graph_from_question(
    question="What is the treatment?",
    collection_names=["minzdrav", "clinical_en"]
)
```

## Backward Compatibility

### Legacy Code Still Works

The following legacy code continues to work without changes:

```python
# Old session-based code
qdrant = QdrantManager()

# This still works (calls search_multi_collection internally)
results = qdrant.search("query", session_name="minzdrav", k=5)

# This still works
docs = qdrant.get_documents(session_name="minzdrav")

# This still works
qdrant.delete_session("minzdrav")
```

### Gradual Migration

You can migrate gradually:
1. Start with legacy sessions working as-is
2. Create new collections for new projects
3. Optionally migrate legacy data later
4. Mix legacy and new collections in queries

## Data Structure

### Metadata Collection

The `document_metadata` collection now stores:

**For legacy documents:**
```json
{
  "document_name": "протокол.pdf",
  "session_name": "minzdrav",
  "status": "completed",
  "chunks": 10,
  "entities": 25,
  "created_at": "2025-01-15T10:30:00Z",
  "updated_at": "2025-01-15T10:35:00Z"
}
```

**For new collection documents:**
```json
{
  "document_name": "diabetes.md",
  "collection_name": "clinical_en",
  "status": "completed",
  "chunks": 8,
  "entities": 15,
  "created_at": "2026-02-01T14:20:00Z",
  "updated_at": "2026-02-01T14:22:00Z"
}
```

### Document Chunks

**Legacy (in medical_documents collection):**
```json
{
  "text": "Chunk text...",
  "session_name": "minzdrav",
  "document_name": "протокол.pdf",
  "chunk_id": 0,
  "entities": [...],
  "entity_count": 5
}
```

**New collection (in dedicated collection):**
```json
{
  "text": "Chunk text...",
  "collection_name": "clinical_en",
  "document_name": "diabetes.md",
  "chunk_id": 0,
  "entities": [...],
  "entity_count": 3
}
```

## Troubleshooting

### Legacy Session Not Showing

**Problem**: Existing session doesn't appear in UI

**Solution**:
1. Check if documents exist in `medical_documents` collection
2. Verify `session_name` field is set on chunks
3. Refresh the UI

### Collection Name Validation Error

**Problem**: "Название должно содержать только латинские буквы..."

**Solution**:
- Use only lowercase letters, numbers, underscores
- Length: 3-50 characters
- Valid: `my_collection_2025`
- Invalid: `My Collection`, `ab`, `collection-name`

### Citations Not Showing Collection Name

**Problem**: Old citation format still appearing

**Solution**:
- This happens if using old `search()` method directly
- Use `search_multi_collection()` for proper citations
- UI automatically uses correct method

### Cannot Delete Legacy Session

**Problem**: Delete button disabled for legacy session

**Solution**:
- Legacy sessions can be deleted (removes data from medical_documents)
- New collections are fully deleted
- Both use the same "Удалить коллекцию" button
- Make sure to check "Подтвердить удаление"

## Testing Checklist

After migration, verify these features:

- [ ] UI loads successfully
- [ ] Legacy sessions appear with `*` marker
- [ ] Can create new collection
- [ ] Can upload to new collection
- [ ] Can upload to legacy session
- [ ] Document table shows collection column
- [ ] Can select multiple collections for query
- [ ] Can query single collection
- [ ] Can query multiple collections simultaneously
- [ ] Citations include collection name
- [ ] Citations include document name
- [ ] Graph shows entities from all selected collections
- [ ] Can delete new collection
- [ ] Can delete legacy session
- [ ] Refresh button updates document list

## Performance Notes

### Multi-Collection Search

- Each collection is searched with limit K
- Results are merged and sorted by score
- Returns top K*N results (N = number of collections)
- Recommended: Select max 5 collections for optimal performance

### Collection vs Session

**New Collections (Recommended):**
- Dedicated Qdrant collection per topic
- Faster search (no filtering needed)
- Better organization
- Can optimize per collection

**Legacy Sessions:**
- Stored in single `medical_documents` collection
- Requires filtering by `session_name`
- Slightly slower for large datasets
- Good for backward compatibility

## Future Enhancements

Potential features to add:

1. **Collection Export/Import**: Backup and restore collections
2. **Collection Merging**: Combine multiple collections
3. **Access Control**: Per-collection permissions
4. **Collection Metadata**: Description, tags, language
5. **Migration Tool**: Batch convert legacy sessions to collections
6. **Collection Templates**: Pre-configured collections
7. **Collection Stats**: Analytics and usage metrics
8. **Collection Versioning**: Track changes over time

## Support

For issues or questions:
1. Check this guide
2. Review `MIGRATION_COMPLETE.md` for technical details
3. Run `verify_migration.py` to check setup
4. Check logs in console output

## Rollback

If you need to rollback:

1. **Code Rollback**: Git revert changes
2. **Data Safety**: All legacy data in `medical_documents` is untouched
3. **New Collections**: Can be deleted without affecting legacy data
4. **No Data Migration**: Nothing to rollback data-wise

## Summary

The migration to collection-based architecture provides:

✅ Better organization with dedicated collections
✅ Multi-collection querying for comprehensive answers
✅ Enhanced citations with collection and document names
✅ Full backward compatibility with legacy sessions
✅ Zero data migration required
✅ Gradual migration path

Enjoy the new collection-based architecture! 🎉
