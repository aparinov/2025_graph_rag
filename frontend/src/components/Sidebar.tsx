import { useState, useRef } from "react";
import { Plus, Upload, Trash2, ChevronDown } from "lucide-react";
import type { Collection } from "../types";

interface Props {
  collections: Collection[];
  selected: Set<string>;
  onToggle: (name: string) => void;
  onCreate: (name: string) => Promise<void>;
  onUpload: (files: FileList, collection: string) => Promise<void>;
  onDeleteRequest: (name: string) => void;
}

export default function Sidebar({
  collections,
  selected,
  onToggle,
  onCreate,
  onUpload,
  onDeleteRequest,
}: Props) {
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadTarget, setUploadTarget] = useState("");
  const [createError, setCreateError] = useState("");
  const [uploadError, setUploadError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setCreating(true);
    setCreateError("");
    try {
      await onCreate(newName.trim().toLowerCase());
      setNewName("");
    } catch (e: any) {
      setCreateError(e.message);
    } finally {
      setCreating(false);
    }
  };

  const handleUpload = async () => {
    const files = fileRef.current?.files;
    if (!files?.length || !uploadTarget) return;
    setUploading(true);
    setUploadError("");
    try {
      await onUpload(files, uploadTarget);
      if (fileRef.current) fileRef.current.value = "";
    } catch (e: any) {
      setUploadError(e.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-gray-800 bg-gray-900">
      {/* Collections list */}
      <div className="flex-1 overflow-y-auto p-4">
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">
          Collections
        </h2>

        {collections.length === 0 && (
          <p className="text-sm text-gray-500">No collections yet.</p>
        )}

        <div className="space-y-1">
          {collections.map((c) => (
            <label
              key={c.name}
              className="flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-gray-800"
            >
              <input
                type="checkbox"
                checked={selected.has(c.name)}
                onChange={() => onToggle(c.name)}
                className="accent-blue-500"
              />
              <span className="truncate">{c.name}</span>
              {c.type === "legacy" && (
                <span className="ml-auto text-[10px] text-gray-600">legacy</span>
              )}
            </label>
          ))}
        </div>
      </div>

      {/* Management accordion */}
      <div className="border-t border-gray-800">
        <details className="group">
          <summary className="flex cursor-pointer items-center gap-2 px-4 py-3 text-xs font-semibold uppercase tracking-wider text-gray-500 hover:text-gray-300">
            <ChevronDown className="h-3.5 w-3.5 transition-transform group-open:rotate-180" />
            Manage
          </summary>

          <div className="space-y-4 px-4 pb-4">
            {/* Create collection */}
            <div>
              <div className="mb-1 flex items-center gap-1 text-xs text-gray-400">
                <Plus className="h-3 w-3" /> New collection
              </div>
              <div className="flex gap-1">
                <input
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleCreate()}
                  placeholder="my_collection"
                  className="min-w-0 flex-1 rounded bg-gray-800 px-2 py-1 text-sm outline-none focus:ring-1 focus:ring-blue-500"
                />
                <button
                  onClick={handleCreate}
                  disabled={creating}
                  className="rounded bg-blue-600 px-2 py-1 text-xs font-medium hover:bg-blue-500 disabled:opacity-50"
                >
                  {creating ? "..." : "Create"}
                </button>
              </div>
              {createError && (
                <p className="mt-1 text-xs text-red-400">{createError}</p>
              )}
            </div>

            {/* Upload */}
            <div>
              <div className="mb-1 flex items-center gap-1 text-xs text-gray-400">
                <Upload className="h-3 w-3" /> Upload documents
              </div>
              <select
                value={uploadTarget}
                onChange={(e) => setUploadTarget(e.target.value)}
                className="mb-1 w-full rounded bg-gray-800 px-2 py-1 text-sm outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="">Select collection...</option>
                {collections.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name}
                  </option>
                ))}
              </select>
              <input
                ref={fileRef}
                type="file"
                multiple
                accept=".pdf,.md,.markdown"
                className="mb-1 w-full text-xs text-gray-400 file:mr-2 file:rounded file:border-0 file:bg-gray-700 file:px-2 file:py-1 file:text-xs file:text-gray-200"
              />
              <button
                onClick={handleUpload}
                disabled={uploading || !uploadTarget}
                className="w-full rounded bg-blue-600 px-2 py-1 text-xs font-medium hover:bg-blue-500 disabled:opacity-50"
              >
                {uploading ? "Uploading..." : "Upload & Process"}
              </button>
              {uploadError && (
                <p className="mt-1 text-xs text-red-400">{uploadError}</p>
              )}
            </div>

            {/* Delete */}
            <div>
              <div className="mb-1 flex items-center gap-1 text-xs text-gray-400">
                <Trash2 className="h-3 w-3" /> Delete collection
              </div>
              {collections.map((c) => (
                <button
                  key={c.name}
                  onClick={() => onDeleteRequest(c.name)}
                  className="mb-1 flex w-full items-center gap-2 rounded border border-gray-700 px-2 py-1 text-left text-sm text-gray-300 hover:border-red-600 hover:text-red-400"
                >
                  <Trash2 className="h-3 w-3 shrink-0" />
                  <span className="truncate">{c.name}</span>
                </button>
              ))}
              {collections.length === 0 && (
                <p className="text-xs text-gray-600">No collections to delete.</p>
              )}
            </div>
          </div>
        </details>
      </div>
    </aside>
  );
}
