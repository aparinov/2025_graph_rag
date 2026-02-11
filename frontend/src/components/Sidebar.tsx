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
    <aside
      className="flex w-64 shrink-0 flex-col"
      style={{ background: "var(--bg-sidebar)", borderRight: "1px solid var(--border)" }}
    >
      {/* Collections list */}
      <div className="flex-1 overflow-y-auto p-4">
        <h2
          className="mb-3 text-xs font-semibold uppercase tracking-wider"
          style={{ color: "var(--text-muted)" }}
        >
          Коллекции
        </h2>

        {collections.length === 0 && (
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            Коллекций пока нет.
          </p>
        )}

        <div className="space-y-1">
          {collections.map((c) => (
            <label
              key={c.name}
              className="flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 text-sm hover:opacity-80"
            >
              <input
                type="checkbox"
                checked={selected.has(c.name)}
                onChange={() => onToggle(c.name)}
                className="accent-[var(--accent)]"
              />
              <span className="truncate">{c.name}</span>
              {c.type === "legacy" && (
                <span className="ml-auto text-[10px]" style={{ color: "var(--text-muted)" }}>
                  legacy
                </span>
              )}
            </label>
          ))}
        </div>
      </div>

      {/* Management accordion */}
      <div style={{ borderTop: "1px solid var(--border)" }}>
        <details className="group">
          <summary
            className="flex cursor-pointer items-center gap-2 px-4 py-3 text-xs font-semibold uppercase tracking-wider hover:opacity-80"
            style={{ color: "var(--text-muted)" }}
          >
            <ChevronDown className="h-3.5 w-3.5 transition-transform group-open:rotate-180" />
            Управление
          </summary>

          <div className="space-y-4 px-4 pb-4">
            {/* Create */}
            <div>
              <div className="mb-1 flex items-center gap-1 text-xs" style={{ color: "var(--text-secondary)" }}>
                <Plus className="h-3 w-3" /> Новая коллекция
              </div>
              <div className="flex gap-1">
                <input
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleCreate()}
                  placeholder="my_collection"
                  className="min-w-0 flex-1 rounded px-2 py-1 text-sm outline-none"
                  style={{ background: "var(--bg-input)", color: "var(--text-primary)" }}
                />
                <button
                  onClick={handleCreate}
                  disabled={creating}
                  className="rounded px-2 py-1 text-xs font-medium text-white disabled:opacity-50"
                  style={{ background: "var(--accent)" }}
                >
                  {creating ? "..." : "Создать"}
                </button>
              </div>
              {createError && <p className="mt-1 text-xs text-red-500">{createError}</p>}
            </div>

            {/* Upload */}
            <div>
              <div className="mb-1 flex items-center gap-1 text-xs" style={{ color: "var(--text-secondary)" }}>
                <Upload className="h-3 w-3" /> Загрузка документов
              </div>
              <select
                value={uploadTarget}
                onChange={(e) => setUploadTarget(e.target.value)}
                className="mb-1 w-full rounded px-2 py-1 text-sm outline-none"
                style={{ background: "var(--bg-input)", color: "var(--text-primary)" }}
              >
                <option value="">Выберите коллекцию...</option>
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
                className="mb-1 w-full text-xs file:mr-2 file:rounded file:border-0 file:px-2 file:py-1 file:text-xs"
                style={{ color: "var(--text-secondary)" }}
              />
              <button
                onClick={handleUpload}
                disabled={uploading || !uploadTarget}
                className="w-full rounded px-2 py-1 text-xs font-medium text-white disabled:opacity-50"
                style={{ background: "var(--accent)" }}
              >
                {uploading ? "Загрузка..." : "Загрузить и обработать"}
              </button>
              {uploadError && <p className="mt-1 text-xs text-red-500">{uploadError}</p>}
            </div>

            {/* Delete */}
            <div>
              <div className="mb-1 flex items-center gap-1 text-xs" style={{ color: "var(--text-secondary)" }}>
                <Trash2 className="h-3 w-3" /> Удалить коллекцию
              </div>
              {collections.map((c) => (
                <button
                  key={c.name}
                  onClick={() => onDeleteRequest(c.name)}
                  className="mb-1 flex w-full items-center gap-2 rounded border px-2 py-1 text-left text-sm hover:border-red-500 hover:text-red-500"
                  style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
                >
                  <Trash2 className="h-3 w-3 shrink-0" />
                  <span className="truncate">{c.name}</span>
                </button>
              ))}
              {collections.length === 0 && (
                <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                  Нет коллекций для удаления.
                </p>
              )}
            </div>
          </div>
        </details>
      </div>
    </aside>
  );
}
