import { useState, useRef } from "react";
import { RefreshCw } from "lucide-react";
import type { Collection } from "../types";
import * as api from "../api";

interface Props {
  collections: Collection[];
  onClose: () => void;
}

export default function MigrateModal({ collections, onClose }: Props) {
  const [target, setTarget] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<{
    matched: string[];
    unmatched_deleted: string[];
  } | null>(null);
  const [error, setError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const handleMigrate = async () => {
    const files = fileRef.current?.files;
    if (!files?.length || !target) return;
    setRunning(true);
    setError("");
    setResult(null);
    try {
      const res = await api.migrateFiles(files, target);
      setResult(res);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: "var(--bg-overlay)" }}
    >
      <div
        className="mx-4 w-full max-w-md rounded-lg p-6 shadow-xl"
        style={{ background: "var(--bg-modal)" }}
      >
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900/50">
            <RefreshCw className="h-5 w-5 text-blue-500 dark:text-blue-400" />
          </div>
          <div>
            <h3 className="text-sm font-semibold">Миграция файлов</h3>
            <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
              Обновить file_name в Qdrant без переиндексации
            </p>
          </div>
        </div>

        <div className="mb-3">
          <label className="mb-1 block text-xs" style={{ color: "var(--text-secondary)" }}>
            Коллекция
          </label>
          <select
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            className="w-full rounded px-2 py-1.5 text-sm outline-none"
            style={{ background: "var(--bg-input)", color: "var(--text-primary)" }}
          >
            <option value="">Выберите коллекцию...</option>
            {collections.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </select>
        </div>

        <div className="mb-4">
          <label className="mb-1 block text-xs" style={{ color: "var(--text-secondary)" }}>
            Файлы
          </label>
          <input
            ref={fileRef}
            type="file"
            multiple
            accept=".pdf,.md,.markdown"
            className="w-full text-xs file:mr-2 file:rounded file:border-0 file:px-2 file:py-1 file:text-xs"
            style={{ color: "var(--text-secondary)" }}
          />
        </div>

        {result && (
          <div className="mb-4 rounded p-3 text-xs" style={{ background: "var(--bg-card)" }}>
            {result.matched.length > 0 && (
              <div className="mb-2">
                <span className="font-medium text-green-500">Обновлено ({result.matched.length}):</span>
                <ul className="ml-3 mt-1 list-disc" style={{ color: "var(--text-secondary)" }}>
                  {result.matched.map((f) => (
                    <li key={f}>{f}</li>
                  ))}
                </ul>
              </div>
            )}
            {result.unmatched_deleted.length > 0 && (
              <div>
                <span className="font-medium text-yellow-500">
                  Не найдены, удалены ({result.unmatched_deleted.length}):
                </span>
                <ul className="ml-3 mt-1 list-disc" style={{ color: "var(--text-secondary)" }}>
                  {result.unmatched_deleted.map((f) => (
                    <li key={f}>{f}</li>
                  ))}
                </ul>
              </div>
            )}
            {result.matched.length === 0 && result.unmatched_deleted.length === 0 && (
              <p style={{ color: "var(--text-muted)" }}>Нет результатов.</p>
            )}
          </div>
        )}

        {error && <p className="mb-4 text-xs text-red-500">{error}</p>}

        <div className="flex justify-end gap-2">
          <button
            onClick={onClose}
            className="rounded px-3 py-1.5 text-sm hover:opacity-80"
            style={{ color: "var(--text-secondary)" }}
          >
            Закрыть
          </button>
          <button
            onClick={handleMigrate}
            disabled={running || !target}
            className="rounded px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
            style={{ background: "var(--accent)" }}
          >
            {running ? "Миграция..." : "Начать миграцию"}
          </button>
        </div>
      </div>
    </div>
  );
}
