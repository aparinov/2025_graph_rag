import { useCallback, useEffect, useRef, useState } from "react";
import { Moon, Sun } from "lucide-react";
import type { Collection, ChatMessage, Document } from "./types";
import * as api from "./api";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";
import DocumentsPanel from "./components/DocumentsPanel";
import DeleteConfirmModal from "./components/DeleteConfirmModal";

type Tab = "chat" | "docs";

export default function App() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [tab, setTab] = useState<Tab>("chat");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [docs, setDocs] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [dark, setDark] = useState(true);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Theme ──
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  // ── Load collections on mount ──
  const refreshCollections = useCallback(async () => {
    try {
      const data = await api.fetchCollections();
      setCollections(data);
    } catch (e) {
      console.error("Failed to fetch collections", e);
    }
  }, []);

  useEffect(() => {
    refreshCollections();
  }, [refreshCollections]);

  // ── Load documents when selection changes ──
  const refreshDocs = useCallback(async () => {
    if (selected.size === 0) {
      setDocs([]);
      return;
    }
    try {
      const allDocs = await Promise.all(
        [...selected].map((c) => api.fetchDocuments(c)),
      );
      setDocs(allDocs.flat());
    } catch (e) {
      console.error("Failed to fetch documents", e);
    }
  }, [selected]);

  useEffect(() => {
    refreshDocs();
  }, [refreshDocs]);

  // ── Auto-refresh docs every 5s when there are processing/queued docs ──
  useEffect(() => {
    const hasActive = docs.some(
      (d) => d.status === "queued" || d.status === "processing",
    );
    if (hasActive) {
      pollRef.current = setInterval(refreshDocs, 5000);
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [docs, refreshDocs]);

  const toggleCollection = (name: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const handleCreate = async (name: string) => {
    await api.createCollection(name);
    await refreshCollections();
  };

  const handleUpload = async (files: FileList, collection: string) => {
    await api.uploadFiles(files, collection);
    await refreshDocs();
  };

  const confirmDelete = async () => {
    if (!deleteTarget) return;
    await api.deleteCollection(deleteTarget);
    setSelected((prev) => {
      const next = new Set(prev);
      next.delete(deleteTarget);
      return next;
    });
    setDeleteTarget(null);
    await refreshCollections();
    await refreshDocs();
  };

  const handleSend = async (question: string) => {
    const userMsg: ChatMessage = { role: "user", content: question };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    try {
      const { answer, sources } = await api.sendChat(question, [...selected]);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: answer, sources },
      ]);
    } catch (e: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Ошибка: ${e.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen flex-col" style={{ background: "var(--bg-page)", color: "var(--text-primary)" }}>
      {/* Header */}
      <header
        className="flex shrink-0 items-center justify-between px-6 py-3"
        style={{ borderBottom: "1px solid var(--border)" }}
      >
        <h1 className="text-lg font-semibold tracking-tight">MedGraph RAG</h1>
        <button
          onClick={() => setDark((d) => !d)}
          className="rounded-lg p-2 transition-colors hover:opacity-80"
          style={{ background: "var(--bg-card)" }}
          title={dark ? "Светлая тема" : "Тёмная тема"}
        >
          {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
      </header>

      <div className="flex min-h-0 flex-1">
        <Sidebar
          collections={collections}
          selected={selected}
          onToggle={toggleCollection}
          onCreate={handleCreate}
          onUpload={handleUpload}
          onDeleteRequest={setDeleteTarget}
        />

        <main className="flex min-w-0 flex-1 flex-col">
          {/* Tabs */}
          <div className="flex shrink-0" style={{ borderBottom: "1px solid var(--border)" }}>
            <button
              className={`px-5 py-2.5 text-sm font-medium transition-colors ${
                tab === "chat"
                  ? "border-b-2 text-[var(--accent)]"
                  : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              }`}
              style={tab === "chat" ? { borderColor: "var(--accent)" } : undefined}
              onClick={() => setTab("chat")}
            >
              Чат
            </button>
            <button
              className={`px-5 py-2.5 text-sm font-medium transition-colors ${
                tab === "docs"
                  ? "border-b-2 text-[var(--accent)]"
                  : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              }`}
              style={tab === "docs" ? { borderColor: "var(--accent)" } : undefined}
              onClick={() => setTab("docs")}
            >
              Документы
            </button>
          </div>

          <div className="min-h-0 flex-1">
            {tab === "chat" ? (
              <Chat
                messages={messages}
                onSend={handleSend}
                loading={loading}
                disabled={selected.size === 0}
              />
            ) : (
              <DocumentsPanel docs={docs} onRefresh={refreshDocs} />
            )}
          </div>
        </main>
      </div>

      {deleteTarget && (
        <DeleteConfirmModal
          name={deleteTarget}
          onConfirm={confirmDelete}
          onCancel={() => setDeleteTarget(null)}
        />
      )}
    </div>
  );
}
