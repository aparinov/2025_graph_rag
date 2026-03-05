import { useEffect, useRef, useState } from "react";
import { Send, Loader2, FileText, ChevronDown } from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { ChatMessage, Source } from "../types";

interface Props {
  messages: ChatMessage[];
  onSend: (question: string) => void;
  loading: boolean;
  disabled: boolean;
}

/** Split a bilingual answer into RU / EN sections. */
function splitBilingual(text: string): { ru: string; en: string } | null {
  const ruMatch = text.match(/## RU\s*\n([\s\S]*?)(?=## EN|$)/);
  const enMatch = text.match(/## EN\s*\n([\s\S]*)/);
  if (ruMatch && enMatch) {
    return { ru: ruMatch[1].trim(), en: enMatch[1].trim() };
  }
  return null;
}

function SourcesCitation({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);

  if (!sources.length) return null;

  return (
    <div className="mt-3" style={{ borderTop: "1px solid var(--border)" }}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-1.5 pt-2 text-xs font-medium"
        style={{ color: "var(--text-secondary)" }}
      >
        <FileText className="h-3 w-3" />
        Источники ({sources.length})
        <ChevronDown
          className={`ml-auto h-3 w-3 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>

      {open && (
        <div className="mt-2 space-y-2">
          {sources.map((s, i) => (
            <blockquote
              key={i}
              className="rounded-md border-l-3 px-3 py-2 text-xs"
              style={{
                borderColor: "var(--accent)",
                background: "var(--bg-page)",
                color: "var(--text-secondary)",
              }}
            >
              <div className="mb-1 flex items-center gap-2 font-medium" style={{ color: "var(--text-primary)" }}>
                <FileText className="h-3 w-3 shrink-0" style={{ color: "var(--accent)" }} />
                {s.file_url ? (
                  <a
                    href={`/viewer.html?url=${encodeURIComponent(s.file_url)}&name=${encodeURIComponent(s.document)}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline hover:opacity-80"
                    style={{ color: "var(--accent)" }}
                  >
                    {s.document}
                  </a>
                ) : (
                  s.document
                )}
                <span className="ml-auto font-normal" style={{ color: "var(--text-muted)" }}>
                  {s.collection} &middot; {(s.score * 100).toFixed(0)}%
                </span>
              </div>
              <p className="leading-relaxed italic">"{s.text}..."</p>
            </blockquote>
          ))}
        </div>
      )}
    </div>
  );
}

function AssistantMessage({ content, sources }: { content: string; sources?: Source[] }) {
  const bilingual = splitBilingual(content);

  const body = bilingual ? (
    <div className="grid grid-cols-2 gap-4">
      <div>
        <div
          className="mb-2 inline-block rounded px-2 py-0.5 text-xs font-semibold uppercase"
          style={{ background: "var(--accent)", color: "var(--text-on-primary)" }}
        >
          RU
        </div>
        <div className="prose prose-sm max-w-none dark:prose-invert">
          <ReactMarkdown>{bilingual.ru}</ReactMarkdown>
        </div>
      </div>
      <div
        className="pl-4"
        style={{ borderLeft: "1px solid var(--border)" }}
      >
        <div
          className="mb-2 inline-block rounded px-2 py-0.5 text-xs font-semibold uppercase"
          style={{ background: "var(--bg-input)", color: "var(--text-secondary)" }}
        >
          EN
        </div>
        <div className="prose prose-sm max-w-none dark:prose-invert">
          <ReactMarkdown>{bilingual.en}</ReactMarkdown>
        </div>
      </div>
    </div>
  ) : (
    <div className="prose prose-sm max-w-none dark:prose-invert">
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );

  return (
    <div>
      {body}
      {sources && <SourcesCitation sources={sources} />}
    </div>
  );
}

export default function Chat({ messages, onSend, loading, disabled }: Props) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const send = () => {
    const q = input.trim();
    if (!q || loading || disabled) return;
    setInput("");
    onSend(q);
  };

  return (
    <div className="flex h-full flex-col">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {messages.length === 0 && (
          <div className="flex h-full items-center justify-center" style={{ color: "var(--text-muted)" }}>
            {disabled
              ? "Выберите коллекцию, чтобы начать диалог."
              : "Задайте вопрос по вашим документам."}
          </div>
        )}

        <div className="mx-auto max-w-4xl space-y-4">
          {messages.map((m, i) => (
            <div
              key={i}
              className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`rounded-lg px-4 py-3 ${m.role === "user" ? "max-w-[70%]" : "max-w-full"}`}
                style={{
                  background:
                    m.role === "user" ? "var(--bg-user-msg)" : "var(--bg-assist-msg)",
                  color:
                    m.role === "user" ? "var(--text-on-primary)" : "var(--text-primary)",
                }}
              >
                {m.role === "assistant" ? (
                  <AssistantMessage content={m.content} sources={m.sources} />
                ) : (
                  <p className="text-sm whitespace-pre-wrap">{m.content}</p>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div
                className="flex items-center gap-2 rounded-lg px-4 py-3 text-sm"
                style={{ background: "var(--bg-card)", color: "var(--text-secondary)" }}
              >
                <Loader2 className="h-4 w-4 animate-spin" />
                Думаю...
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input */}
      <div className="shrink-0 px-6 py-3" style={{ borderTop: "1px solid var(--border)" }}>
        <div className="mx-auto flex max-w-4xl gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
            placeholder={
              disabled
                ? "Сначала выберите коллекцию..."
                : "Спросите о ваших документах..."
            }
            disabled={disabled}
            className="min-w-0 flex-1 rounded-lg px-4 py-2.5 text-sm outline-none disabled:opacity-50"
            style={{ background: "var(--bg-input)", color: "var(--text-primary)" }}
          />
          <button
            onClick={send}
            disabled={loading || disabled || !input.trim()}
            className="flex items-center gap-1 rounded-lg px-4 py-2.5 text-sm font-medium text-white disabled:opacity-50"
            style={{ background: "var(--accent)" }}
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
