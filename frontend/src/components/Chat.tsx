import { useEffect, useRef, useState } from "react";
import { Send, Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { ChatMessage } from "../types";

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

function AssistantMessage({ content }: { content: string }) {
  const bilingual = splitBilingual(content);
  const [lang, setLang] = useState<"ru" | "en">("ru");

  if (!bilingual) {
    return (
      <div className="prose prose-invert max-w-none text-sm">
        <ReactMarkdown>{content}</ReactMarkdown>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-2 flex gap-1">
        <button
          onClick={() => setLang("ru")}
          className={`rounded px-2 py-0.5 text-xs font-medium ${
            lang === "ru"
              ? "bg-blue-600 text-white"
              : "bg-gray-700 text-gray-300 hover:bg-gray-600"
          }`}
        >
          RU
        </button>
        <button
          onClick={() => setLang("en")}
          className={`rounded px-2 py-0.5 text-xs font-medium ${
            lang === "en"
              ? "bg-blue-600 text-white"
              : "bg-gray-700 text-gray-300 hover:bg-gray-600"
          }`}
        >
          EN
        </button>
      </div>
      <div className="prose prose-invert max-w-none text-sm">
        <ReactMarkdown>{lang === "ru" ? bilingual.ru : bilingual.en}</ReactMarkdown>
      </div>
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
          <div className="flex h-full items-center justify-center text-gray-600">
            {disabled
              ? "Select a collection to start chatting."
              : "Ask a question about your documents."}
          </div>
        )}

        <div className="mx-auto max-w-3xl space-y-4">
          {messages.map((m, i) => (
            <div
              key={i}
              className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-lg px-4 py-3 ${
                  m.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-800 text-gray-100"
                }`}
              >
                {m.role === "assistant" ? (
                  <AssistantMessage content={m.content} />
                ) : (
                  <p className="text-sm whitespace-pre-wrap">{m.content}</p>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="flex items-center gap-2 rounded-lg bg-gray-800 px-4 py-3 text-sm text-gray-400">
                <Loader2 className="h-4 w-4 animate-spin" />
                Thinking...
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input */}
      <div className="shrink-0 border-t border-gray-800 px-6 py-3">
        <div className="mx-auto flex max-w-3xl gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
            placeholder={
              disabled
                ? "Select a collection first..."
                : "Ask about your documents..."
            }
            disabled={disabled}
            className="min-w-0 flex-1 rounded-lg bg-gray-800 px-4 py-2.5 text-sm outline-none placeholder:text-gray-500 focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
          />
          <button
            onClick={send}
            disabled={loading || disabled || !input.trim()}
            className="flex items-center gap-1 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium hover:bg-blue-500 disabled:opacity-50"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
