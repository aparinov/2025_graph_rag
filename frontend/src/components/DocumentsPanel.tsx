import { RefreshCw } from "lucide-react";
import type { Document } from "../types";

interface Props {
  docs: Document[];
  onRefresh: () => void;
}

const STATUS_LABEL: Record<string, string> = {
  completed: "Готово",
  processing: "Обработка",
  queued: "В очереди",
  error: "Ошибка",
  unknown: "Неизвестно",
};

const STATUS_BADGE_DARK: Record<string, string> = {
  completed: "bg-green-900 text-green-300",
  processing: "bg-yellow-900 text-yellow-300",
  queued: "bg-blue-900 text-blue-300",
  error: "bg-red-900 text-red-300",
  unknown: "bg-gray-700 text-gray-400",
};

const STATUS_BADGE_LIGHT: Record<string, string> = {
  completed: "bg-green-100 text-green-800",
  processing: "bg-yellow-100 text-yellow-800",
  queued: "bg-blue-100 text-blue-800",
  error: "bg-red-100 text-red-800",
  unknown: "bg-gray-100 text-gray-600",
};

function pluralDocs(n: number): string {
  if (n % 10 === 1 && n % 100 !== 11) return `${n} документ`;
  if ([2, 3, 4].includes(n % 10) && ![12, 13, 14].includes(n % 100))
    return `${n} документа`;
  return `${n} документов`;
}

export default function DocumentsPanel({ docs, onRefresh }: Props) {
  const isDark = document.documentElement.classList.contains("dark");
  const badges = isDark ? STATUS_BADGE_DARK : STATUS_BADGE_LIGHT;

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center justify-between px-6 py-3">
        <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
          {pluralDocs(docs.length)}
        </span>
        <button
          onClick={onRefresh}
          className="flex items-center gap-1 rounded px-2 py-1 text-xs hover:opacity-80"
          style={{ color: "var(--text-secondary)" }}
        >
          <RefreshCw className="h-3 w-3" /> Обновить
        </button>
      </div>

      <div className="flex-1 overflow-auto px-6 pb-4">
        {docs.length === 0 ? (
          <div className="flex h-32 items-center justify-center text-sm" style={{ color: "var(--text-muted)" }}>
            Нет документов в выбранных коллекциях.
          </div>
        ) : (
          <table className="w-full text-left text-sm">
            <thead className="sticky top-0 text-xs uppercase" style={{ background: "var(--bg-page)", color: "var(--text-muted)" }}>
              <tr>
                <th className="px-3 py-2">Коллекция</th>
                <th className="px-3 py-2">Документ</th>
                <th className="px-3 py-2">Статус</th>
                <th className="px-3 py-2">Создан</th>
                <th className="px-3 py-2 text-right">Чанков</th>
                <th className="px-3 py-2 text-right">Сущностей</th>
              </tr>
            </thead>
            <tbody>
              {docs.map((d, i) => (
                <tr
                  key={`${d.collection}-${d.name}-${i}`}
                  className="hover:opacity-80"
                  style={{ borderBottom: "1px solid var(--border)" }}
                >
                  <td className="px-3 py-2" style={{ color: "var(--text-secondary)" }}>
                    {d.collection}
                  </td>
                  <td className="px-3 py-2 font-medium">{d.name}</td>
                  <td className="px-3 py-2">
                    <span
                      className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${badges[d.status] || badges.unknown}`}
                      title={d.error_message || undefined}
                    >
                      {STATUS_LABEL[d.status] || d.status}
                    </span>
                  </td>
                  <td className="px-3 py-2" style={{ color: "var(--text-secondary)" }}>
                    {d.created_at}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">{d.chunks}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{d.entities}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
