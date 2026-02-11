import { RefreshCw } from "lucide-react";
import type { Document } from "../types";

interface Props {
  docs: Document[];
  onRefresh: () => void;
}

const STATUS_BADGE: Record<string, string> = {
  completed: "bg-green-900 text-green-300",
  processing: "bg-yellow-900 text-yellow-300",
  queued: "bg-blue-900 text-blue-300",
  error: "bg-red-900 text-red-300",
  unknown: "bg-gray-700 text-gray-400",
};

export default function DocumentsPanel({ docs, onRefresh }: Props) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-center justify-between px-6 py-3">
        <span className="text-sm text-gray-400">
          {docs.length} document{docs.length !== 1 && "s"}
        </span>
        <button
          onClick={onRefresh}
          className="flex items-center gap-1 rounded px-2 py-1 text-xs text-gray-400 hover:bg-gray-800 hover:text-gray-200"
        >
          <RefreshCw className="h-3 w-3" /> Refresh
        </button>
      </div>

      <div className="flex-1 overflow-auto px-6 pb-4">
        {docs.length === 0 ? (
          <div className="flex h-32 items-center justify-center text-sm text-gray-600">
            No documents in selected collections.
          </div>
        ) : (
          <table className="w-full text-left text-sm">
            <thead className="sticky top-0 bg-gray-950 text-xs uppercase text-gray-500">
              <tr>
                <th className="px-3 py-2">Collection</th>
                <th className="px-3 py-2">Document</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2">Created</th>
                <th className="px-3 py-2 text-right">Chunks</th>
                <th className="px-3 py-2 text-right">Entities</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {docs.map((d, i) => (
                <tr key={`${d.collection}-${d.name}-${i}`} className="hover:bg-gray-900">
                  <td className="px-3 py-2 text-gray-300">{d.collection}</td>
                  <td className="px-3 py-2 font-medium">{d.name}</td>
                  <td className="px-3 py-2">
                    <span
                      className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_BADGE[d.status] || STATUS_BADGE.unknown}`}
                      title={d.error_message || undefined}
                    >
                      {d.status}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-gray-400">{d.created_at}</td>
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
