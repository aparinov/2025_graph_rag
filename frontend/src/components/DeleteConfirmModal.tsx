import { AlertTriangle } from "lucide-react";

interface Props {
  name: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function DeleteConfirmModal({ name, onConfirm, onCancel }: Props) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: "var(--bg-overlay)" }}
    >
      <div
        className="mx-4 w-full max-w-sm rounded-lg p-6 shadow-xl"
        style={{ background: "var(--bg-modal)" }}
      >
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-red-100 dark:bg-red-900/50">
            <AlertTriangle className="h-5 w-5 text-red-500 dark:text-red-400" />
          </div>
          <div>
            <h3 className="text-sm font-semibold">Удаление коллекции</h3>
            <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
              Это действие нельзя отменить.
            </p>
          </div>
        </div>

        <p className="mb-6 text-sm" style={{ color: "var(--text-secondary)" }}>
          Вы уверены, что хотите удалить{" "}
          <span className="font-medium" style={{ color: "var(--text-primary)" }}>
            &laquo;{name}&raquo;
          </span>{" "}
          и все её документы?
        </p>

        <div className="flex justify-end gap-2">
          <button
            onClick={onCancel}
            className="rounded px-3 py-1.5 text-sm hover:opacity-80"
            style={{ color: "var(--text-secondary)" }}
          >
            Отмена
          </button>
          <button
            onClick={onConfirm}
            className="rounded px-3 py-1.5 text-sm font-medium text-white"
            style={{ background: "var(--danger)" }}
          >
            Удалить
          </button>
        </div>
      </div>
    </div>
  );
}
