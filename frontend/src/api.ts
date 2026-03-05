import type { Collection, Document, Source } from "./types";

const BASE = "/api";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

export function fetchCollections() {
  return request<Collection[]>("/collections");
}

export function createCollection(name: string) {
  return request<{ status: string; name: string }>("/collections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export function deleteCollection(name: string) {
  return request<{ status: string }>(`/collections/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

export function fetchDocuments(collectionName: string) {
  return request<Document[]>(
    `/collections/${encodeURIComponent(collectionName)}/documents`,
  );
}

export async function uploadFiles(files: FileList, collection: string) {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  form.append("collection", collection);

  return request<{ status: string; job_id: string; file_count: number }>(
    "/upload",
    { method: "POST", body: form },
  );
}

export function migrateFiles(files: FileList, collection: string) {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  form.append("collection", collection);

  return request<{ matched: string[]; unmatched_deleted: string[] }>(
    "/migrate-files",
    { method: "POST", body: form },
  );
}

export function sendChat(question: string, collections: string[]) {
  return request<{ answer: string; sources: Source[] }>("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, collections }),
  });
}
