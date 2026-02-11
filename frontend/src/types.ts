export interface Collection {
  name: string;
  type: "legacy" | "collection";
  doc_count: number;
}

export interface Document {
  collection: string;
  name: string;
  status: string;
  created_at: string;
  chunks: number;
  entities: number;
  error_message: string;
}

export interface Source {
  collection: string;
  document: string;
  chunk_id: number;
  text: string;
  score: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}
