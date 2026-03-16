export interface Category {
  id: number;
  name: string;
  color: string | null;
  created_at: string;
  document_count: number;
}

export interface Document {
  id: number;
  title: string;
  file_path: string;
  file_type: string;
  content: string | null;
  category_id: number | null;
  status: "pending" | "processing" | "completed" | "failed";
  error_message: string | null;
  created_at: string;
  updated_at: string;
  category: Category | null;
}

export interface QAResult {
  answer: string;
  sources: QASource[];
  question: string;
}

export interface QASource {
  title: string;
  content: string;
  file_path: string;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
}

export interface GroupedDocuments {
  category: Category | null;
  documents: Document[];
}
