"""SQLite-backed metadata index for hard filters and candidate scope resolution."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _lookup_docstore_item(docstore: Any, docstore_id: Any) -> Any | None:
    internal_dict = getattr(docstore, "_dict", None)
    if isinstance(internal_dict, dict):
        return internal_dict.get(docstore_id)
    search_fn = getattr(docstore, "search", None)
    if callable(search_fn):
        try:
            return search_fn(docstore_id)
        except Exception:
            return None
    return None


def _build_vectorstore_chunk_map(vectorstore: Any) -> dict[str, dict[str, Any]]:
    mapping = getattr(vectorstore, "index_to_docstore_id", None)
    docstore = getattr(vectorstore, "docstore", None)
    if mapping is None or docstore is None:
        return {}

    if isinstance(mapping, dict):
        items = mapping.items()
    elif isinstance(mapping, list):
        items = enumerate(mapping)
    else:
        return {}

    chunk_map: dict[str, dict[str, Any]] = {}
    for row_id, docstore_id in items:
        doc = _lookup_docstore_item(docstore, docstore_id)
        if doc is None:
            continue
        metadata = dict(getattr(doc, "metadata", {}) or {})
        chunk_id = str(metadata.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        chunk_map[chunk_id] = {
            "faiss_row_id": int(row_id),
            "docstore_id": str(docstore_id),
        }
    return chunk_map


def rebuild_metadata_store(db_path: Path, splits: list[Any], vectorstore: Any | None = None) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore_chunk_map = _build_vectorstore_chunk_map(vectorstore) if vectorstore is not None else {}
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.executescript(
            """
            DROP TABLE IF EXISTS papers;
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS paper_authors;
            DROP TABLE IF EXISTS paper_tags;
            DROP TABLE IF EXISTS paper_collections;

            CREATE TABLE papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT,
                year INTEGER,
                venue TEXT
            );

            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                faiss_row_id INTEGER,
                docstore_id TEXT,
                section TEXT,
                page_start INTEGER,
                page_end INTEGER
            );

            CREATE TABLE paper_authors (
                paper_id TEXT NOT NULL,
                author TEXT NOT NULL,
                author_norm TEXT NOT NULL
            );

            CREATE TABLE paper_tags (
                paper_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                tag_norm TEXT NOT NULL
            );

            CREATE TABLE paper_collections (
                paper_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                collection_norm TEXT NOT NULL
            );

            CREATE INDEX idx_chunks_paper_id ON chunks(paper_id);
            CREATE INDEX idx_papers_year ON papers(year);
            CREATE INDEX idx_paper_authors_paper_id ON paper_authors(paper_id);
            CREATE INDEX idx_paper_tags_paper_id ON paper_tags(paper_id);
            CREATE INDEX idx_paper_collections_paper_id ON paper_collections(paper_id);
            """
        )

        seen_papers: set[str] = set()
        seen_authors: set[tuple[str, str]] = set()
        seen_tags: set[tuple[str, str]] = set()
        seen_collections: set[tuple[str, str]] = set()
        seen_chunks: set[str] = set()

        for split in splits:
            metadata = dict(getattr(split, "metadata", {}) or {})
            paper_id = str(metadata.get("paper_id") or "").strip()
            chunk_id = str(metadata.get("chunk_id") or "").strip()
            if not paper_id or not chunk_id:
                continue

            if paper_id not in seen_papers:
                cur.execute(
                    """
                    INSERT INTO papers (paper_id, title, year, venue)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        paper_id,
                        metadata.get("paper_title") or metadata.get("title"),
                        metadata.get("year"),
                        metadata.get("venue"),
                    ),
                )
                seen_papers.add(paper_id)

            if chunk_id not in seen_chunks:
                cur.execute(
                    """
                    INSERT INTO chunks (chunk_id, paper_id, faiss_row_id, docstore_id, section, page_start, page_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        paper_id,
                        (vectorstore_chunk_map.get(chunk_id) or {}).get("faiss_row_id"),
                        (vectorstore_chunk_map.get(chunk_id) or {}).get("docstore_id"),
                        metadata.get("section"),
                        metadata.get("page_start"),
                        metadata.get("page_end"),
                    ),
                )
                seen_chunks.add(chunk_id)

            for author in metadata.get("authors") or []:
                author_text = str(author or "").strip()
                author_norm = _normalize_text(author_text)
                key = (paper_id, author_norm)
                if author_norm and key not in seen_authors:
                    cur.execute(
                        """
                        INSERT INTO paper_authors (paper_id, author, author_norm)
                        VALUES (?, ?, ?)
                        """,
                        (paper_id, author_text, author_norm),
                    )
                    seen_authors.add(key)

            for tag in metadata.get("tags") or []:
                tag_text = str(tag or "").strip()
                tag_norm = _normalize_text(tag_text)
                key = (paper_id, tag_norm)
                if tag_norm and key not in seen_tags:
                    cur.execute(
                        """
                        INSERT INTO paper_tags (paper_id, tag, tag_norm)
                        VALUES (?, ?, ?)
                        """,
                        (paper_id, tag_text, tag_norm),
                    )
                    seen_tags.add(key)

            for collection in metadata.get("collections") or []:
                collection_text = str(collection or "").strip()
                collection_norm = _normalize_text(collection_text)
                key = (paper_id, collection_norm)
                if collection_norm and key not in seen_collections:
                    cur.execute(
                        """
                        INSERT INTO paper_collections (paper_id, collection, collection_norm)
                        VALUES (?, ?, ?)
                        """,
                        (paper_id, collection_text, collection_norm),
                    )
                    seen_collections.add(key)

        conn.commit()
    finally:
        conn.close()


class MetadataStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def get_active_chunk_ids(self) -> set[str]:
        """Return chunk ids currently present in metadata.db."""
        if not self.db_path.exists():
            return set()
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.cursor()
            cur.execute("SELECT chunk_id FROM chunks")
            return {str(row[0]) for row in cur.fetchall() if str(row[0] or "").strip()}
        finally:
            conn.close()

    def resolve_candidate_scope(self, filters: dict[str, Any]) -> dict[str, Any]:
        normalized_filters = filters if isinstance(filters, dict) else {}
        if not normalized_filters:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM papers")
                paper_count = int(cur.fetchone()[0])
                cur.execute("SELECT COUNT(*) FROM chunks")
                chunk_count = int(cur.fetchone()[0])
            finally:
                conn.close()
            return {
                "candidate_source": "full_index_default",
                "paper_ids": set(),
                "chunk_ids": set(),
                "faiss_row_ids": [],
                "docstore_ids": [],
                "paper_count": paper_count,
                "chunk_count": chunk_count,
            }

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            paper_sql = ["SELECT p.paper_id FROM papers p WHERE 1=1"]
            params: list[Any] = []

            year_range = normalized_filters.get("date_range") or {}
            year_from = year_range.get("from")
            year_to = year_range.get("to")
            if year_from is not None:
                paper_sql.append("AND p.year IS NOT NULL AND p.year >= ?")
                params.append(int(year_from))
            if year_to is not None:
                paper_sql.append("AND p.year IS NOT NULL AND p.year <= ?")
                params.append(int(year_to))

            field_specs = (
                ("authors", "paper_authors", "author_norm"),
                ("tags", "paper_tags", "tag_norm"),
                ("collections", "paper_collections", "collection_norm"),
            )
            for filter_key, table_name, column_name in field_specs:
                values = [item for item in normalized_filters.get(filter_key, []) if _normalize_text(item)]
                if not values:
                    continue
                like_clauses = [f"instr({column_name}, ?) > 0" for _ in values]
                paper_sql.append(
                    "AND EXISTS ("
                    f"SELECT 1 FROM {table_name} t "
                    "WHERE t.paper_id = p.paper_id AND ("
                    + " OR ".join(like_clauses)
                    + "))"
                )
                params.extend(_normalize_text(item) for item in values)

            cur.execute(" ".join(paper_sql), params)
            paper_ids = {str(row["paper_id"]) for row in cur.fetchall() if str(row["paper_id"]).strip()}
            if not paper_ids:
                return {
                    "candidate_source": "hard_filtered_empty",
                    "paper_ids": set(),
                    "chunk_ids": set(),
                    "faiss_row_ids": [],
                    "docstore_ids": [],
                    "paper_count": 0,
                    "chunk_count": 0,
                }

            placeholders = ",".join("?" for _ in paper_ids)
            cur.execute(
                f"SELECT chunk_id, faiss_row_id, docstore_id FROM chunks WHERE paper_id IN ({placeholders})",
                list(paper_ids),
            )
            chunk_ids: set[str] = set()
            faiss_row_ids: list[int] = []
            docstore_ids: list[str] = []
            for row in cur.fetchall():
                chunk_id = str(row["chunk_id"] or "").strip()
                if chunk_id:
                    chunk_ids.add(chunk_id)
                if row["faiss_row_id"] is not None:
                    faiss_row_ids.append(int(row["faiss_row_id"]))
                if row["docstore_id"] is not None and str(row["docstore_id"]).strip():
                    docstore_ids.append(str(row["docstore_id"]).strip())
            return {
                "candidate_source": "hard_filtered" if chunk_ids else "hard_filtered_empty",
                "paper_ids": paper_ids,
                "chunk_ids": chunk_ids,
                "faiss_row_ids": faiss_row_ids,
                "docstore_ids": docstore_ids,
                "paper_count": len(paper_ids),
                "chunk_count": len(chunk_ids),
            }
        finally:
            conn.close()
