"""Run dataset-based evaluation against the current ZoteroAgent baseline.

This script focuses on retrieval-oriented diagnostics so we can:
1. get a reproducible baseline over the current dataset
2. identify bottlenecks in the retrieval chain before tuning prompts/models
"""

from __future__ import annotations

import argparse
import shutil
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="ignore")

from src.config import AppConfig, parse_args, resolve_config
from src.indexer import build_embeddings, build_llm
from src.metadata_store import MetadataStore
from src.qa_agent import ZoteroAgent


def _default_runtime_config() -> Path:
    appdata = Path(os.getenv("APPDATA", str(Path.home())))
    return appdata / "AskMyZotero" / "config.yaml"


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _paper_signature(filename: str) -> str:
    text = str(filename or "")
    match = re.search(r"(\d{4}\s*-\s*.+)$", text)
    if match:
        text = match.group(1)
    return _normalize_text(text)


def _paper_tokens(filename: str) -> set[str]:
    normalized = _normalize_text(filename)
    return {token for token in normalized.split() if token and not re.fullmatch(r"\d{4}", token)}


def _same_paper(expected_name: str, candidate_name: str) -> bool:
    expected_sig = _paper_signature(expected_name)
    candidate_sig = _paper_signature(candidate_name)
    if not expected_sig or not candidate_sig:
        return False
    if expected_sig == candidate_sig:
        return True
    expected_tokens = _paper_tokens(expected_name)
    candidate_tokens = _paper_tokens(candidate_name)
    if not expected_tokens or not candidate_tokens:
        return False
    overlap = len(expected_tokens & candidate_tokens)
    min_size = min(len(expected_tokens), len(candidate_tokens))
    return overlap >= max(3, int(min_size * 0.6))


def _extract_filename(path_or_name: str) -> str:
    text = str(path_or_name or "").strip()
    if not text:
        return ""
    return Path(text).name or text


def _page_match(expected_pages: list[int], candidate_pages: list[int]) -> bool:
    expected = {int(page) for page in expected_pages if isinstance(page, int)}
    candidate = {int(page) for page in candidate_pages if isinstance(page, int)}
    if not expected or not candidate:
        return False
    return bool(expected & candidate)


def _rank_of_expected(items: list[dict[str, Any]], expected_pdf: str) -> int | None:
    for index, item in enumerate(items, start=1):
        title = item.get("title") or item.get("paper_title") or item.get("source_path") or item.get("source")
        if _same_paper(expected_pdf, _extract_filename(str(title or ""))):
            return index
    return None


def _collect_candidate_pages(items: list[dict[str, Any]], expected_pdf: str) -> list[int]:
    pages: list[int] = []
    for item in items:
        title = item.get("title") or item.get("paper_title") or item.get("source_path") or item.get("source")
        if not _same_paper(expected_pdf, _extract_filename(str(title or ""))):
            continue
        page = item.get("page")
        if isinstance(page, int):
            pages.append(page)
        for evidence in item.get("evidences", []):
            evidence_page = evidence.get("page")
            if isinstance(evidence_page, int):
                pages.append(evidence_page)
    return pages


def _classify_failure(sample: dict[str, Any], result: dict[str, Any], paper_rank: int | None, chunk_rank: int | None, page_hit: bool) -> str:
    predicted_intent = result.get("intent")
    answer_type = result.get("answer_type")
    uses_paper_rank = answer_type == "paper_list"
    if predicted_intent != sample["expected_intent"]:
        return "intent_mismatch"
    if uses_paper_rank and paper_rank is None and chunk_rank is None:
        return "target_not_recalled"
    if (not uses_paper_rank) and chunk_rank is None:
        return "target_not_recalled"
    if uses_paper_rank and paper_rank is not None and paper_rank > 3:
        return "target_ranked_too_low"
    if (not uses_paper_rank) and chunk_rank is not None and chunk_rank > 5:
        return "target_chunk_ranked_too_low"
    if uses_paper_rank and paper_rank is None and chunk_rank is not None:
        return "answer_or_aggregation_gap"
    if not page_hit and sample.get("source_pages"):
        return "wrong_evidence_page"
    return "answer_or_aggregation_gap"


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    intent_correct = sum(1 for item in records if item["intent_correct"])
    paper_hit_at_1 = sum(1 for item in records if item["paper_rank"] == 1)
    paper_hit_at_3 = sum(1 for item in records if item["paper_rank"] is not None and item["paper_rank"] <= 3)
    paper_hit_at_5 = sum(1 for item in records if item["paper_rank"] is not None and item["paper_rank"] <= 5)
    chunk_hit_at_5 = sum(1 for item in records if item["chunk_rank"] is not None and item["chunk_rank"] <= 5)
    target_hit_at_1 = sum(1 for item in records if item["target_rank"] == 1)
    target_hit_at_3 = sum(1 for item in records if item["target_rank"] is not None and item["target_rank"] <= 3)
    target_hit_at_5 = sum(1 for item in records if item["target_rank"] is not None and item["target_rank"] <= 5)
    page_hit = sum(1 for item in records if item["page_hit"])

    by_type: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[item["question_type"]].append(item)
    for question_type, items in grouped.items():
        by_type[question_type] = {
            "count": len(items),
            "intent_accuracy": round(sum(1 for item in items if item["intent_correct"]) / len(items), 4),
            "target_hit_at_3": round(sum(1 for item in items if item["target_rank"] is not None and item["target_rank"] <= 3) / len(items), 4),
            "paper_hit_at_1": round(sum(1 for item in items if item["paper_rank"] == 1) / len(items), 4),
            "paper_hit_at_3": round(sum(1 for item in items if item["paper_rank"] is not None and item["paper_rank"] <= 3) / len(items), 4),
            "page_hit_rate": round(sum(1 for item in items if item["page_hit"]) / len(items), 4),
        }

    failure_counts = Counter(item["failure_bucket"] for item in records if not item["success_case"])
    return {
        "sample_count": total,
        "intent_accuracy": round(intent_correct / total, 4) if total else 0.0,
        "paper_hit_at_1": round(paper_hit_at_1 / total, 4) if total else 0.0,
        "paper_hit_at_3": round(paper_hit_at_3 / total, 4) if total else 0.0,
        "paper_hit_at_5": round(paper_hit_at_5 / total, 4) if total else 0.0,
        "chunk_hit_at_5": round(chunk_hit_at_5 / total, 4) if total else 0.0,
        "target_hit_at_1": round(target_hit_at_1 / total, 4) if total else 0.0,
        "target_hit_at_3": round(target_hit_at_3 / total, 4) if total else 0.0,
        "target_hit_at_5": round(target_hit_at_5 / total, 4) if total else 0.0,
        "page_hit_rate": round(page_hit / total, 4) if total else 0.0,
        "by_question_type": by_type,
        "failure_buckets": dict(failure_counts),
    }


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _build_output_dir(user_value: str | None) -> Path:
    if user_value:
        return Path(user_value).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path(".askmyzotero") / "eval" / timestamp).resolve()


def _build_eval_agent(config: AppConfig) -> ZoteroAgent:
    embeddings = build_embeddings(config)
    vectorstore = FAISS.load_local(
        str(config.db_save_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    agent = ZoteroAgent.__new__(ZoteroAgent)
    agent.config = config
    agent.vectorstore = vectorstore
    agent.metadata_store = MetadataStore(config.metadata_db_path)
    agent.retriever = vectorstore.as_retriever(search_kwargs={"k": config.top_k})
    agent.llm = build_llm(config)
    agent._last_llm_debug = {}
    return agent


def _prepare_eval_metadata_db(config: AppConfig, output_dir: Path) -> Path:
    target_path = output_dir / "metadata_eval.db"
    shutil.copyfile(config.metadata_db_path, target_path)
    return target_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AskMyZotero baseline evaluation on a JSONL dataset.")
    parser.add_argument("--dataset", default="dataset/testpdf_qa_samples_40_refined.jsonl", help="Path to the evaluation dataset JSONL.")
    parser.add_argument("--output-dir", default=None, help="Directory to store summary and per-sample results.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick dry runs.")
    parser.add_argument("--top-k", type=int, default=None, help="Override top_k for evaluation.")
    args, remaining_argv = parser.parse_known_args()

    runtime_config = _default_runtime_config()
    if not os.getenv("ASKMYZOTERO_CONFIG") and runtime_config.exists():
        os.environ["ASKMYZOTERO_CONFIG"] = str(runtime_config)

    dataset_path = Path(args.dataset).expanduser().resolve()
    output_dir = _build_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0], *remaining_argv]
    try:
        config_args = parse_args()
    finally:
        sys.argv = original_argv
    config = resolve_config(config_args)
    if args.top_k is not None and args.top_k > 0:
        config.top_k = args.top_k
    config.metadata_db_path = _prepare_eval_metadata_db(config, output_dir)

    print(f"[eval] dataset={dataset_path}")
    print(f"[eval] output_dir={output_dir}")
    print(f"[eval] runtime_config={os.getenv('ASKMYZOTERO_CONFIG', '<repo config.yaml>')}")
    print(f"[eval] index_root={config.work_dir / config.index_name}")

    agent = _build_eval_agent(config)
    samples = _load_dataset(dataset_path)
    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[eval] {index}/{len(samples)} {sample['id']} :: {sample['question']}")
        result = agent.handle_query(sample["question"], top_k=config.top_k)

        papers = result.get("papers", [])
        chunks = result.get("snippets", [])
        paper_rank = _rank_of_expected(papers, sample["source_pdf"])
        chunk_rank = _rank_of_expected(chunks, sample["source_pdf"])
        candidate_pages = _collect_candidate_pages(papers, sample["source_pdf"])
        if not candidate_pages:
            candidate_pages = _collect_candidate_pages(chunks, sample["source_pdf"])
        page_hit = _page_match(sample.get("source_pages", []), candidate_pages)
        failure_bucket = _classify_failure(sample, result, paper_rank, chunk_rank, page_hit)
        target_rank = paper_rank if result.get("answer_type") == "paper_list" else chunk_rank
        success_case = (
            result.get("intent") == sample["expected_intent"]
            and target_rank is not None
            and target_rank <= (3 if result.get("answer_type") == "paper_list" else 5)
            and page_hit
        )

        row = {
            "id": sample["id"],
            "question": sample["question"],
            "question_type": sample["question_type"],
            "expected_intent": sample["expected_intent"],
            "predicted_intent": result.get("intent"),
            "intent_correct": result.get("intent") == sample["expected_intent"],
            "expected_pdf": sample["source_pdf"],
            "paper_rank": paper_rank,
            "chunk_rank": chunk_rank,
            "target_rank": target_rank,
            "expected_pages": sample.get("source_pages", []),
            "matched_pages": candidate_pages,
            "page_hit": page_hit,
            "answer_type": result.get("answer_type"),
            "status": result.get("status"),
            "success_case": success_case,
            "failure_bucket": "ok" if success_case else failure_bucket,
            "top_paper_titles": [paper.get("title") for paper in papers[:5]],
            "top_chunk_titles": [chunk.get("paper_title") or chunk.get("title") for chunk in chunks[:5]],
            "debug": result.get("debug", {}),
            "answer": result.get("answer", ""),
        }
        rows.append(row)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "config": {
            "work_dir": str(config.work_dir),
            "index_name": config.index_name,
            "zotero_path": str(config.zotero_path),
            "top_k": config.top_k,
            "chat_model": config.chat_model,
            "embedding_model": config.embedding_model,
        },
        "metrics": _summarize(rows),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "per_sample.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[eval] done")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
