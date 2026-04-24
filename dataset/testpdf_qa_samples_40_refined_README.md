# testpdf QA samples (40, refined)

This file accompanies `dataset/testpdf_qa_samples_40_refined.jsonl`.

Format:
- One JSON object per line.
- Every record includes:
  - `id`
  - `question`
  - `answer`
  - `source_pdf`
  - `question_type`
  - `expected_intent`
  - `source_pages`
  - `evidence`
  - `keywords`
  - `difficulty`
  - `notes`

Coverage summary:
- Sample count: `40`
- PDFs covered: `8 / 8`
- Question types:
  - `fact_qa`: 8
  - `comparison`: 8
  - `definition`: 8
  - `survey`: 8
  - `paper_lookup`: 8
- Per-PDF distribution:
  - `Chen 等 - 2025 - Joint Entity–Relation Extraction for Knowledge Graph Construction in Marine Ranching Equipment.pdf`: 5
  - `Cheng 等 - 2026 - Conditional Memory via Scalable Lookup A New Axis of Sparsity for Large Language Models.pdf`: 5
  - `Dong 等 - 2025 - Youtu-GraphRAG Vertically Unified Agents for Graph Retrieval-Augmented Complex Reasoning.pdf`: 5
  - `Gu 等 - 2021 - miTAR a hybrid deep learning-based approach for predicting miRNA targets.pdf`: 5
  - `Gutiérrez 等 - 2025 - From RAG to Memory Non-Parametric Continual Learning for Large Language Models.pdf`: 5
  - `Ma 等 - 2025 - Think-on-Graph 2.0 Deep and Faithful Large Language Model Reasoning with Knowledge-guided Retrieval.pdf`: 5
  - `Sun 等 - 2024 - Think-on-Graph Deep and Responsible Reasoning of Large Language Model on Knowledge Graph.pdf`: 5
  - `Wu 等 - 2025 - Think-on-Graph 3.0 Efficient and Adaptive LLM Reasoning on Heterogeneous Graphs via Multi-Agent Dua.pdf`: 5

Refinement goals used for this pass:
- Keep full coverage over all current PDFs under `testpdf/`.
- Balance all five supported `question_type` values exactly.
- Strengthen the set beyond title-and-abstract lookup by adding more method, experiment, ablation, and conclusion-grounded questions.
- Keep answers short, judgeable, and directly tied to evidence snippets already present in `dataset/testpdf_page_texts.json`.

Design notes:
- Each PDF contributes one sample for each supported type:
  - `paper_lookup`
  - `definition`
  - `fact_qa`
  - `comparison`
  - `survey`
- The set intentionally mixes:
  - contribution-identification questions
  - architecture or method-definition questions
  - numeric result questions
  - within-paper comparison questions
  - short system-overview or benchmark-overview questions

Evidence profile:
- Most `paper_lookup` items are abstract-anchored for reliable identification.
- The refined set adds many non-page-1 items from:
  - methods / architecture pages
  - experimental setup pages
  - result tables
  - ablation or analysis sections
  - conclusion pages
- In practice, well over half of the samples cite at least one non-page-1 source page.

Notes:
- Primary source used: `dataset/testpdf_page_texts.json`
- No PDF re-extraction was required for this first-pass refined set.
- Filenames in `source_pdf` are aligned to the current PDFs under `testpdf/`.
