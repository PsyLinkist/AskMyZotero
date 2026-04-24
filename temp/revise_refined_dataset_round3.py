import json
from pathlib import Path


path = Path("dataset/testpdf_qa_samples_40_refined.jsonl")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def by_id(target: str) -> dict:
    for row in rows:
        if row["id"] == target:
            return row
    raise KeyError(target)


r = by_id("testpdf-qa-006")
r["source_pages"] = [2, 3]
r["evidence"] = [
    {
        "page": 2,
        "snippet": "we advocate for a complementary axis of sparsity: conditional memory. Whereas conditional computation sparsely activates parameters to process dynamic logic, conditional memory relies on sparse lookup operations",
    },
    {
        "page": 3,
        "snippet": "Engram is a conditional memory module designed to augment the Transformer backbone by structurally separating static pattern storage from dynamic computation",
    },
]

r = by_id("testpdf-qa-010")
r["question"] = "What are the two main functional stages of Engram, and how are the retrieved embeddings fused back into the backbone?"
r["answer"] = "Engram has two main stages: retrieval and fusion. It first retrieves static embeddings from compressed suffix n-grams via hashing, then fuses them with hidden states using context-aware gating, a short depthwise causal convolution, and a residual connection."
r["source_pages"] = [3, 4]
r["evidence"] = [
    {
        "page": 3,
        "snippet": "the module processes each position in two functional phases: retrieval and fusion. First, we extract and compress suffix n-grams to deterministically retrieve static embedding vectors via hashing",
    },
    {
        "page": 4,
        "snippet": "we employ a context-aware gating mechanism to adaptively filter the memory priors and introduce a short, depthwise causal convolution before adding the result back through a residual connection",
    },
]
r["difficulty"] = "hard"
r["notes"] = "Narrowed to a single architecture-mechanism target and upgraded to hard."

r = by_id("testpdf-qa-011")
r["source_pages"] = [14]
r["evidence"] = [
    {
        "page": 14,
        "snippet": "Anonymity Reversion, a novel task to mitigate knowledge leakage in LLMs, deeply measuring the real performance of GraphRAG frameworks supported by a carefully curated anonymous dataset",
    }
]

r = by_id("testpdf-qa-018")
r["evidence"] = [
    {
        "page": 8,
        "snippet": "We obtained an average accuracy of 97.9% on the DeepMirTar dataset",
    },
    {
        "page": 9,
        "snippet": "Our models achieved higher accuracies than the DeepMirTar (97.9% vs 93.5%) and miRAW (96.5% vs 93.5%) studies",
    },
]

r = by_id("testpdf-qa-020")
r["question"] = "What architectural strengths does the paper attribute to miTAR?"
r["answer"] = "The paper attributes miTAR's architecture strength to its CNN-plus-RNN hybrid design, which learns spatial and sequential features directly from raw sequences instead of relying on hand-crafted features."
r["source_pages"] = [1, 14]
r["evidence"] = [
    {
        "page": 1,
        "snippet": "This approach integrates convolutional neural networks (CNNs) that excel in learning spatial features and recurrent neural networks (RNNs) that discern sequential features",
    },
    {
        "page": 14,
        "snippet": "The miTAR model is the first to use CNN to capture the potential spatial features directly from the sequences of miRNAs and genes",
    },
]
r["notes"] = "Reduced scope from architecture plus empirical strengths to architecture only."

r = by_id("testpdf-qa-021")
r["source_pages"] = [3, 4]
r["evidence"] = [
    {
        "page": 3,
        "snippet": "Although both GraphRAG and LightRAG use a KG just like our HippoRAG 2 approach, our KG is used to aid in the retrieval process rather than to expand the retrieval corpus itself",
    },
    {
        "page": 4,
        "snippet": "For offline indexing, we use an LLM to extract open KG triples from passages, with synonym detection applied to phrase nodes. Together, these phrases and passages form the open KG",
    },
]

r = by_id("testpdf-qa-022")
r["evidence"] = [
    {
        "page": 1,
        "snippet": "We propose HippoRAG 2, a framework that outperforms standard RAG comprehensively on factual, sense-making, and associative memory tasks",
    },
    {
        "page": 4,
        "snippet": "For offline indexing, we use an LLM to extract open KG triples from passages, with synonym detection applied to phrase nodes. Together, these phrases and passages form the open KG",
    },
]

r = by_id("testpdf-qa-023")
r["difficulty"] = "hard"

r = by_id("testpdf-qa-025")
r["question"] = "What retrieval workflow does the HippoRAG 2 paper describe?"
r["answer"] = "HippoRAG 2 first extracts open-KG triples offline, then during online retrieval it scores passages and triples to choose seed nodes, filters triples with the LLM, runs Personalized PageRank over the graph, and finally performs QA reading on the selected passages."
r["source_pages"] = [4]
r["evidence"] = [
    {
        "page": 4,
        "snippet": "For offline indexing, we use an LLM to extract open KG triples from passages, with synonym detection applied to phrase nodes. Together, these phrases and passages form the open KG. For online retrieval, an embedding model scores both the passages and triples to identify the seed nodes. The PPR algorithm then performs context-based retrieval on the KG",
    }
]
r["difficulty"] = "hard"
r["notes"] = "Reduced scope from evaluation categories plus workflow to workflow only."

r = by_id("testpdf-qa-026")
r["source_pages"] = [2, 3]
r["evidence"] = [
    {
        "page": 2,
        "snippet": "we propose Think-on-Graph 2.0 (ToG-2), a tight-coupling hybrid (KG Text) RAG paradigm which effectively integrates unstructured knowledge from texts with structured insights from KGs",
    },
    {
        "page": 3,
        "snippet": "ToG-2 achieves in-depth and reliable context retrieval through the guide of KGs and performs precise graph retrieval by treating documents as node contexts",
    },
]

r = by_id("testpdf-qa-029")
r["difficulty"] = "hard"

r = by_id("testpdf-qa-030")
r["evidence"] = [
    {
        "page": 4,
        "snippet": "ToG-2 begins by extracting entities from the given question as initial topic entities. It then performs an iterative process of graph retrieval, context retrieval and LLM reasoning",
    },
    {
        "page": 10,
        "snippet": "the combination of triple-link reasoning and entity context documents is a highly effective pattern",
    },
]

r = by_id("testpdf-qa-031")
r["source_pages"] = [2]
r["evidence"] = [
    {
        "page": 2,
        "snippet": "Using the beam search algorithm in KG and LLM reasoning, ToG allows the LLM to dynamically explore a number of reasoning paths and keep the most promising ones",
    }
]

r = by_id("testpdf-qa-035")
r["difficulty"] = "hard"

r = by_id("testpdf-qa-036")
r["answer"] = "Wu et al. (2025), Think-on-Graph 3.0: Efficient and Adaptive LLM Reasoning on Heterogeneous Graphs via Multi-Agent Dual-Evolving Context Retrieval, introduces MACER."
r["source_pages"] = [2, 3]
r["evidence"] = [
    {
        "page": 2,
        "snippet": "a novel MACER (Multi-Agent Context Evolution and Retrieval) mechanism, which pioneeringly incorporates a dual-evolution mechanism of Evolving Query and Evolving Sub-Graph for precise evidence retrieval",
    },
    {
        "page": 3,
        "snippet": "We propose MACER, a novel multi-agent framework that introduces a dual-evolution mechanism integrating Evolving Query and Evolving Sub-Graph within graph-based RAG",
    },
]

r = by_id("testpdf-qa-037")
r["evidence"] = [
    {
        "page": 1,
        "snippet": "This paper presents Think-on-Graph 3.0 (ToG-3), a novel framework that introduces Multi-Agent Context Evolution and Retrieval (MACER) mechanism to overcome these limitations",
    },
    {
        "page": 6,
        "snippet": "The constructor agent applies the transition operator over evolving queries and evolving sub-graphs during the MACER process",
    },
]

r = by_id("testpdf-qa-039")
r["difficulty"] = "hard"
r["evidence"] = [
    {
        "page": 7,
        "snippet": "Ours Average EM 0.453 F1 0.312; HippoRAG-2 Average EM 0.438 F1 0.311; GraphRAG Average EM 0.295 F1 0.012; LightRAG Average EM 0.270 F1 0.015",
    },
    {
        "page": 8,
        "snippet": "our proposed method consistently outperforms all competitors, achieving the highest average EM (0.453) and F1 (0.312) scores across all three benchmarks",
    },
]

r = by_id("testpdf-qa-040")
r["difficulty"] = "hard"
r["evidence"] = [
    {
        "page": 1,
        "snippet": "Our core innovation is the dynamic construction and refinement of a Chunk-Triplets-Community heterogeneous graph index",
    },
    {
        "page": 6,
        "snippet": "The complete MACER process continues until the Reflector outputs the STOP action, and the final answer is synthesized from the full trajectory",
    },
    {
        "page": 14,
        "snippet": "Constructor Agent extracts and links graph elements, Retriever Agent retrieves the most relevant nodes, and Reflector or Responser Agent uses the retrieved passages as context for answer generation",
    },
]

path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

readme_path = Path("dataset/testpdf_qa_samples_40_refined_README.md")
readme = readme_path.read_text(encoding="utf-8")
readme = readme.replace("Added more method, experiment, ablation, and conclusion pages instead of relying only on page 1.", "Added more method, experiment, ablation, and conclusion pages instead of relying only on page 1.\n- Introduced an explicit hard tier after review round 2.\n- Tightened several survey and paper_lookup items to improve auditability and reduce double-barreled prompts.")
readme_path.write_text(readme, encoding="utf-8")

print("revised", path)
