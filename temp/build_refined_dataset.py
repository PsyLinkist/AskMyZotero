import json
from collections import Counter
from pathlib import Path


def pick(names: list[str], substr: str) -> str:
    for name in names:
        if substr in name:
            return name
    raise KeyError(substr)


names = [p.name for p in Path("testpdf").glob("*.pdf")]
pdf = {
    "chen": pick(names, "Marine Ranching Equipment"),
    "cheng": pick(names, "Conditional Memory via Scalable Lookup"),
    "dong": pick(names, "Youtu-GraphRAG Vertically Unified Agents"),
    "gu": pick(names, "miTAR a hybrid deep learning-based approach"),
    "gutierrez": pick(names, "From RAG to Memory"),
    "ma": pick(names, "Think-on-Graph 2.0"),
    "sun": pick(names, "Think-on-Graph Deep and Responsible"),
    "wu": pick(names, "Think-on-Graph 3.0"),
}

records = [
    {
        "id": "chen_definition_01",
        "question": "What knowledge graph framework does the marine ranching equipment paper propose?",
        "answer": "It proposes the first knowledge graph framework tailored for marine ranching equipment, combining hybrid ontology design, joint entity-relation extraction, and graph-based knowledge storage.",
        "source_pdf": pdf["chen"],
        "question_type": "definition",
        "expected_intent": "definition",
        "source_pages": [1],
        "evidence": [
            "This study proposes the first knowledge graph framework tailored for marine ranching equipment, integrating hybrid ontology design, joint entity-relation extraction, and graph-based knowledge storage."
        ],
        "keywords": ["knowledge graph", "marine ranching equipment", "hybrid ontology"],
        "difficulty": "easy",
        "notes": "Definition grounded in the abstract rather than the title alone.",
    },
    {
        "id": "chen_fact_02",
        "question": "How many key concepts and semantic relationships are defined in the ontology for the marine ranching equipment KG?",
        "answer": "The ontology defines seven key concepts and eight semantic relationships.",
        "source_pdf": pdf["chen"],
        "question_type": "fact_qa",
        "expected_intent": "fact_qa",
        "source_pages": [1],
        "evidence": [
            "A domain ontology was constructed through a combination of the top-down and the bottom-up approach, defining seven key concepts and eight semantic relationships."
        ],
        "keywords": ["ontology", "seven concepts", "eight relationships"],
        "difficulty": "easy",
        "notes": "Precise factual sample with a single unambiguous answer.",
    },
    {
        "id": "chen_comparison_03",
        "question": "Compared with BiLSTM-CRF and BERT-BiLSTM-CRF, why is the proposed model stronger for this extraction task?",
        "answer": "Because it combines BERT contextual embeddings, BiGRU sequence modeling, and CRF label-dependency optimization, and the paper reports that it outperforms BiLSTM-CRF and BERT-BiLSTM-CRF on precision, recall, and F1.",
        "source_pdf": pdf["chen"],
        "question_type": "comparison",
        "expected_intent": "comparison",
        "source_pages": [1, 16],
        "evidence": [
            "A novel BERT-BiGRU-CRF model was developed, leveraging contextual embeddings from BERT, parameter-efficient sequence modeling via BiGRU, and label dependency optimization using CRF.",
            "Experimental results demonstrated superior performance over BiLSTM-CRF and BERT-BiLSTM-CRF, achieving 86.58% precision, 77.82% recall, and 81.97% F1 score.",
        ],
        "keywords": ["BERT-BiGRU-CRF", "BiLSTM-CRF", "comparison"],
        "difficulty": "medium",
        "notes": "Model-comparison item with method and result evidence.",
    },
    {
        "id": "chen_survey_04",
        "question": "Summarize the main pipeline used to build the marine ranching equipment knowledge graph.",
        "answer": "The pipeline starts from user questionnaires and ontology design, collects and cleans semi-structured and unstructured data into a domain corpus, applies joint entity-relation extraction with BERT-BiGRU-CRF and a TE+SE+Ri+BMESO tagging strategy, and stores the resulting graph in Neo4j for visualization and updates.",
        "source_pdf": pdf["chen"],
        "question_type": "survey",
        "expected_intent": "survey",
        "source_pages": [1, 2],
        "evidence": [
            "The limitations in existing KG are obtained through targeted questionnaires for diverse users and employees.",
            "Semi-structured data from enterprises and standards, combined with unstructured data from the literature were systematically collected, cleaned via Scrapy and regular expression, and standardized into JSON format.",
            "A novel BERT-BiGRU-CRF model was developed...",
            "The Neo4j-based KG encapsulated 2153 nodes and 3872 edges, enabling scalable visualization and dynamic updates.",
        ],
        "keywords": ["pipeline", "Neo4j", "TE+SE+Ri+BMESO"],
        "difficulty": "hard",
        "notes": "Multi-step summary sample intended to probe aggregation.",
    },
    {
        "id": "chen_paper_lookup_05",
        "question": "Which paper introduces the TE+SE+Ri+BMESO tagging strategy for multi-relation extraction in a marine ranching equipment KG?",
        "answer": "Chen et al. (2025), Joint Entity-Relation Extraction for Knowledge Graph Construction in Marine Ranching Equipment.",
        "source_pdf": pdf["chen"],
        "question_type": "paper_lookup",
        "expected_intent": "paper_lookup",
        "source_pages": [1],
        "evidence": [
            "The TE + SE + Ri + BMESO tagging strategy was introduced to address multi-relation extraction challenges by linking theme entities to secondary entities."
        ],
        "keywords": ["paper lookup", "TE+SE+Ri+BMESO", "marine ranching"],
        "difficulty": "easy",
        "notes": "Paper identification based on a method detail rather than title keywords only.",
    },
    {
        "id": "cheng_definition_06",
        "question": "What is Engram in the conditional memory paper?",
        "answer": "Engram is a conditional memory module that modernizes classic n-gram embedding into an O(1) lookup primitive for large language models.",
        "source_pdf": pdf["cheng"],
        "question_type": "definition",
        "expected_intent": "definition",
        "source_pages": [1],
        "evidence": [
            "We introduce conditional memory as a complementary sparsity axis, instantiated via Engram, a module that modernizes classic n-gram embedding for O(1) lookup."
        ],
        "keywords": ["Engram", "conditional memory", "O(1) lookup"],
        "difficulty": "easy",
        "notes": "Direct concept-definition sample.",
    },
    {
        "id": "cheng_fact_07",
        "question": "To what parameter scale is Engram expanded in the paper?",
        "answer": "The paper scales Engram to 27B parameters.",
        "source_pdf": pdf["cheng"],
        "question_type": "fact_qa",
        "expected_intent": "fact_qa",
        "source_pages": [1],
        "evidence": [
            "Guided by this law, we scale Engram to 27B parameters, achieving superior performance over a strictly iso-parameter and iso-FLOPs MoE baseline."
        ],
        "keywords": ["27B", "scaling", "Engram"],
        "difficulty": "easy",
        "notes": "Simple fact from the abstract.",
    },
    {
        "id": "cheng_comparison_08",
        "question": "Compared with Mixture-of-Experts, what new sparsity axis does the paper argue for?",
        "answer": "It argues for conditional memory as a complementary sparsity axis, contrasting static memory lookup with MoE-style conditional computation.",
        "source_pdf": pdf["cheng"],
        "question_type": "comparison",
        "expected_intent": "comparison",
        "source_pages": [1],
        "evidence": [
            "While Mixture-of-Experts (MoE) scales capacity via conditional computation, Transformers lack a native primitive for knowledge lookup.",
            "We introduce conditional memory as a complementary sparsity axis.",
        ],
        "keywords": ["MoE", "conditional computation", "conditional memory"],
        "difficulty": "medium",
        "notes": "Tests the comparison claim in the abstract.",
    },
    {
        "id": "cheng_survey_09",
        "question": "Summarize the main benefits claimed for Engram in the paper.",
        "answer": "The paper claims Engram improves knowledge retrieval, reasoning, code and math performance, and long-context retrieval, while also enabling efficient deterministic prefetching from host memory with negligible overhead.",
        "source_pdf": pdf["cheng"],
        "question_type": "survey",
        "expected_intent": "survey",
        "source_pages": [1],
        "evidence": [
            "We observe even larger gains in general reasoning (e.g., BBH +5.0; ARC-Challenge +3.7) and code/math domains.",
            "It frees up attention capacity for global context, substantially boosting long-context retrieval.",
            "Its deterministic addressing enables runtime prefetching from host memory, incurring negligible overhead.",
        ],
        "keywords": ["reasoning", "long-context retrieval", "prefetching"],
        "difficulty": "medium",
        "notes": "Broad summary question spanning multiple claims.",
    },
    {
        "id": "cheng_paper_lookup_10",
        "question": "Which paper proposes conditional memory via scalable lookup as a new axis of sparsity for large language models?",
        "answer": "Cheng et al. (2026), Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models.",
        "source_pdf": pdf["cheng"],
        "question_type": "paper_lookup",
        "expected_intent": "paper_lookup",
        "source_pages": [1],
        "evidence": [
            "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models",
            "We envision conditional memory as an indispensable modeling primitive for next-generation sparse models.",
        ],
        "keywords": ["paper lookup", "conditional memory", "Engram"],
        "difficulty": "easy",
        "notes": "Paper lookup grounded by the abstract and title phrase.",
    },
    {
        "id": "dong_definition_11",
        "question": "What is Youtu-GraphRAG according to the paper?",
        "answer": "Youtu-GraphRAG is a vertically unified agentic GraphRAG paradigm that jointly integrates graph construction and retrieval through a shared graph schema.",
        "source_pdf": pdf["dong"],
        "question_type": "definition",
        "expected_intent": "definition",
        "source_pages": [1, 2],
        "evidence": [
            "We propose a vertically unified agentic paradigm, Youtu-GraphRAG, to jointly connect the entire framework as an intricate integration.",
            "Youtu-GraphRAG proposes a unified paradigm for superior complex reasoning.",
        ],
        "keywords": ["Youtu-GraphRAG", "unified paradigm", "graph schema"],
        "difficulty": "easy",
        "notes": "Definition sample based on system-level framing.",
    },
    {
        "id": "dong_fact_12",
        "question": "What token-cost saving and accuracy gain does Youtu-GraphRAG report over prior baselines?",
        "answer": "It reports up to 90.71% token-cost savings and 16.62% higher accuracy over state-of-the-art baselines.",
        "source_pdf": pdf["dong"],
        "question_type": "fact_qa",
        "expected_intent": "fact_qa",
        "source_pages": [1, 2],
        "evidence": [
            "Extensive experiments across six challenging benchmarks demonstrate the robustness of Youtu-GraphRAG, remarkably moving the Pareto frontier with up to 90.71% saving of token costs and 16.62% higher accuracy over state-of-the-art baselines."
        ],
        "keywords": ["90.71%", "16.62%", "Pareto frontier"],
        "difficulty": "easy",
        "notes": "Result-oriented fact sample.",
    },
    {
        "id": "dong_comparison_13",
        "question": "How does Youtu-GraphRAG differ from earlier GraphRAG methods that optimize only one stage?",
        "answer": "Earlier methods typically optimize graph construction or graph retrieval in isolation, while Youtu-GraphRAG unifies both through the same schema-guided agentic framework and coordinated reasoning process.",
        "source_pdf": pdf["dong"],
        "question_type": "comparison",
        "expected_intent": "comparison",
        "source_pages": [2, 3],
        "evidence": [
            "Current methods focus on either graph construction or retrieval in isolation, while Youtu-GraphRAG proposes a unified paradigm.",
            "A toy overview of Youtu-GraphRAG that unifies graph construction and retrieval through a schema-guided agentic paradigm.",
        ],
        "keywords": ["comparison", "construction", "retrieval", "schema-guided"],
        "difficulty": "medium",
        "notes": "Non-title comparison grounded in intro and figure description.",
    },
    {
        "id": "dong_survey_14",
        "question": "Summarize the main components of Youtu-GraphRAG.",
        "answer": "Its main components are a seed graph schema with adaptive expansion, a dually-perceived community detection method that builds a hierarchical knowledge tree, an agentic retriever that decomposes queries into parallel sub-queries with reflection, and an anonymity-based evaluation setting for fairer GraphRAG measurement.",
        "source_pdf": pdf["dong"],
        "question_type": "survey",
        "expected_intent": "survey",
        "source_pages": [1, 2, 3],
        "evidence": [
            "A seed graph schema is introduced...",
            "We develop novel dually-perceived community detection...",
            "An agentic retriever is designed to interpret the same graph schema...",
            "We propose a tailored anonymous dataset and a novel Anonymity Reversion task...",
        ],
        "keywords": ["seed schema", "knowledge tree", "agentic retriever", "Anonymity Reversion"],
        "difficulty": "hard",
        "notes": "Aggregates contributions across the abstract and overview pages.",
    },
    {
        "id": "dong_paper_lookup_15",
        "question": "Which paper introduces the Anonymity Reversion task for evaluating GraphRAG more fairly?",
        "answer": "Dong et al. (2025), Youtu-GraphRAG: Vertically Unified Agents for Graph Retrieval-Augmented Complex Reasoning.",
        "source_pdf": pdf["dong"],
        "question_type": "paper_lookup",
        "expected_intent": "paper_lookup",
        "source_pages": [1, 2],
        "evidence": [
            "We propose a tailored anonymous dataset and a novel Anonymity Reversion task that deeply measures the real performance of the GraphRAG frameworks."
        ],
        "keywords": ["paper lookup", "Anonymity Reversion", "GraphRAG evaluation"],
        "difficulty": "easy",
        "notes": "Paper lookup keyed off a specific evaluation idea.",
    },
    {
        "id": "gu_definition_16",
        "question": "What is miTAR in the 2021 paper?",
        "answer": "miTAR is a hybrid deep learning approach for miRNA target prediction that combines CNNs and RNNs to learn both spatial and sequential features.",
        "source_pdf": pdf["gu"],
        "question_type": "definition",
        "expected_intent": "definition",
        "source_pages": [1],
        "evidence": [
            "We developed a novel hybrid deep learning-based approach that integrates convolutional neural networks and recurrent neural networks.",
            "We present a new DL-based approach for predicting miRNA targets and demonstrate that our approach outperforms the current alternatives.",
        ],
        "keywords": ["miTAR", "miRNA target prediction", "CNN", "RNN"],
        "difficulty": "easy",
        "notes": "Method definition based on the abstract.",
    },
    {
        "id": "gu_fact_17",
        "question": "Which architectural layer does the paper say improves the performance of all its hybrid models?",
        "answer": "The paper says that adding a max pooling layer between the CNN and RNN improves the performance of all the models.",
        "source_pdf": pdf["gu"],
        "question_type": "fact_qa",
        "expected_intent": "fact_qa",
        "source_pages": [1],
        "evidence": [
            "We examined the contribution of a Max pooling layer in between the CNN and RNN and demonstrated that it improves the performance of all our models."
        ],
        "keywords": ["max pooling", "CNN", "RNN"],
        "difficulty": "easy",
        "notes": "Specific architecture fact rather than generic paper title recall.",
    },
    {
        "id": "gu_comparison_18",
        "question": "Compared with methods that rely on predefined features, what advantage does miTAR claim?",
        "answer": "miTAR avoids hand-crafted predefined features and instead learns intrinsic spatial and sequential features directly from raw miRNA and gene sequences.",
        "source_pdf": pdf["gu"],
        "question_type": "comparison",
        "expected_intent": "comparison",
        "source_pages": [1],
        "evidence": [
            "The majority of these methods depend on pre-defined features that require considerable efforts and resources to compute and often prove suboptimal.",
            "The inputs for our approach are raw sequences of miRNAs and genes that can be obtained effortlessly.",
        ],
        "keywords": ["predefined features", "raw sequences", "comparison"],
        "difficulty": "medium",
        "notes": "Compares representational assumptions rather than only metrics.",
    },
    {
        "id": "gu_survey_19",
        "question": "Summarize the main claims made by the miTAR paper.",
        "answer": "The paper claims that its hybrid CNN-RNN approach improves miRNA target prediction accuracy, remains robust on small datasets, benefits from max pooling, and can be unified into a model that generalizes across datasets.",
        "source_pdf": pdf["gu"],
        "question_type": "survey",
        "expected_intent": "survey",
        "source_pages": [1],
        "evidence": [
            "The two models consistently outperform the previous methods according to evaluation metrics on test datasets.",
            "Our approach is more robust than other methods on small datasets.",
            "A unified model was developed that is robust on fitting different input datasets.",
        ],
        "keywords": ["robustness", "small datasets", "unified model"],
        "difficulty": "medium",
        "notes": "High-level synthesis of findings from the abstract.",
    },
    {
        "id": "gu_paper_lookup_20",
        "question": "Which paper introduces miTAR as a hybrid deep learning-based approach for predicting miRNA targets?",
        "answer": "Gu et al. (2021), miTAR: a hybrid deep learning-based approach for predicting miRNA targets.",
        "source_pdf": pdf["gu"],
        "question_type": "paper_lookup",
        "expected_intent": "paper_lookup",
        "source_pages": [1],
        "evidence": [
            "miTAR: a hybrid deep learning-based approach for predicting miRNA targets"
        ],
        "keywords": ["paper lookup", "miTAR", "miRNA"],
        "difficulty": "easy",
        "notes": "Canonical paper lookup sample.",
    },
    {
        "id": "gutierrez_definition_21",
        "question": "What is HippoRAG 2 according to the paper?",
        "answer": "HippoRAG 2 is a framework for non-parametric continual learning in large language models that aims to improve factual, sense-making, and associative memory over standard RAG and earlier structured variants.",
        "source_pdf": pdf["gutierrez"],
        "question_type": "definition",
        "expected_intent": "definition",
        "source_pages": [1],
        "evidence": [
            "We propose HippoRAG 2, a framework that outperforms standard RAG comprehensively on factual, sense-making, and associative memory tasks."
        ],
        "keywords": ["HippoRAG 2", "non-parametric continual learning", "memory"],
        "difficulty": "easy",
        "notes": "Definition sample grounded in the abstract.",
    },
    {
        "id": "gutierrez_fact_22",
        "question": "What core algorithm does HippoRAG 2 build on, and what memory improvement does it report?",
        "answer": "It builds on Personalized PageRank and reports a 7% improvement in associative memory tasks over the state-of-the-art embedding model.",
        "source_pdf": pdf["gutierrez"],
        "question_type": "fact_qa",
        "expected_intent": "fact_qa",
        "source_pages": [1],
        "evidence": [
            "HippoRAG 2 builds upon the Personalized PageRank algorithm used in HippoRAG...",
            "This combination pushes this RAG system closer to the effectiveness of human long-term memory, achieving a 7% improvement in associative memory tasks over the state-of-the-art embedding model.",
        ],
        "keywords": ["Personalized PageRank", "7% improvement", "associative memory"],
        "difficulty": "medium",
        "notes": "Joint factual sample combining method and result.",
    },
    {
        "id": "gutierrez_comparison_23",
        "question": "Compared with standard vector-retrieval RAG and recent structured variants, what gap is HippoRAG 2 trying to close?",
        "answer": "It tries to avoid the trade-off where standard RAG is simpler and robust but limited as long-term memory, while structured variants improve sense-making and associativity yet can fall below standard RAG on basic factual memory.",
        "source_pdf": pdf["gutierrez"],
        "question_type": "comparison",
        "expected_intent": "comparison",
        "source_pages": [1],
        "evidence": [
            "Its reliance on vector retrieval hinders its ability to mimic the dynamic and interconnected nature of human long-term memory.",
            "Recent RAG approaches augment vector embeddings with various structures... However, their performance on more basic factual memory tasks drops considerably below standard RAG.",
        ],
        "keywords": ["standard RAG", "structured RAG", "factual memory"],
        "difficulty": "hard",
        "notes": "Comparison sample about system trade-offs instead of just metrics.",
    },
    {
        "id": "gutierrez_survey_24",
        "question": "Summarize how HippoRAG 2 improves over earlier memory-oriented RAG approaches.",
        "answer": "HippoRAG 2 keeps the Personalized PageRank foundation of HippoRAG but adds deeper passage integration and more effective online LLM use so it can improve factual, sense-making, and associative memory together rather than trading one off against another.",
        "source_pdf": pdf["gutierrez"],
        "question_type": "survey",
        "expected_intent": "survey",
        "source_pages": [1],
        "evidence": [
            "HippoRAG 2 builds upon the Personalized PageRank algorithm used in HippoRAG and enhances it with deeper passage integration and more effective online use of an LLM.",
            "We address this unintended deterioration and propose HippoRAG 2...",
        ],
        "keywords": ["deeper passage integration", "online LLM use", "memory tasks"],
        "difficulty": "medium",
        "notes": "High-level summary of method evolution.",
    },
    {
        "id": "gutierrez_paper_lookup_25",
        "question": "Which paper explicitly frames its contribution as moving from RAG to memory for non-parametric continual learning?",
        "answer": "Gutiérrez et al. (2025), From RAG to Memory: Non-Parametric Continual Learning for Large Language Models.",
        "source_pdf": pdf["gutierrez"],
        "question_type": "paper_lookup",
        "expected_intent": "paper_lookup",
        "source_pages": [1],
        "evidence": [
            "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models"
        ],
        "keywords": ["paper lookup", "RAG to Memory", "continual learning"],
        "difficulty": "easy",
        "notes": "Paper-lookup sample keyed to the paper's framing phrase.",
    },
    {
        "id": "ma_definition_26",
        "question": "What is Think-on-Graph 2.0 (ToG-2)?",
        "answer": "ToG-2 is a hybrid RAG framework that tightly couples graph retrieval and context retrieval so LLMs can perform deeper and more faithful reasoning.",
        "source_pdf": pdf["ma"],
        "question_type": "definition",
        "expected_intent": "definition",
        "source_pages": [1],
        "evidence": [
            "We introduce Think-on-Graph 2.0 (ToG-2), a hybrid RAG framework that iteratively retrieves information from both unstructured and structured knowledge sources in a tight-coupling manner."
        ],
        "keywords": ["ToG-2", "hybrid RAG", "tight coupling"],
        "difficulty": "easy",
        "notes": "Definition sample grounded in the abstract.",
    },
    {
        "id": "ma_fact_27",
        "question": "What performance claim does the ToG-2 paper make about GPT-3.5 and smaller models?",
        "answer": "It claims ToG-2 achieves overall SOTA on 6 of 7 knowledge-intensive datasets with GPT-3.5 and can raise smaller models such as LLaMA-2-13B to the level of GPT-3.5 direct reasoning.",
        "source_pdf": pdf["ma"],
        "question_type": "fact_qa",
        "expected_intent": "fact_qa",
        "source_pages": [1],
        "evidence": [
            "Extensive experiments demonstrate that ToG-2 achieves overall state-of-the-art performance on 6 out of 7 knowledge-intensive datasets with GPT-3.5, and can elevate the performance of smaller models (e.g., LLAMA-2-13B) to the level of GPT-3.5's direct reasoning."
        ],
        "keywords": ["6 of 7", "GPT-3.5", "LLaMA-2-13B"],
        "difficulty": "medium",
        "notes": "Result-focused sample.",
    },
    {
        "id": "ma_comparison_28",
        "question": "Compared with text-based RAG, what problem is ToG-2 designed to solve?",
        "answer": "It is designed to overcome the shallow and incomplete retrieval of text-based RAG by using knowledge graphs to guide deep context retrieval and documents to support more reliable graph retrieval.",
        "source_pdf": pdf["ma"],
        "question_type": "comparison",
        "expected_intent": "comparison",
        "source_pages": [1],
        "evidence": [
            "Current RAG methods often fall short of ensuring the depth and completeness of retrieved information.",
            "ToG-2 leverages knowledge graphs to link documents via entities, facilitating deep and knowledge-guided context retrieval. Simultaneously, it utilizes documents as entity contexts to achieve precise and efficient graph retrieval.",
        ],
        "keywords": ["text-based RAG", "depth", "completeness", "knowledge-guided"],
        "difficulty": "medium",
        "notes": "Ability-comparison sample grounded in problem statement and method.",
    },
    {
        "id": "ma_survey_29",
        "question": "Summarize how ToG-2 performs retrieval during reasoning.",
        "answer": "ToG-2 alternates between graph retrieval and context retrieval: the KG links documents through entities for deeper clue discovery, while documents serve as entity contexts to make graph retrieval more precise and efficient, and this iterative collaboration supports final answer generation.",
        "source_pdf": pdf["ma"],
        "question_type": "survey",
        "expected_intent": "survey",
        "source_pages": [1],
        "evidence": [
            "ToG-2 alternates between graph retrieval and context retrieval to search for in-depth clues relevant to the question, enabling LLMs to generate answers."
        ],
        "keywords": ["graph retrieval", "context retrieval", "iterative"],
        "difficulty": "hard",
        "notes": "Summary sample intended to probe retrieval-process understanding.",
    },
    {
        "id": "ma_paper_lookup_30",
        "question": "Which paper introduces ToG-2 as a deep and faithful large language model reasoning framework with knowledge-guided retrieval?",
        "answer": "Ma et al. (2025), Think-on-Graph 2.0: Deep and Faithful Large Language Model Reasoning with Knowledge-guided Retrieval Augmented Generation.",
        "source_pdf": pdf["ma"],
        "question_type": "paper_lookup",
        "expected_intent": "paper_lookup",
        "source_pages": [1],
        "evidence": [
            "THINK-ON-GRAPH 2.0: DEEP AND FAITHFUL LARGE LANGUAGE MODEL REASONING WITH KNOWLEDGE-GUIDED RETRIEVAL AUGMENTED GENERATION"
        ],
        "keywords": ["paper lookup", "ToG-2", "knowledge-guided retrieval"],
        "difficulty": "easy",
        "notes": "Canonical paper-lookup sample.",
    },
    {
        "id": "sun_definition_31",
        "question": "What is Think-on-Graph (ToG) in the 2024 paper?",
        "answer": "ToG is an approach under an LLM-KG integration paradigm where the LLM acts as an agent that interactively explores a knowledge graph with beam search to perform deep and responsible reasoning.",
        "source_pdf": pdf["sun"],
        "question_type": "definition",
        "expected_intent": "definition",
        "source_pages": [1],
        "evidence": [
            "We propose a new LLM-KG integrating paradigm...",
            "We further implement this paradigm by introducing a new approach called Think-on-Graph (ToG), in which the LLM agent iteratively executes beam search on KG.",
        ],
        "keywords": ["ToG", "LLM-KG", "beam search"],
        "difficulty": "easy",
        "notes": "Definition sample grounded in abstract text.",
    },
    {
        "id": "sun_fact_32",
        "question": "What notable performance and deployment claims does the ToG paper make?",
        "answer": "The paper claims ToG achieves overall SOTA on 6 of 9 datasets without additional training, and that small LLMs with ToG can even exceed large models such as GPT-4 on some scenarios, reducing deployment cost.",
        "source_pdf": pdf["sun"],
        "question_type": "fact_qa",
        "expected_intent": "fact_qa",
        "source_pages": [1],
        "evidence": [
            "The performance of ToG with small LLM models could exceed large LLM such as GPT-4 in certain scenarios and this reduces the cost of LLM deployment and application.",
            "ToG achieves overall SOTA in 6 out of 9 datasets where most previous SOTAs rely on additional training.",
        ],
        "keywords": ["6 out of 9", "GPT-4", "small models"],
        "difficulty": "medium",
        "notes": "Result and deployment claim combined.",
    },
    {
        "id": "sun_comparison_33",
        "question": "Compared with standalone LLM reasoning, what limitation is ToG intended to address?",
        "answer": "It is intended to reduce hallucination and improve deep, responsible reasoning on knowledge-intensive tasks by retrieving and reasoning over explicit KG evidence instead of relying only on parametric knowledge.",
        "source_pdf": pdf["sun"],
        "question_type": "comparison",
        "expected_intent": "comparison",
        "source_pages": [1],
        "evidence": [
            "Although large language models often struggle with hallucination problems, especially in scenarios requiring deep and responsible reasoning, these issues could be partially addressed by introducing external knowledge graphs in LLM reasoning."
        ],
        "keywords": ["hallucination", "responsible reasoning", "external knowledge graphs"],
        "difficulty": "medium",
        "notes": "Comparison of reasoning settings rather than paper-title matching.",
    },
    {
        "id": "sun_survey_34",
        "question": "Summarize the key advantages the ToG paper claims for its framework.",
        "answer": "The paper claims ToG improves deep reasoning power, provides knowledge traceability and correctability, works as a plug-and-play framework across LLMs, KGs, and prompts without extra training, and can reduce deployment cost by enabling smaller models to perform strongly.",
        "source_pdf": pdf["sun"],
        "question_type": "survey",
        "expected_intent": "survey",
        "source_pages": [1],
        "evidence": [
            "Compared with LLMs, ToG has better deep reasoning power.",
            "ToG has the ability of knowledge traceability and knowledge correctability.",
            "ToG provides a flexible plug-and-play framework... without any additional training cost.",
        ],
        "keywords": ["traceability", "correctability", "plug-and-play"],
        "difficulty": "medium",
        "notes": "Compact survey of claimed advantages.",
    },
    {
        "id": "sun_paper_lookup_35",
        "question": "Which paper proposes Think-on-Graph as a deep and responsible reasoning method on knowledge graphs?",
        "answer": "Sun et al. (2024), Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph.",
        "source_pdf": pdf["sun"],
        "question_type": "paper_lookup",
        "expected_intent": "paper_lookup",
        "source_pages": [1],
        "evidence": [
            "THINK-ON-GRAPH: DEEP AND RESPONSIBLE REASONING OF LARGE LANGUAGE MODEL ON KNOWLEDGE GRAPH"
        ],
        "keywords": ["paper lookup", "ToG", "knowledge graph"],
        "difficulty": "easy",
        "notes": "Canonical paper lookup sample.",
    },
    {
        "id": "wu_definition_36",
        "question": "What is Think-on-Graph 3.0 (ToG-3) according to the paper?",
        "answer": "ToG-3 is a graph-based RAG framework built around a Multi-Agent Context Evolution and Retrieval (MACER) mechanism and a Chunk-Triplets-Community heterogeneous graph index for adaptive reasoning.",
        "source_pdf": pdf["wu"],
        "question_type": "definition",
        "expected_intent": "definition",
        "source_pages": [1],
        "evidence": [
            "This paper presents Think-on-Graph 3.0 (ToG-3), a novel framework that introduces Multi-Agent Context Evolution and Retrieval (MACER).",
            "Our core innovation is the dynamic construction and refinement of a Chunk-Triplets-Community heterogeneous graph index.",
        ],
        "keywords": ["ToG-3", "MACER", "heterogeneous graph index"],
        "difficulty": "easy",
        "notes": "Definition grounded in abstract content.",
    },
    {
        "id": "wu_fact_37",
        "question": "Which agents make up the multi-agent system in ToG-3?",
        "answer": "The multi-agent system includes Constructor, Retriever, Reflector, and Responser agents.",
        "source_pdf": pdf["wu"],
        "question_type": "fact_qa",
        "expected_intent": "fact_qa",
        "source_pages": [1],
        "evidence": [
            "A multi-agent system, comprising Constructor, Retriever, Reflector, and Responser agents, collaboratively engages in an iterative process..."
        ],
        "keywords": ["Constructor", "Retriever", "Reflector", "Responser"],
        "difficulty": "easy",
        "notes": "Specific architecture fact sample.",
    },
    {
        "id": "wu_comparison_38",
        "question": "Compared with static graph-based RAG methods, what key adaptation does ToG-3 add during reasoning?",
        "answer": "Unlike static methods that build a graph index in one pass, ToG-3 adaptively evolves both the query and the subgraph during reasoning, allowing targeted graph refinement for the actual question.",
        "source_pdf": pdf["wu"],
        "question_type": "comparison",
        "expected_intent": "comparison",
        "source_pages": [1],
        "evidence": [
            "This approach addresses a critical limitation of prior Graph-based RAG methods, which typically construct a static graph index in a single pass without adapting to the actual query.",
            "A dual-evolution mechanism of Evolving Query and Evolving Sub-Graph...",
        ],
        "keywords": ["static graph index", "Evolving Query", "Evolving Sub-Graph"],
        "difficulty": "medium",
        "notes": "Comparison sample focused on adaptive indexing.",
    },
    {
        "id": "wu_survey_39",
        "question": "Summarize the main retrieval-and-reasoning mechanism of ToG-3.",
        "answer": "ToG-3 uses a multi-agent loop in which evidence retrieval, answer generation, sufficiency reflection, and the evolution of both query and subgraph happen iteratively over a Chunk-Triplets-Community graph, enabling deeper and more precise reasoning with lightweight models.",
        "source_pdf": pdf["wu"],
        "question_type": "survey",
        "expected_intent": "survey",
        "source_pages": [1],
        "evidence": [
            "This dual-evolving multi-agent system allows ToG-3 to adaptively build a targeted graph index during reasoning...",
            "A multi-agent system... collaboratively engages in an iterative process of evidence retrieval, answer generation, sufficiency reflection, and, crucially, evolving query and subgraph.",
        ],
        "keywords": ["iterative process", "dual-evolving", "Chunk-Triplets-Community"],
        "difficulty": "hard",
        "notes": "Designed to probe mechanism-level summarization.",
    },
    {
        "id": "wu_paper_lookup_40",
        "question": "Which paper introduces the MACER mechanism and dual-evolving context retrieval in Think-on-Graph 3.0?",
        "answer": "Wu et al. (2025), Think-on-Graph 3.0: Efficient and Adaptive LLM Reasoning on Heterogeneous Graphs via Multi-Agent Dual-Evolving Context Retrieval.",
        "source_pdf": pdf["wu"],
        "question_type": "paper_lookup",
        "expected_intent": "paper_lookup",
        "source_pages": [1],
        "evidence": [
            "This paper presents Think-on-Graph 3.0 (ToG-3), a novel framework that introduces Multi-Agent Context Evolution and Retrieval (MACER)."
        ],
        "keywords": ["paper lookup", "MACER", "dual-evolving"],
        "difficulty": "easy",
        "notes": "Paper lookup anchored on a unique mechanism name.",
    },
]

jsonl_path = Path("dataset/testpdf_qa_samples_40_refined.jsonl")
readme_path = Path("dataset/testpdf_qa_samples_40_refined_README.md")
jsonl_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")

pdf_counts = Counter(r["source_pdf"] for r in records)
type_counts = Counter(r["question_type"] for r in records)
readme = [
    "# testpdf QA samples (40, refined)",
    "",
    "This dataset supersedes the older 30-sample set for the expanded `testpdf/` collection.",
    "",
    "Files:",
    "- `dataset/testpdf_qa_samples_40_refined.jsonl`",
    "- `dataset/testpdf_page_texts.json`",
    "",
    "Format:",
    "- One JSON object per line.",
    "- Every record includes `id`, `question`, `answer`, `source_pdf`, `question_type`, `expected_intent`, `source_pages`, `evidence`, `keywords`, `difficulty`, and `notes`.",
    "",
    f"- Total samples: {len(records)}",
    f"- PDFs covered: {len(pdf_counts)}",
    f"- Question types covered: {len(type_counts)}",
    "",
    "Coverage by PDF:",
]
readme.extend([f"- `{k}`: {v}" for k, v in pdf_counts.items()])
readme.extend(
    [
        "",
        "Coverage by question type:",
    ]
)
readme.extend([f"- `{k}`: {v}" for k, v in type_counts.items()])
readme.extend(
    [
        "",
        "Refinement notes:",
        "- Expanded coverage from 4 PDFs to 8 PDFs.",
        "- Kept the five AskMyZotero-supported question types.",
        "- Added more method, result, and mechanism-oriented questions to reduce over-reliance on title-only lookup.",
        "- Intended for a review-and-revise workflow using the dataset QA design reviewer.",
        "",
    ]
)
readme_path.write_text("\n".join(readme), encoding="utf-8")

print("written", jsonl_path)
print("written", readme_path)
print("records", len(records))
