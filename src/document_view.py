from typing import Any


def group_snippets_by_paper(snippets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将片段按论文（source）聚合，避免同一篇文献重复显示多张卡片。"""
    grouped: dict[str, dict[str, Any]] = {}
    for snip in snippets:
        source = snip.get("source_path") or snip.get("source", "unknown")
        paper = grouped.setdefault(
            source,
            {
                "title": snip.get("title", snip.get("source", "unknown")),
                "source_path": source,
                "pages": set(),
                "snippets": [],
                "best_rank": 10**9,
            },
        )
        page = snip.get("page")
        if page is not None:
            paper["pages"].add(page)
        rank = snip.get("rank", 10**9)
        if isinstance(rank, int):
            paper["best_rank"] = min(paper["best_rank"], rank)
        paper["snippets"].append({"content": snip.get("content", ""), "page": page, "rank": rank})

    papers = []
    # 同论文多片段排序：先按检索顺序(rank)，再按页码
    ordered_papers = sorted(
        grouped.values(),
        key=lambda d: (d["best_rank"], -len(d["snippets"]), d["title"].lower()),
    )

    for rank, data in enumerate(ordered_papers, start=1):
        sorted_snippets = sorted(
            data["snippets"],
            key=lambda s: (s.get("rank", 10**9), s.get("page") if s.get("page") is not None else 10**9),
        )
        pages_sorted = sorted(data["pages"])
        evidence = [
            {
                "page": s.get("page"),
                "content": (s.get("content") or "").strip(),
                "rank": s.get("rank"),
            }
            for s in sorted_snippets
        ]
        papers.append(
            {
                "title": data["title"],
                "source_path": data["source_path"],
                "page": pages_sorted[0] if pages_sorted else None,
                "pages": pages_sorted,
                "snippet_count": len(sorted_snippets),
                "evidence_snippets": evidence,
                "abstract": "",
                "score": float(max(0.0, 1.0 - (rank - 1) * 0.1)),
            }
        )
    return papers
