"""Cross-round citation tracker for ReAct agent.

Tracks retrieval results across multiple rounds, deduplicates by chunk_id,
and formats a numbered citation table for the final answer.
"""

from __future__ import annotations

from typing import Dict, List

from src.core.types import RetrievalResult


class CitationTracker:
    """Accumulates retrieval results across ReAct rounds.

    Deduplicates by chunk_id and provides ordered source listing
    for citation table generation. Citations are based ONLY on
    actual retrieval results, not LLM self-reported sources —
    this prevents citation hallucination.
    """

    def __init__(self) -> None:
        self._seen_ids: Dict[str, str] = {}  # chunk_id -> source_path
        self._ordered_sources: List[str] = []

    def add(self, results: List[RetrievalResult]) -> None:
        """Add retrieval results from one round.

        Deduplicates by chunk_id — if the same chunk is retrieved
        in multiple rounds, its source only appears once.

        Args:
            results: List of RetrievalResult from a tool call.
        """
        for r in results:
            if r.chunk_id not in self._seen_ids:
                source = r.metadata.get("source_path", r.chunk_id)
                self._seen_ids[r.chunk_id] = source
                if source not in self._ordered_sources:
                    self._ordered_sources.append(source)

    @property
    def all_sources(self) -> List[str]:
        """Deduplicated list of source paths in order of first appearance.

        Returns:
            List of unique source_path strings.
        """
        return list(self._ordered_sources)

    def format(self) -> str:
        """Generate a numbered citation table.

        Returns:
            Multi-line string with [1] source_path format,
            or empty string if no citations accumulated.
        """
        if not self._ordered_sources:
            return ""

        lines = []
        for i, source in enumerate(self._ordered_sources, start=1):
            lines.append(f"[{i}] {source}")

        return "\n".join(lines)
