"""Tests for citation_tracker — cross-round citation dedup and formatting."""

import pytest
from src.agentic.citation_tracker import CitationTracker
from src.core.types import RetrievalResult


class TestCitationTrackerAdd:
    """Test adding results and dedup."""

    def test_add_single_result(self):
        tracker = CitationTracker()
        results = [
            RetrievalResult(
                chunk_id="doc1_chunk_001",
                score=0.95,
                text="Azure OpenAI 配置...",
                metadata={"source_path": "docs/azure-guide.pdf", "title": "Azure Guide"},
            )
        ]
        tracker.add(results)
        assert len(tracker.all_sources) == 1
        assert "docs/azure-guide.pdf" in tracker.all_sources

    def test_add_dedup_same_chunk_id(self):
        tracker = CitationTracker()
        r1 = RetrievalResult(
            chunk_id="doc1_chunk_001",
            score=0.95,
            text="Azure config...",
            metadata={"source_path": "docs/azure-guide.pdf"},
        )
        r2 = RetrievalResult(
            chunk_id="doc1_chunk_001",  # same chunk_id
            score=0.90,
            text="Azure config...",
            metadata={"source_path": "docs/azure-guide.pdf"},
        )
        tracker.add([r1])
        tracker.add([r2])
        assert len(tracker.all_sources) == 1

    def test_add_multiple_rounds_across_sources(self):
        tracker = CitationTracker()
        tracker.add([
            RetrievalResult(
                chunk_id="doc1_chunk_001", score=0.95, text="...",
                metadata={"source_path": "docs/projectA.md"},
            )
        ])
        tracker.add([
            RetrievalResult(
                chunk_id="doc2_chunk_003", score=0.88, text="...",
                metadata={"source_path": "docs/projectB.md"},
            )
        ])
        sources = tracker.all_sources
        assert len(sources) == 2
        assert "docs/projectA.md" in sources
        assert "docs/projectB.md" in sources

    def test_add_empty_list(self):
        tracker = CitationTracker()
        tracker.add([])
        assert len(tracker.all_sources) == 0


class TestCitationTrackerFormat:
    """Test citation table formatting."""

    def test_format_single(self):
        tracker = CitationTracker()
        tracker.add([
            RetrievalResult(
                chunk_id="doc1_chunk_001", score=0.95, text="...",
                metadata={"source_path": "docs/report.pdf", "title": "Annual Report"},
            )
        ])
        formatted = tracker.format()
        assert "[1]" in formatted
        assert "docs/report.pdf" in formatted

    def test_format_multiple_ordered(self):
        tracker = CitationTracker()
        tracker.add([
            RetrievalResult(
                chunk_id="doc1_chunk_001", score=0.95, text="...",
                metadata={"source_path": "docs/first.pdf"},
            )
        ])
        tracker.add([
            RetrievalResult(
                chunk_id="doc2_chunk_001", score=0.80, text="...",
                metadata={"source_path": "docs/second.pdf"},
            )
        ])
        formatted = tracker.format()
        lines = formatted.strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("[1]")
        assert lines[1].startswith("[2]")
        assert "docs/first.pdf" in lines[0]
        assert "docs/second.pdf" in lines[1]

    def test_format_empty(self):
        tracker = CitationTracker()
        assert tracker.format() == ""
