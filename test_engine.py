"""Tests for engine.py — unit tests with mocked external calls."""

import json
from unittest.mock import MagicMock, patch

import pytest

# Must mock external deps before importing engine
with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
    with patch("anthropic.Anthropic"):
        import engine


class TestClassifySource:
    def test_peer_reviewed(self):
        assert engine.classify_source("https://nature.com/articles/123") == "peer_reviewed"
        assert engine.classify_source("https://pubmed.ncbi.nlm.nih.gov/12345") == "peer_reviewed"

    def test_academic(self):
        assert engine.classify_source("https://stanford.edu/papers/foo") == "academic"

    def test_government(self):
        assert engine.classify_source("https://cdc.gov/data") == "government"

    def test_major_news(self):
        assert engine.classify_source("https://bbc.com/news/article") == "major_news"
        assert engine.classify_source("https://reuters.com/story") == "major_news"

    def test_minor_news_fallback(self):
        assert engine.classify_source("https://randomsite.com/article") == "minor_news"

    def test_empty_url(self):
        assert engine.classify_source("") == "unknown"
        assert engine.classify_source(None) == "unknown"


class TestDetectStance:
    def test_support(self):
        assert engine._detect_stance_from_snippet(
            "Climate change is real",
            "Study confirms warming trend",
            "New research confirms and demonstrates rising temperatures"
        ) == "support"

    def test_contradict(self):
        assert engine._detect_stance_from_snippet(
            "Vaccines cause autism",
            "Myth debunked by researchers",
            "This claim is false and has been debunked"
        ) == "contradict"

    def test_neutral(self):
        assert engine._detect_stance_from_snippet(
            "Some claim",
            "A news article",
            "The situation is complicated with many factors"
        ) == "neutral"


class TestSearchWeb:
    @patch.object(engine, "DDGS")
    def test_returns_results(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [{"title": "Result", "href": "https://example.com", "body": "text"}]
        mock_ddgs_cls.return_value = mock_ddgs

        results = engine.search_web("test query")
        assert len(results) == 1
        assert results[0]["title"] == "Result"

    @patch.object(engine, "DDGS")
    def test_returns_empty_on_error(self, mock_ddgs_cls):
        mock_ddgs_cls.side_effect = Exception("Network error")
        results = engine.search_web("test query")
        assert results == []


class TestExtractArticle:
    @patch("engine.trafilatura")
    def test_url_extraction(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html>content</html>"
        mock_traf.extract.return_value = "Article text here " * 20
        mock_meta = MagicMock()
        mock_meta.title = "Test Article"
        mock_traf.extract_metadata.return_value = mock_meta

        result = engine.extract_article("https://example.com/article")
        assert result["title"] == "Test Article"
        assert result["source_url"] == "https://example.com/article"
        assert "Article text here" in result["text"]

    def test_raw_text(self):
        result = engine.extract_article("This is some raw article text that I pasted in directly.")
        assert result["source_url"] is None
        assert "raw article text" in result["text"]

    @patch("engine.trafilatura")
    def test_url_fetch_failure(self, mock_traf):
        mock_traf.fetch_url.return_value = None
        with pytest.raises(ValueError, match="Could not fetch URL"):
            engine.extract_article("https://bad-url.com")


class TestClaudeJson:
    @patch.object(engine, "claude")
    def test_parses_json(self, mock_claude):
        mock_claude.return_value = '{"key": "value"}'
        result = engine.claude_json("test prompt")
        assert result == {"key": "value"}

    @patch.object(engine, "claude")
    def test_parses_json_from_code_block(self, mock_claude):
        mock_claude.return_value = '```json\n{"key": "value"}\n```'
        result = engine.claude_json("test prompt")
        assert result == {"key": "value"}

    @patch.object(engine, "claude")
    def test_parses_array(self, mock_claude):
        mock_claude.return_value = '[{"a": 1}, {"b": 2}]'
        result = engine.claude_json("test prompt")
        assert len(result) == 2


class TestScoreClaim:
    @patch.object(engine, "search_web")
    def test_no_results_returns_50(self, mock_search):
        mock_search.return_value = []
        result = engine.score_claim(
            {"claim": "test claim", "search_query": "test"},
            "context"
        )
        assert result["score"] == 50
        assert result["sources"] == []

    @patch.object(engine, "search_web")
    def test_with_supporting_results(self, mock_search):
        mock_search.return_value = [
            {"title": "Study confirms claim", "href": "https://nature.com/123", "body": "Research confirms and demonstrates this."},
            {"title": "Evidence shows truth", "href": "https://bbc.com/news", "body": "Evidence shows and proves the claim."},
        ]
        result = engine.score_claim(
            {"claim": "test claim", "search_query": "test"},
            "context"
        )
        assert result["score"] > 50
        assert result["supporting"] >= 1
        assert len(result["sources"]) <= 4


class TestFindOpposingArticle:
    @patch.object(engine, "search_web")
    def test_no_results_returns_empty(self, mock_search):
        mock_search.return_value = []
        result = engine.find_opposing_article(
            {"opposing_search_query": "test", "thesis": "some thesis"}
        )
        assert result["text"] == ""
        assert result["title"] == "No opposing article found"

    @patch.object(engine, "fetch_article_text")
    @patch.object(engine, "claude")
    @patch.object(engine, "search_web")
    def test_selects_opposing_article(self, mock_search, mock_claude, mock_fetch):
        mock_search.return_value = [
            {"title": "Opposing view", "href": "https://example.com/oppose", "body": "Counter argument text"},
        ]
        mock_claude.return_value = "1"
        mock_fetch.return_value = "Full opposing article text here."

        result = engine.find_opposing_article(
            {"opposing_search_query": "test", "thesis": "some thesis"}
        )
        assert result["title"] == "Opposing view"
        assert result["source_url"] == "https://example.com/oppose"
