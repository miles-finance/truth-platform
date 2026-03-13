"""
Truth Platform Engine — Phase 1 + Phase 2
Phase 1: Article extraction → opposing search → point/counterpoint summary
Phase 2: Claim-level scoring with source strength analysis
"""

import sys
import json
import re
from pathlib import Path
from typing import Optional
import trafilatura
import requests
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
import anthropic

# Load Anthropic key — env var first (Railway), then local load_keys fallback
import os
_api_key = os.environ.get("ANTHROPIC_API_KEY")
if not _api_key:
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "agent-misc"))
        from load_keys import anthropic_api_key
        _api_key = anthropic_api_key()
    except Exception:
        raise RuntimeError("ANTHROPIC_API_KEY env var not set and load_keys fallback failed")

CLIENT = anthropic.Anthropic(api_key=_api_key)
MODEL = "claude-sonnet-4-6"


# ─── Article Extraction ────────────────────────────────────────────────────────

def extract_article(url_or_text: str) -> dict:
    """Extract article content from URL or treat as raw text."""
    if url_or_text.strip().startswith("http"):
        downloaded = trafilatura.fetch_url(url_or_text)
        if not downloaded:
            raise ValueError(f"Could not fetch URL: {url_or_text}")
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False
        )
        if not text:
            raise ValueError("Could not extract content from URL")
        # Try to get title/metadata
        meta = trafilatura.extract_metadata(downloaded)
        title = meta.title if meta and meta.title else url_or_text[:80]
        source_url = url_or_text
    else:
        text = url_or_text
        title = text[:80] + "..."
        source_url = None

    return {
        "text": text[:8000],  # cap for LLM
        "title": title,
        "source_url": source_url
    }


# ─── Web Search ────────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo and return results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        return [{"title": f"Search error: {e}", "href": "", "body": ""}]


def fetch_article_text(url: str, max_chars: int = 4000) -> str:
    """Fetch and extract text from a URL, with fallback to snippet."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False)
            if text and len(text) > 200:
                return text[:max_chars]
    except Exception:
        pass
    return ""


# ─── Claude Helpers ─────────────────────────────────────────────────────────

def claude(prompt: str, system: str = None, max_tokens: int = 2048) -> str:
    """Call Claude and return the text response."""
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": MODEL, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system
    resp = CLIENT.messages.create(**kwargs)
    return resp.content[0].text


def claude_json(prompt: str, system: str = None, max_tokens: int = 2048) -> dict | list:
    """Call Claude and parse JSON from response."""
    text = claude(prompt, system=system, max_tokens=max_tokens)
    # Extract JSON from markdown code blocks if present
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1)
    else:
        # Try to find raw JSON object/array
        text = text.strip()
    return json.loads(text)


# ─── Phase 1: Point/Counterpoint ──────────────────────────────────────────────

def analyze_article_thesis(article_text: str, article_title: str) -> dict:
    """Extract thesis, positions, and search query for opposing content."""
    prompt = f"""Article title: {article_title}

Article text:
{article_text}

Extract the following as JSON:
{{
  "thesis": "1-2 sentence summary of the article's central argument",
  "positions": ["key position 1", "key position 2", "key position 3"],
  "ideology_lean": "left/right/center/scientific-consensus/contrarian/other",
  "topic_keywords": ["keyword1", "keyword2", "keyword3"],
  "opposing_search_query": "A web search query that would find the most diametrically opposed perspective on this topic"
}}

Return ONLY the JSON object, no other text."""

    return claude_json(prompt, system="You are an expert at identifying ideological and argumentative positions in text. Return ONLY valid JSON.")


def find_opposing_article(article_analysis: dict, original_url: str = None) -> dict:
    """Search for and retrieve the best opposing article."""
    query = article_analysis["opposing_search_query"]
    results = search_web(query, max_results=8)

    # Filter out the original source
    filtered = [r for r in results if not original_url or original_url not in r.get("href", "")]

    # Score results with Claude to pick the most diametrically opposed
    if not filtered:
        return {"text": "No opposing content found.", "title": "No results", "source_url": None, "snippet": ""}

    snippets = "\n\n".join([
        f"[{i+1}] Title: {r.get('title','')}\nURL: {r.get('href','')}\nSnippet: {r.get('body','')[:300]}"
        for i, r in enumerate(filtered[:6])
    ])

    selection_prompt = f"""Original article thesis: {article_analysis['thesis']}

Search results:
{snippets}

Which result [1-6] is the MOST diametrically opposed to the original article's thesis?
Respond with ONLY the number (e.g., "3")."""

    choice_text = claude(selection_prompt).strip()
    try:
        idx = int(re.search(r"\d", choice_text).group()) - 1
        idx = max(0, min(idx, len(filtered) - 1))
    except Exception:
        idx = 0

    best = filtered[idx]
    url = best.get("href", "")

    # Try to fetch full text
    full_text = fetch_article_text(url, max_chars=6000) if url else ""
    if not full_text:
        full_text = best.get("body", "")

    return {
        "text": full_text[:6000],
        "title": best.get("title", "Opposing Article"),
        "source_url": url,
        "snippet": best.get("body", "")[:500]
    }


def generate_point_counterpoint(article_a: dict, article_b: dict) -> dict:
    """Generate the Phase 1 point/counterpoint summary."""
    prompt = f"""You are comparing two articles with opposing viewpoints.

ARTICLE A — "{article_a['title']}"
{article_a['text'][:3000]}

ARTICLE B — "{article_b['title']}"
{article_b['text'][:3000]}

Generate a structured point/counterpoint analysis as JSON:
{{
  "article_a_summary": "2-3 sentence neutral summary of Article A's argument",
  "article_b_summary": "2-3 sentence neutral summary of Article B's argument",
  "core_disagreement": "1 sentence: what is the fundamental thing they disagree on?",
  "points": [
    {{
      "topic": "topic name (e.g. 'Economic impact')",
      "article_a_position": "Article A's position on this topic",
      "article_b_position": "Article B's position on this topic"
    }}
  ]
}}

Include 4-6 points covering the most significant disagreements. Return ONLY the JSON."""

    return claude_json(prompt, system="You are a neutral expert analyst. Return ONLY valid JSON.", max_tokens=2000)


# ─── Phase 2: Claim Scoring ──────────────────────────────────────────────────

SOURCE_STRENGTH = {
    "peer_reviewed": 10,
    "academic": 8,
    "government": 8,
    "major_news": 6,
    "think_tank": 5,
    "minor_news": 4,
    "blog": 2,
    "unknown": 3
}

TRUSTED_DOMAINS = {
    "nature.com": "peer_reviewed", "science.org": "peer_reviewed",
    "pubmed.ncbi.nlm.nih.gov": "peer_reviewed", "scholar.google.com": "academic",
    "jstor.org": "academic", ".edu": "academic", ".gov": "government",
    "bbc.com": "major_news", "reuters.com": "major_news", "apnews.com": "major_news",
    "nytimes.com": "major_news", "wsj.com": "major_news", "ft.com": "major_news",
    "economist.com": "major_news", "brookings.edu": "think_tank",
    "pewresearch.org": "think_tank", "rand.org": "think_tank"
}


def classify_source(url: str) -> str:
    if not url:
        return "unknown"
    url_lower = url.lower()
    for domain, stype in TRUSTED_DOMAINS.items():
        if domain in url_lower:
            return stype
    if ".edu" in url_lower:
        return "academic"
    if ".gov" in url_lower:
        return "government"
    return "minor_news"


def extract_claims(article_text: str, article_title: str) -> list[dict]:
    """Extract individual verifiable claims from an article."""
    prompt = f"""Article: "{article_title}"

{article_text[:4000]}

Extract 5-8 specific, verifiable factual claims or assertions from this article.
Each claim should be independently checkable.

Return as JSON array:
[
  {{"claim": "specific verifiable statement", "search_query": "query to find evidence for/against this claim"}}
]

Return ONLY the JSON array."""

    return claude_json(prompt, system="You extract specific, verifiable factual claims. Return ONLY valid JSON.")


def score_claim(claim: dict, article_context: str) -> dict:
    """Search for evidence and score a single claim."""
    results = search_web(claim["search_query"], max_results=6)

    sources = []
    supporting = 0
    contradicting = 0
    total_strength = 0

    for r in results[:5]:
        url = r.get("href", "")
        snippet = r.get("body", "")
        title = r.get("title", "")
        source_type = classify_source(url)
        strength = SOURCE_STRENGTH.get(source_type, 3)

        # Ask Claude if this source supports or contradicts the claim
        stance_prompt = f"""Claim: "{claim['claim']}"

Source title: {title}
Source snippet: {snippet[:400]}

Does this source SUPPORT or CONTRADICT the claim? Answer with one word: "support", "contradict", or "neutral"."""

        try:
            stance = claude(stance_prompt, max_tokens=10).strip().lower()
            if "support" in stance:
                supporting += 1
                total_strength += strength
            elif "contradict" in stance:
                contradicting += 1
                total_strength -= strength // 2
        except Exception:
            stance = "neutral"

        sources.append({
            "url": url,
            "title": title,
            "source_type": source_type,
            "strength": strength,
            "stance": stance,
            "snippet": snippet[:200]
        })

    # Calculate score 0-100
    n_sources = len(sources)
    if n_sources == 0:
        score = 50  # unknown
    else:
        # Base: % supporting weighted by strength
        max_possible = sum(SOURCE_STRENGTH[s["source_type"]] for s in sources)
        if max_possible > 0:
            raw_score = max(0, total_strength) / max_possible
        else:
            raw_score = 0.5

        # Adjust for source count (more sources = more confident)
        coverage_bonus = min(n_sources / 5, 1.0) * 0.2
        score = int((raw_score * 0.8 + coverage_bonus) * 100)
        score = max(0, min(100, score))

    # Credibility label
    if score >= 80:
        label = "Well-Supported"
        color = "green"
    elif score >= 60:
        label = "Moderately Supported"
        color = "yellow"
    elif score >= 40:
        label = "Mixed Evidence"
        color = "orange"
    else:
        label = "Weakly Supported"
        color = "red"

    return {
        "claim": claim["claim"],
        "score": score,
        "label": label,
        "color": color,
        "supporting": supporting,
        "contradicting": contradicting,
        "sources": sources[:4]  # top 4 sources
    }


def score_all_claims(article: dict) -> list[dict]:
    """Run Phase 2 scoring on all claims in an article."""
    claims = extract_claims(article["text"], article["title"])
    scored = []
    for claim in claims:
        try:
            result = score_claim(claim, article["text"])
            scored.append(result)
        except Exception as e:
            scored.append({
                "claim": claim.get("claim", "Unknown claim"),
                "score": 50,
                "label": "Error",
                "color": "gray",
                "supporting": 0,
                "contradicting": 0,
                "sources": [],
                "error": str(e)
            })
    return scored


# ─── Full Analysis Pipeline ───────────────────────────────────────────────────

def run_phase1(url_or_text: str) -> dict:
    """Run Phase 1: extract article, find opposing view, generate point/counterpoint."""
    # Extract original article
    article_a = extract_article(url_or_text)

    # Analyze thesis and find opposing search query
    analysis = analyze_article_thesis(article_a["text"], article_a["title"])
    article_a["analysis"] = analysis

    # Find opposing article
    article_b = find_opposing_article(analysis, original_url=article_a.get("source_url"))
    article_b["analysis"] = analyze_article_thesis(article_b["text"], article_b["title"])

    # Generate point/counterpoint
    point_counterpoint = generate_point_counterpoint(article_a, article_b)

    return {
        "article_a": {
            "title": article_a["title"],
            "source_url": article_a.get("source_url"),
            "summary": point_counterpoint["article_a_summary"],
            "ideology_lean": analysis.get("ideology_lean", "unknown")
        },
        "article_b": {
            "title": article_b["title"],
            "source_url": article_b.get("source_url"),
            "summary": point_counterpoint["article_b_summary"],
            "ideology_lean": article_b["analysis"].get("ideology_lean", "unknown")
        },
        "core_disagreement": point_counterpoint["core_disagreement"],
        "points": point_counterpoint["points"],
        # Store full text for Phase 2
        "_article_a_text": article_a["text"],
        "_article_b_text": article_b["text"]
    }


def run_phase2(phase1_result: dict) -> dict:
    """Run Phase 2: score claims from both articles."""
    article_a = {
        "text": phase1_result.get("_article_a_text", ""),
        "title": phase1_result["article_a"]["title"]
    }
    article_b = {
        "text": phase1_result.get("_article_b_text", ""),
        "title": phase1_result["article_b"]["title"]
    }

    claims_a = score_all_claims(article_a)
    claims_b = score_all_claims(article_b)

    # Average scores
    avg_a = sum(c["score"] for c in claims_a) / len(claims_a) if claims_a else 0
    avg_b = sum(c["score"] for c in claims_b) / len(claims_b) if claims_b else 0

    return {
        "article_a_claims": claims_a,
        "article_b_claims": claims_b,
        "article_a_credibility_score": round(avg_a),
        "article_b_credibility_score": round(avg_b)
    }
