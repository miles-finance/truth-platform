#!/usr/bin/env python3
"""
Truth Platform — Article Analysis & Claim Scoring Engine

Phase 1: Analyze an article, find the most diametrically opposed content,
          return a point/counterpoint summary.
Phase 2: Score each claim from both sides with a quantitative evidence rating
          based on the number and strength of confirmatory sources.

Usage:
    python3 truth_engine.py "https://example.com/article"
    python3 truth_engine.py --text "paste article text here"
    python3 truth_engine.py --phase 1 "https://example.com/article"
    python3 truth_engine.py --phase 2 "https://example.com/article"
"""

import sys
import json
import re
import time
import argparse
import textwrap
from dataclasses import dataclass, field, asdict
from typing import Optional

import httpx
import anthropic
from ddgs import DDGS
from trafilatura import fetch_url, extract

# ── Load API key ─────────────────────────────────────────────────────
sys.path.insert(0, "/Users/mileslavin/Documents/AGENTS/agent-misc")
try:
    from load_keys import anthropic_api_key
    API_KEY = anthropic_api_key()
except Exception:
    import os
    API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

CLIENT = anthropic.Anthropic(api_key=API_KEY)
MODEL = "claude-sonnet-4-20250514"
SEARCH = DDGS()

# User-Agent for fetching
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"


# ── Data Models ──────────────────────────────────────────────────────
@dataclass
class Claim:
    text: str
    side: str  # "original" or "opposing"
    category: str = ""  # e.g. "economic", "scientific", "moral"


@dataclass
class ScoredClaim:
    claim: str
    side: str
    score: float  # 0-10
    confidence: str  # "high", "medium", "low"
    supporting_sources: list = field(default_factory=list)
    contradicting_sources: list = field(default_factory=list)
    reasoning: str = ""


@dataclass
class Phase1Result:
    original_title: str
    original_summary: str
    original_position: str
    original_claims: list[Claim]
    opposing_title: str
    opposing_summary: str
    opposing_position: str
    opposing_claims: list[Claim]
    opposing_sources: list[dict]
    point_counterpoint: list[dict]  # [{point: str, counterpoint: str, topic: str}]


@dataclass
class Phase2Result:
    scored_claims: list[ScoredClaim]
    overall_original_score: float
    overall_opposing_score: float
    verdict: str
    methodology: str


@dataclass
class FullAnalysis:
    phase1: Phase1Result
    phase2: Phase2Result


# ── Article Extraction ───────────────────────────────────────────────
def extract_article(url_or_text: str) -> dict:
    """Extract article content from a URL or return raw text."""
    if url_or_text.startswith("http://") or url_or_text.startswith("https://"):
        text = None
        # Try trafilatura first (best at article extraction)
        downloaded = fetch_url(url_or_text)
        if downloaded:
            text = extract(downloaded, include_comments=False, include_tables=False)

        # Fallback: httpx with browser headers
        if not text or len(text) < 200:
            resp = httpx.get(url_or_text, follow_redirects=True, timeout=15,
                             headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"})
            if resp.status_code >= 400:
                raise ValueError(f"URL returned HTTP {resp.status_code}")
            extracted = extract(resp.text, include_comments=False, include_tables=False)
            if extracted and len(extracted) > len(text or ""):
                text = extracted

        if not text or len(text) < 200:
            raise ValueError(f"Could not extract sufficient text from URL (got {len(text or '')} chars)")

        return {"text": text, "url": url_or_text, "source": "web"}
    else:
        return {"text": url_or_text, "url": None, "source": "direct_input"}


def search_web(query: str, max_results: int = 8) -> list[dict]:
    """Search the web and return results with snippets."""
    results = []
    try:
        for r in SEARCH.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    except Exception as e:
        print(f"  [search warning] {e}", file=sys.stderr)
    return results


def fetch_article_text(url: str) -> str:
    """Fetch and extract text from a URL."""
    try:
        # Try trafilatura first
        downloaded = fetch_url(url)
        if downloaded:
            text = extract(downloaded, include_comments=False, include_tables=False)
            if text and len(text) > 200:
                return text[:8000]
        # Fallback: httpx with full browser UA
        resp = httpx.get(url, follow_redirects=True, timeout=15,
                         headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"})
        if resp.status_code == 200:
            text = extract(resp.text, include_comments=False)
            if text and len(text) > 200:
                return text[:8000]
            # Last resort: strip HTML tags roughly
            import re as _re
            raw = _re.sub(r'<[^>]+>', ' ', resp.text)
            raw = _re.sub(r'\s+', ' ', raw).strip()
            if len(raw) > 200:
                return raw[:8000]
        return f"[Could not fetch: HTTP {resp.status_code}]"
    except Exception as e:
        return f"[Could not fetch: {e}]"


# ── Claude API Helpers ───────────────────────────────────────────────
def ask_claude(system: str, prompt: str, max_tokens: int = 4096) -> str:
    """Send a prompt to Claude and return the text response."""
    resp = CLIENT.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def ask_claude_json(system: str, prompt: str, max_tokens: int = 4096) -> dict:
    """Send a prompt to Claude and parse the JSON response."""
    text = ask_claude(system, prompt, max_tokens)
    # Extract JSON from response (handle markdown code blocks)
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    # Try to find JSON object or array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


# ── Phase 1: Point/Counterpoint ─────────────────────────────────────
SYSTEM_ANALYZE = """You are a rigorous analytical engine for a truth-validation platform.
Your job is to analyze articles with complete intellectual honesty — no bias, no hedging, no false balance.
Extract the core position, key claims, and identify what the strongest possible opposition would argue.
Always respond in valid JSON."""

SYSTEM_SEARCH = """You generate precise web search queries designed to find content that is
DIAMETRICALLY OPPOSED to a given position. Not just "different" — the strongest, most articulate
counterargument. Think: if this article is the prosecution, find the best defense attorney.
Always respond in valid JSON."""

SYSTEM_SYNTHESIZE = """You are a truth-platform synthesis engine. You produce clear, structured
point/counterpoint summaries. You never take sides — you present each position at its strongest.
The reader should finish understanding BOTH arguments better than they did before.
Always respond in valid JSON."""


def phase1_analyze_article(article: dict) -> dict:
    """Step 1: Analyze the original article."""
    print("  [Phase 1.1] Analyzing original article...")
    result = ask_claude_json(
        SYSTEM_ANALYZE,
        f"""Analyze this article and extract its core components.

ARTICLE TEXT:
{article['text'][:12000]}

Return JSON:
{{
    "title": "article title or generated title",
    "summary": "2-3 sentence summary of the article's main argument",
    "position": "the article's core position in one sentence",
    "bias_direction": "left/right/pro-industry/anti-industry/pro-regulation/libertarian/etc.",
    "claims": [
        {{
            "text": "specific factual or argumentative claim",
            "category": "economic|scientific|moral|political|social|technical",
            "strength": "strong|moderate|weak (how central is this to the argument)"
        }}
    ],
    "opposition_angle": "what would the strongest counterargument focus on?"
}}"""
    )
    return result


def phase1_find_opposition(analysis: dict) -> list[dict]:
    """Step 2: Generate search queries and find opposing content."""
    print("  [Phase 1.2] Generating opposition search queries...")
    queries = ask_claude_json(
        SYSTEM_SEARCH,
        f"""Given this article's position, generate 4-5 search queries that would find
the MOST DIAMETRICALLY OPPOSED content. Not just "different" — find the strongest counter-position.

ARTICLE POSITION: {analysis['position']}
BIAS DIRECTION: {analysis.get('bias_direction', 'unknown')}
KEY CLAIMS: {json.dumps(analysis['claims'][:5])}
SUGGESTED OPPOSITION ANGLE: {analysis.get('opposition_angle', '')}

Return JSON:
{{
    "queries": [
        "search query 1 targeting the strongest opposing view",
        "search query 2 targeting academic/research counterevidence",
        "search query 3 targeting opposing expert opinions",
        "search query 4 targeting data that contradicts the claims"
    ],
    "opposition_thesis": "what the ideal opposing article would argue"
}}"""
    )

    print("  [Phase 1.3] Searching for opposing content...")
    all_results = []
    seen_urls = set()
    for q in queries.get("queries", [])[:5]:
        results = search_web(q, max_results=6)
        for r in results:
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                all_results.append(r)
        time.sleep(0.5)  # Rate limiting

    return all_results, queries.get("opposition_thesis", "")


def phase1_select_best_opposition(analysis: dict, search_results: list[dict],
                                   opposition_thesis: str) -> list[dict]:
    """Step 3: Select and analyze the best opposing source(s)."""
    print("  [Phase 1.4] Selecting best opposing source...")

    # Ask Claude to rank the results by opposition strength
    ranking = ask_claude_json(
        SYSTEM_ANALYZE,
        f"""From these search results, identify the top 5 that most directly OPPOSE this position:

ORIGINAL POSITION: {analysis['position']}
DESIRED OPPOSITION: {opposition_thesis}

SEARCH RESULTS:
{json.dumps(search_results[:25], indent=2)}

Return JSON:
{{
    "ranked": [
        {{
            "url": "url of the best opposing source",
            "title": "title",
            "why": "why this is a strong counterargument",
            "opposition_strength": 1-10
        }}
    ]
}}"""
    )

    # Fetch the top opposing articles — try up to 6, keep best 2
    opposing_texts = []
    for item in ranking.get("ranked", [])[:6]:
        url = item.get("url", "")
        if not url:
            continue
        print(f"    Trying: {item.get('title', url)[:60]}...")
        text = fetch_article_text(url)
        if "[Could not fetch" not in text:
            opposing_texts.append({
                "url": url,
                "title": item.get("title", ""),
                "text": text,
                "opposition_strength": item.get("opposition_strength", 5),
            })
            print(f"    ✓ Got {len(text)} chars")
            if len(opposing_texts) >= 2:
                break
        else:
            print(f"    ✗ {text[:80]}")

    # Fallback: if we couldn't fetch any full articles, use search snippets as content
    if not opposing_texts:
        print("    [fallback] Using search snippets as opposing content...")
        # Collect the best snippets from search results
        snippet_text = "\n\n".join([
            f"Source: {r['title']} ({r['url']})\n{r['snippet']}"
            for r in search_results[:10]
            if r.get("snippet")
        ])
        if snippet_text:
            opposing_texts.append({
                "url": "multiple_search_results",
                "title": "Aggregated opposing perspectives from search results",
                "text": snippet_text[:8000],
                "opposition_strength": 5,
            })

    return opposing_texts


def phase1_synthesize(article: dict, analysis: dict, opposing_articles: list[dict]) -> Phase1Result:
    """Step 4: Create the point/counterpoint synthesis."""
    print("  [Phase 1.5] Synthesizing point/counterpoint...")

    opposing_content = "\n\n---\n\n".join([
        f"SOURCE: {a['title']} ({a['url']})\n{a['text'][:6000]}"
        for a in opposing_articles
    ])

    synthesis = ask_claude_json(
        SYSTEM_SYNTHESIZE,
        f"""Create a structured point/counterpoint analysis.

ORIGINAL ARTICLE:
Title: {analysis.get('title', 'Unknown')}
Position: {analysis['position']}
Summary: {analysis['summary']}
Key Claims: {json.dumps(analysis['claims'])}

OPPOSING SOURCES:
{opposing_content[:15000]}

Return JSON:
{{
    "opposing_title": "synthesized title for the opposing position",
    "opposing_summary": "2-3 sentence summary of the opposing argument",
    "opposing_position": "the opposition's core position in one sentence",
    "opposing_claims": [
        {{
            "text": "specific opposing claim",
            "category": "economic|scientific|moral|political|social|technical"
        }}
    ],
    "point_counterpoint": [
        {{
            "topic": "the issue being debated",
            "point": "the original article's argument on this topic",
            "counterpoint": "the opposing argument on this topic",
            "key_tension": "what makes this disagreement fundamental"
        }}
    ]
}}""",
        max_tokens=6000,
    )

    return Phase1Result(
        original_title=analysis.get("title", "Unknown"),
        original_summary=analysis.get("summary", ""),
        original_position=analysis.get("position", ""),
        original_claims=[
            Claim(text=c["text"], side="original", category=c.get("category", ""))
            for c in analysis.get("claims", [])
        ],
        opposing_title=synthesis.get("opposing_title", ""),
        opposing_summary=synthesis.get("opposing_summary", ""),
        opposing_position=synthesis.get("opposing_position", ""),
        opposing_claims=[
            Claim(text=c["text"], side="opposing", category=c.get("category", ""))
            for c in synthesis.get("opposing_claims", [])
        ],
        opposing_sources=[{"url": a["url"], "title": a["title"]} for a in opposing_articles],
        point_counterpoint=synthesis.get("point_counterpoint", []),
    )


def run_phase1(article_input: str) -> Phase1Result:
    """Run the complete Phase 1 pipeline."""
    print("\n━━━ PHASE 1: Point/Counterpoint Analysis ━━━\n")

    # Extract article
    print("  [Phase 1.0] Extracting article content...")
    article = extract_article(article_input)
    if not article["text"] or len(article["text"]) < 100:
        raise ValueError("Could not extract sufficient article content.")

    # Analyze
    analysis = phase1_analyze_article(article)

    # Find opposition
    search_results, opposition_thesis = phase1_find_opposition(analysis)
    if not search_results:
        raise ValueError("Could not find opposing sources via web search.")

    # Select best opposition
    opposing_articles = phase1_select_best_opposition(analysis, search_results, opposition_thesis)
    if not opposing_articles:
        raise ValueError("Could not fetch any opposing articles.")

    # Synthesize
    result = phase1_synthesize(article, analysis, opposing_articles)

    print("\n  ✓ Phase 1 complete.\n")
    return result


# ── Phase 2: Claim Scoring ───────────────────────────────────────────
SYSTEM_SCORE = """You are a claim-scoring engine for a truth-validation platform.
You evaluate individual claims based on evidence found through web search.
Your scoring must be rigorous and evidence-based:
- Score 0-10 where 10 = overwhelmingly supported by high-quality evidence
- Weight academic papers and peer-reviewed research highest
- Weight established news outlets and expert analysis medium
- Weight opinion pieces, blogs, and social media lowest
- Contradicting evidence from strong sources should significantly reduce the score
Always respond in valid JSON."""

SYSTEM_QUERIES = """You generate precise search queries to find evidence FOR and AGAINST specific claims.
For each claim, generate queries that would find:
1. The strongest SUPPORTING evidence (academic papers, data, expert testimony)
2. The strongest CONTRADICTING evidence (counter-studies, opposing data, expert rebuttals)
Always respond in valid JSON."""


def phase2_generate_evidence_queries(claims: list[Claim]) -> list[dict]:
    """Generate search queries for each claim."""
    print("  [Phase 2.1] Generating evidence search queries...")

    claims_text = json.dumps([{"text": c.text, "side": c.side, "category": c.category}
                              for c in claims])

    result = ask_claude_json(
        SYSTEM_QUERIES,
        f"""For each claim below, generate 2 search queries:
one to find SUPPORTING evidence and one to find CONTRADICTING evidence.

CLAIMS:
{claims_text}

Return JSON:
{{
    "claim_queries": [
        {{
            "claim": "the claim text",
            "side": "original or opposing",
            "support_query": "search query to find supporting evidence",
            "contradict_query": "search query to find contradicting evidence"
        }}
    ]
}}""",
    )
    return result.get("claim_queries", [])


def phase2_gather_evidence(claim_queries: list[dict]) -> list[dict]:
    """Search for evidence for each claim."""
    print(f"  [Phase 2.2] Gathering evidence for {len(claim_queries)} claims...")

    evidence = []
    for i, cq in enumerate(claim_queries):
        print(f"    Claim {i+1}/{len(claim_queries)}: {cq['claim'][:60]}...")

        support_results = search_web(cq.get("support_query", ""), max_results=5)
        time.sleep(0.3)
        contradict_results = search_web(cq.get("contradict_query", ""), max_results=5)
        time.sleep(0.3)

        evidence.append({
            "claim": cq["claim"],
            "side": cq.get("side", "unknown"),
            "support_results": support_results,
            "contradict_results": contradict_results,
        })

    return evidence


def phase2_score_claims(evidence: list[dict]) -> list[ScoredClaim]:
    """Score each claim based on gathered evidence."""
    print("  [Phase 2.3] Scoring claims based on evidence...")

    # Process in batches to manage token limits
    scored = []
    batch_size = 5
    for i in range(0, len(evidence), batch_size):
        batch = evidence[i:i + batch_size]
        print(f"    Scoring batch {i//batch_size + 1}...")

        result = ask_claude_json(
            SYSTEM_SCORE,
            f"""Score each claim based on the search evidence provided.

For each claim, evaluate:
1. How many sources support it? (quantity)
2. How strong are the supporting sources? (quality: academic > news > blog)
3. How many sources contradict it? (counter-evidence)
4. How strong are the contradicting sources?

Evidence score formula (explain your reasoning):
- Base: start at 5.0
- Each strong supporting source: +0.5 to +1.5 (academic/peer-reviewed: +1.5, major news: +1.0, other: +0.5)
- Each strong contradicting source: -0.5 to -1.5 (same weights)
- Cap at 0.0 minimum, 10.0 maximum

EVIDENCE:
{json.dumps(batch, indent=2)[:12000]}

Return JSON:
{{
    "scores": [
        {{
            "claim": "the claim text",
            "side": "original or opposing",
            "score": 7.5,
            "confidence": "high|medium|low",
            "supporting_sources": [
                {{"title": "source title", "url": "url", "type": "academic|news|expert|blog", "relevance": "how it supports"}}
            ],
            "contradicting_sources": [
                {{"title": "source title", "url": "url", "type": "academic|news|expert|blog", "relevance": "how it contradicts"}}
            ],
            "reasoning": "brief explanation of how the score was derived"
        }}
    ]
}}""",
            max_tokens=6000,
        )

        for s in result.get("scores", []):
            scored.append(ScoredClaim(
                claim=s["claim"],
                side=s.get("side", "unknown"),
                score=s.get("score", 5.0),
                confidence=s.get("confidence", "medium"),
                supporting_sources=s.get("supporting_sources", []),
                contradicting_sources=s.get("contradicting_sources", []),
                reasoning=s.get("reasoning", ""),
            ))

    return scored


def phase2_verdict(scored_claims: list[ScoredClaim]) -> dict:
    """Generate overall verdict based on all scored claims."""
    print("  [Phase 2.4] Generating verdict...")

    original_scores = [c.score for c in scored_claims if c.side == "original"]
    opposing_scores = [c.score for c in scored_claims if c.side == "opposing"]

    orig_avg = sum(original_scores) / len(original_scores) if original_scores else 5.0
    opp_avg = sum(opposing_scores) / len(opposing_scores) if opposing_scores else 5.0

    claims_summary = json.dumps([{
        "claim": c.claim, "side": c.side, "score": c.score,
        "confidence": c.confidence, "reasoning": c.reasoning
    } for c in scored_claims], indent=2)

    verdict = ask_claude_json(
        SYSTEM_SCORE,
        f"""Based on the scored claims below, provide a final verdict.

ORIGINAL ARTICLE average score: {orig_avg:.1f}/10
OPPOSING POSITION average score: {opp_avg:.1f}/10

SCORED CLAIMS:
{claims_summary[:8000]}

Return JSON:
{{
    "verdict": "A 2-3 sentence balanced verdict. Which side has stronger evidence? Where do they each have strong/weak points?",
    "strongest_original_claim": "the best-supported claim from the original article",
    "weakest_original_claim": "the least-supported claim from the original article",
    "strongest_opposing_claim": "the best-supported claim from the opposition",
    "weakest_opposing_claim": "the least-supported claim from the opposition",
    "nuance": "what both sides get partially right or where the truth lies between them"
}}""",
    )
    return verdict, orig_avg, opp_avg


def run_phase2(phase1_result: Phase1Result) -> Phase2Result:
    """Run the complete Phase 2 pipeline."""
    print("\n━━━ PHASE 2: Claim Scoring & Evidence Rating ━━━\n")

    # Combine all claims
    all_claims = phase1_result.original_claims + phase1_result.opposing_claims

    if not all_claims:
        raise ValueError("No claims to score. Phase 1 may have failed.")

    # Generate search queries
    claim_queries = phase2_generate_evidence_queries(all_claims)

    # Gather evidence
    evidence = phase2_gather_evidence(claim_queries)

    # Score claims
    scored_claims = phase2_score_claims(evidence)

    # Generate verdict
    verdict_data, orig_avg, opp_avg = phase2_verdict(scored_claims)

    print("\n  ✓ Phase 2 complete.\n")
    return Phase2Result(
        scored_claims=scored_claims,
        overall_original_score=orig_avg,
        overall_opposing_score=opp_avg,
        verdict=verdict_data.get("verdict", ""),
        methodology=(
            "Claims scored 0-10 based on web evidence. "
            "Academic/peer-reviewed sources weighted highest (+1.5), "
            "major news outlets medium (+1.0), blogs/opinion lowest (+0.5). "
            "Contradicting evidence applies same weights as deductions."
        ),
    )


# ── Full Analysis ────────────────────────────────────────────────────
def run_full_analysis(article_input: str) -> FullAnalysis:
    """Run both phases end-to-end."""
    phase1 = run_phase1(article_input)
    phase2 = run_phase2(phase1)
    return FullAnalysis(phase1=phase1, phase2=phase2)


# ── Output Formatting ────────────────────────────────────────────────
def format_phase1(result: Phase1Result) -> str:
    """Format Phase 1 results for display."""
    lines = []
    lines.append("=" * 70)
    lines.append("  TRUTH PLATFORM — POINT / COUNTERPOINT ANALYSIS")
    lines.append("=" * 70)
    lines.append("")

    # Original
    lines.append(f"📰 ORIGINAL: {result.original_title}")
    lines.append(f"   Position: {result.original_position}")
    lines.append(f"   Summary:  {result.original_summary}")
    lines.append("")

    # Opposition
    lines.append(f"🔄 OPPOSING: {result.opposing_title}")
    lines.append(f"   Position: {result.opposing_position}")
    lines.append(f"   Summary:  {result.opposing_summary}")
    if result.opposing_sources:
        lines.append("   Sources:")
        for s in result.opposing_sources:
            lines.append(f"     • {s['title']}")
            lines.append(f"       {s['url']}")
    lines.append("")

    # Point/Counterpoint
    lines.append("─" * 70)
    lines.append("  POINT / COUNTERPOINT")
    lines.append("─" * 70)
    for i, pc in enumerate(result.point_counterpoint, 1):
        lines.append(f"\n  [{i}] {pc.get('topic', 'Topic')}")
        lines.append(f"  POINT:        {pc.get('point', '')}")
        lines.append(f"  COUNTERPOINT: {pc.get('counterpoint', '')}")
        if pc.get("key_tension"):
            lines.append(f"  TENSION:      {pc['key_tension']}")
    lines.append("")

    return "\n".join(lines)


def format_phase2(result: Phase2Result) -> str:
    """Format Phase 2 results for display."""
    lines = []
    lines.append("=" * 70)
    lines.append("  TRUTH PLATFORM — CLAIM EVIDENCE SCORES")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  Original article avg: {result.overall_original_score:.1f}/10")
    lines.append(f"  Opposing position avg: {result.overall_opposing_score:.1f}/10")
    lines.append("")

    # Score bar helper
    def score_bar(score):
        filled = int(score)
        return "█" * filled + "░" * (10 - filled) + f" {score:.1f}"

    # Original claims
    orig_claims = [c for c in result.scored_claims if c.side == "original"]
    if orig_claims:
        lines.append("─" * 70)
        lines.append("  ORIGINAL ARTICLE CLAIMS")
        lines.append("─" * 70)
        for c in sorted(orig_claims, key=lambda x: x.score, reverse=True):
            lines.append(f"\n  [{score_bar(c.score)}] [{c.confidence.upper()}]")
            lines.append(f"  Claim: {c.claim}")
            lines.append(f"  Reasoning: {c.reasoning}")
            if c.supporting_sources:
                lines.append(f"  Supporting ({len(c.supporting_sources)}):")
                for s in c.supporting_sources[:3]:
                    lines.append(f"    ✓ [{s.get('type', '?')}] {s.get('title', 'Unknown')}")
            if c.contradicting_sources:
                lines.append(f"  Contradicting ({len(c.contradicting_sources)}):")
                for s in c.contradicting_sources[:3]:
                    lines.append(f"    ✗ [{s.get('type', '?')}] {s.get('title', 'Unknown')}")

    # Opposing claims
    opp_claims = [c for c in result.scored_claims if c.side == "opposing"]
    if opp_claims:
        lines.append("")
        lines.append("─" * 70)
        lines.append("  OPPOSING POSITION CLAIMS")
        lines.append("─" * 70)
        for c in sorted(opp_claims, key=lambda x: x.score, reverse=True):
            lines.append(f"\n  [{score_bar(c.score)}] [{c.confidence.upper()}]")
            lines.append(f"  Claim: {c.claim}")
            lines.append(f"  Reasoning: {c.reasoning}")
            if c.supporting_sources:
                lines.append(f"  Supporting ({len(c.supporting_sources)}):")
                for s in c.supporting_sources[:3]:
                    lines.append(f"    ✓ [{s.get('type', '?')}] {s.get('title', 'Unknown')}")
            if c.contradicting_sources:
                lines.append(f"  Contradicting ({len(c.contradicting_sources)}):")
                for s in c.contradicting_sources[:3]:
                    lines.append(f"    ✗ [{s.get('type', '?')}] {s.get('title', 'Unknown')}")

    # Verdict
    lines.append("")
    lines.append("=" * 70)
    lines.append("  VERDICT")
    lines.append("=" * 70)
    lines.append(f"\n  {result.verdict}")
    lines.append(f"\n  Methodology: {result.methodology}")
    lines.append("")

    return "\n".join(lines)


def format_full(analysis: FullAnalysis) -> str:
    """Format full analysis for display."""
    return format_phase1(analysis.phase1) + "\n" + format_phase2(analysis.phase2)


def to_json(analysis: FullAnalysis) -> str:
    """Export full analysis as JSON."""
    return json.dumps({
        "phase1": asdict(analysis.phase1),
        "phase2": asdict(analysis.phase2),
    }, indent=2, default=str)


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Truth Platform — Article Analysis & Claim Scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python3 truth_engine.py "https://example.com/article"
              python3 truth_engine.py --text "Article text here..."
              python3 truth_engine.py --phase 1 "https://example.com/article"
              python3 truth_engine.py --json "https://example.com/article"
        """),
    )
    parser.add_argument("article", nargs="?", help="Article URL or text")
    parser.add_argument("--text", "-t", help="Article text (alternative to URL)")
    parser.add_argument("--phase", "-p", type=int, choices=[1, 2],
                        help="Run only phase 1 or 2 (default: both)")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output as JSON instead of formatted text")
    args = parser.parse_args()

    article_input = args.text or args.article
    if not article_input:
        parser.print_help()
        sys.exit(1)

    try:
        if args.phase == 1:
            result = run_phase1(article_input)
            if args.json:
                print(json.dumps(asdict(result), indent=2, default=str))
            else:
                print(format_phase1(result))
        elif args.phase == 2:
            # Phase 2 requires Phase 1 first
            phase1 = run_phase1(article_input)
            result = run_phase2(phase1)
            if args.json:
                print(json.dumps(asdict(result), indent=2, default=str))
            else:
                print(format_phase2(result))
        else:
            result = run_full_analysis(article_input)
            if args.json:
                print(to_json(result))
            else:
                print(format_full(result))
    except KeyboardInterrupt:
        print("\n\nAborted.")
        sys.exit(130)
    except Exception as e:
        print(f"\n  ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
