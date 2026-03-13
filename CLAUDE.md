# Truth Platform

You are the Truth Platform — an article analysis and claim-scoring engine.

## What This Does
- **Phase 1**: User sends an article → extract claims → find the most diametrically opposed content → return point/counterpoint summary
- **Phase 2**: Score each claim from both sides (0-10) based on number and strength of confirmatory/contradictory sources found via web search

## Architecture
- `truth_engine.py` — single-file engine with both phases, CLI interface, and formatting
- Uses Claude API (Sonnet) for analysis, claim extraction, query generation, scoring
- Uses DuckDuckGo for web search (free, no API key)
- Uses Trafilatura for article extraction from URLs

## How to Run
```bash
cd ~/Documents/AGENTS/Truth-Platform
python3 truth_engine.py "https://example.com/article"          # Full analysis
python3 truth_engine.py --phase 1 "https://example.com/article" # Phase 1 only
python3 truth_engine.py --json "https://example.com/article"     # JSON output
python3 truth_engine.py --text "article text here..."            # Direct text input
```

## Phase 1 Pipeline
1. Extract article content (URL or text)
2. Claude analyzes: title, position, bias direction, key claims, opposition angle
3. Claude generates search queries for diametrically opposed content
4. DuckDuckGo search for opposing articles
5. Claude ranks results by opposition strength, fetches top 2-3
6. Claude synthesizes point/counterpoint summary

## Phase 2 Pipeline
1. Combine all claims from both sides
2. For each claim: generate support + contradict search queries
3. Search web for evidence
4. Claude scores each claim 0-10 based on evidence (academic > news > blog weighting)
5. Generate overall verdict with strongest/weakest claims per side

## Scoring Methodology
- Base: 5.0
- Academic/peer-reviewed source: ±1.5
- Major news outlet: ±1.0
- Blog/opinion: ±0.5
- Range: 0.0 to 10.0

## Dependencies
- anthropic, httpx, duckduckgo-search, trafilatura
- API key loaded via load_keys.py (Anthropic)
