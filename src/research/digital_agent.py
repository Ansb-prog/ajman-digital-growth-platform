"""Deep research agent: estimates a business's digital presence from the public web.

Uses the Claude API with the server-side web search tool. Only public, business-level
information is collected; no personal data about owners or staff. Results are cached
in data/cache/research/ so repeated demo queries cost nothing.

Requires:  export ANTHROPIC_API_KEY=...
Optional:  RESEARCH_MODEL (default claude-sonnet-5-5), WEB_SEARCH_TOOL (default web_search_20260318)

Run:  python -m src.research.digital_agent "Business Name" --area "Al Nuaimiya"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone

from src.data.load import ROOT

CACHE = ROOT / "data" / "cache" / "research"
MODEL = os.getenv("RESEARCH_MODEL", "claude-sonnet-5-5")
WEB_SEARCH_TOOL = os.getenv("WEB_SEARCH_TOOL", "web_search_20260318")

SYSTEM = """You are a market research analyst assessing the online presence and marketing activity of a
business in Ajman, UAE. Use web search to find its website, Google Maps/Business listing, Instagram,
TikTok, Facebook, LinkedIn and WhatsApp Business presence. Rules:
- Only collect public, business-level information. Never record personal data about owners, staff or customers.
- Never guess numbers. If you cannot verify a value, use null.
- Make sure the business you find is in Ajman (or clearly serves Ajman); if unsure, say so in `match_confidence`.
Finish with ONE JSON object inside <json></json> tags, with exactly these keys:
{
  "business_found": bool,
  "match_confidence": "high" | "medium" | "low",
  "website_url": str | null,
  "has_website": bool | null,
  "google_rating": float | null,
  "google_review_count": int | null,
  "instagram_handle": str | null,
  "instagram_followers": int | null,
  "posts_per_month_estimate": float | null,
  "days_since_last_post": int | null,
  "has_tiktok": bool | null,
  "has_whatsapp_business": bool | null,
  "responds_to_reviews": bool | null,
  "marketing_observations": [str],
  "sources": [str]
}"""


def _cache_path(name: str, area: str | None):
    key = hashlib.sha1(f"{name.lower().strip()}|{(area or '').lower().strip()}".encode()).hexdigest()[:16]
    return CACHE / f"{key}.json"


def parse_report(text: str) -> dict:
    """Extract the JSON object from the model's final text."""
    m = re.search(r"<json>(.*?)</json>", text, re.S)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(raw)


def to_signals(report: dict) -> dict:
    """Map an agent report to the model's digital-presence inputs (None = unknown -> default)."""
    return {
        "has_website": report.get("has_website"),
        "google_rating": report.get("google_rating"),
        "google_review_count": report.get("google_review_count"),
        "instagram_followers": report.get("instagram_followers"),
        "posts_per_month": report.get("posts_per_month_estimate"),
        "days_since_last_post": report.get("days_since_last_post"),
        "engagement_rate": None,  # not reliably observable from the public web
    }


def research_business(name: str, area: str | None = None, use_cache: bool = True, max_searches: int = 8) -> dict:
    """Research one business. Returns the parsed report plus metadata. Raises if no API key."""
    path = _cache_path(name, area)
    if use_cache and path.exists():
        return {**json.loads(path.read_text()), "cached": True}
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY to use the research agent.")

    import anthropic

    client = anthropic.Anthropic()
    where = f" in {area}, Ajman, UAE" if area else " in Ajman, UAE"
    messages = [{"role": "user", "content": f"Research the online presence and marketing activity of "
                                            f"'{name}'{where}. Today is {datetime.now(timezone.utc).date()}."}]
    tools = [{"type": WEB_SEARCH_TOOL, "name": "web_search", "max_uses": max_searches}]

    resp = client.messages.create(model=MODEL, max_tokens=4000, system=SYSTEM, messages=messages, tools=tools)
    # Server tools can pause long turns; continue until the model finishes.
    for _ in range(3):
        if resp.stop_reason != "pause_turn":
            break
        messages += [{"role": "assistant", "content": resp.content}]
        resp = client.messages.create(model=MODEL, max_tokens=4000, system=SYSTEM, messages=messages, tools=tools)

    text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
    report = parse_report(text)
    out = {"query": {"name": name, "area": area}, "report": report,
           "researched_at": datetime.now(timezone.utc).isoformat(), "model": MODEL}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return {**out, "cached": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--area")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    print(json.dumps(research_business(a.name, a.area, use_cache=not a.no_cache), indent=2))


if __name__ == "__main__":
    main()
