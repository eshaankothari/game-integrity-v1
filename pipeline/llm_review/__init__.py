"""L8-L9: explain ONE case. Optional, not part of the scoring flow.

    packet      everything known about one player-game as JSON (~3,300 tokens)
    summarize   plain English from RULES, not a model -- cannot invent a claim.
                Free and instant; the fallback when no LLM is configured.
    review      the LLM reviewer proper (Gemini or Anthropic). It reads the packet
                and explains it; it NEVER scores or ranks.

Both readers consume the same packet, so a summary and a review of one game are
answering from identical evidence.
"""
