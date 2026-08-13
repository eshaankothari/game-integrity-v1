"""game-integrity: NBA player-prop screening for the 2023-24 season.

Layers, in the order data moves through them. Each reads only what the layer
above it wrote, and each is idempotent -- re-running fills gaps, never duplicates.

    pipeline.core        config, database access, the network cache
    pipeline.load_data   L0-L4  the loaders: dimensions, market, box scores, play-by-play
    pipeline.score       L5-L6  standardize -> cut -> rank
    pipeline.llm_review  L8-L9  evidence packets, rule-based summaries, optional LLM review
    pipeline.tools       handoff utilities (Postgres -> DuckDB export)

Run any of them as a module, from the repository root:

    python -m pipeline.load_data.load_props        # DRY: reports, writes nothing
    python -m pipeline.load_data.load_props run    # actually executes
    python -m pipeline.score.standardize run

See ARCHITECTURE.md for what every file and key function does.
"""
