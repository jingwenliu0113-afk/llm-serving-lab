"""capserve: a capacity-aware gateway in front of the vLLM 0.30.0 OpenAI server (decisions 027 §3-§5, 029).

It writes no engine or scheduler and never changes vLLM. Research modules (benchmarks/) are imported read-only; the
research harness never imports capserve (tests/test_research_isolation.py). Limits come from a capacity profile:
only a `verified` profile (generated from a frozen, formal Stage C analysis) is a product limit; `provisional` ones are
for development and integration tests only and are refused unless explicitly allowed (027 §4).
"""
