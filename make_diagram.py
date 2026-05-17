"""Generate graph.mmd and graph.png from the compiled article-agent graph."""

from pathlib import Path

from article_agent import graph

OUT = Path(__file__).parent

# Mermaid source — always works, no network needed.
(OUT / "graph.mmd").write_text(graph.get_graph().draw_mermaid())
print("✓ graph.mmd")

# PNG via mermaid.ink — needs network.
try:
    png = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
    (OUT / "graph.png").write_bytes(png)
    print("✓ graph.png")
except Exception as e:
    print(f"✗ PNG render failed: {e}")
    print("  → paste graph.mmd into https://mermaid.live")
