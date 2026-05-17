"""Step 6 — Generate graph.mmd + graph.png."""

from pathlib import Path

from .graph import build_graph


graph = build_graph()
out   = Path(__file__).parent

(out / "graph.mmd").write_text(graph.get_graph().draw_mermaid())
print("✓ graph.mmd")

try:
    png = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
    (out / "graph.png").write_bytes(png)
    print("✓ graph.png")
except Exception as e:
    print(f"✗ PNG render failed: {e}")
    print("  → paste graph.mmd into https://mermaid.live")
