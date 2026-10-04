import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import build_graph

app = build_graph()

try:
    png = app.get_graph().draw_mermaid_png()
    out = Path("docs/graph.png")
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(png)
    print(f"Graph diagram saved to {out}")
except Exception as e:
    print(f"Could not save graph diagram: {e}")
    print(app.get_graph().draw_mermaid())