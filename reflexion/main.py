"""
Step 7 — Run the graph.

    python -m reflexion.main "Why Python is the best language"
    python -m reflexion.main "Why Python is the best language" --stream
"""

import sys

from .graph import build_graph


def main():
    topic    = sys.argv[1] if len(sys.argv) > 1 else "Why Python is the best language"
    streaming = "--stream" in sys.argv

    graph = build_graph()

    initial = {
        "topic":          topic,
        "target_rating":  8,
        "max_iterations": 3,
        "iteration":      0,
        "rating":         0,
        "critique":       "",
        "history":        [],
    }

    if streaming:
        for chunk in graph.stream(initial, stream_mode="updates"):
            for node, update in chunk.items():
                print(f"\n[{node}]  ->  keys: {list(update.keys())}")
                if "rating" in update:
                    print(f"  rating: {update['rating']}/10")
                if "critique" in update:
                    print(f"  critique: {update['critique']}")
                if "image_path" in update:
                    print(f"  image:  {update['image_path']}")
    else:
        final = graph.invoke(initial)
        print(f"\nDone. {final['rating']}/10 after {final['iteration']} iteration(s)")
        print(f"Report: {final['final_report']}")


if __name__ == "__main__":
    main()
