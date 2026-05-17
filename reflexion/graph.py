"""Step 5 — Build the graph."""

from langgraph.graph import StateGraph, START, END

from .state import ThumbnailState
from .nodes import (
    node_web_search, node_prompt_writer, node_generator,
    node_critic, node_saver, should_continue,
)


def build_graph():
    g = StateGraph(ThumbnailState)

    g.add_node("web_search",    node_web_search)
    g.add_node("prompt_writer", node_prompt_writer)
    g.add_node("generator",     node_generator)
    g.add_node("critic",        node_critic)
    g.add_node("saver",         node_saver)

    g.add_edge(START,           "web_search")
    g.add_edge("web_search",    "prompt_writer")
    g.add_edge("prompt_writer", "generator")
    g.add_edge("generator",     "critic")

    g.add_conditional_edges("critic", should_continue, {
        "prompt_writer": "prompt_writer",     # LOOP — score too low
        "saver":         "saver",             # rating reached / out of iterations
    })

    g.add_edge("saver", END)

    return g.compile()
