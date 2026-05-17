"""
Article Reflexion Agent  —  topic in, polished article out.

Graph:    START → search → writer → critic → ┐ (rating < 8 → loop to writer)
                              ▲               │
                              └───────────────┘
                                              (rating >= 8) → save → END
"""

from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
load_dotenv()                              # OPENAI_API_KEY + TAVILY_API_KEY

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from tavily import TavilyClient

from chains import writer_chain, critic_chain


# ============================================================================
# Step 1 — STATE
# ----------------------------------------------------------------------------
# Shared state passed between every node.  TypedDict keys are merged by
# LangGraph using each field's reducer:
#     - `messages` has `add_messages`   → values are APPENDED
#     - every other field has no reducer → values are OVERWRITTEN
#
# (TypedDict has no native field-description like Pydantic's Field(), so we
# document each field with an inline comment.  LangGraph supports both —
# switch to a `class State(BaseModel)` with `Field(description=...)` if you
# want descriptions attached to the field itself.)
# ============================================================================
class State(TypedDict, total=False):
    # total=False — every field is OPTIONAL.  This lets a node return
    # only the keys it wants to update (a "partial state") and still
    # satisfy the `-> State` return type.

    topic:     str
    # ^ the article topic the user typed in, e.g. "Why Python is great"

    messages:  Annotated[list[BaseMessage], add_messages]
    # ^ running conversation: seed → draft → critique → draft → … .
    #   The `add_messages` reducer APPENDS new messages — that's what
    #   lets the writer see all the prior critiques.

    rating:    int
    # ^ latest critic score 1-10.  The conditional edge reads this.

    iteration: int
    # ^ how many drafts written so far.  Safety cap for the loop.


# Loop terminates when EITHER condition is met:
TARGET_RATING  = 8       # critic gave at least this score
MAX_ITERATIONS = 3       # we wrote this many drafts


# ============================================================================
# Step 2 — NODES
# ----------------------------------------------------------------------------
# IMPORTANT — the return pattern:
#
#   Every node receives the FULL state and returns a PARTIAL dict —
#   ONLY the keys you want to change.  LangGraph merges that back using
#   each field's reducer:
#
#       messages  has add_messages reducer  →  values are APPENDED
#       all other fields have no reducer    →  values are OVERWRITTEN
#
#   So when a node writes a new draft, you return:
#         {"messages": [new_draft]}                    # appended
#
#   NOT:  {"messages": state["messages"] + [new_draft]}   # don't do this
# ============================================================================

def search_node(state: State) -> State:
    """One-time Tavily search.  Research goes into the seed HumanMessage."""

    # 1. Run the web search — Tavily returns a dict; we want the "results" list.
    hits = TavilyClient().search(state["topic"], max_results=5)["results"]

    # 2. Format each hit as one bullet line: "- title: first 200 chars of content".
    research_lines = []
    for hit in hits:
        title   = hit["title"]
        snippet = hit["content"][:200]
        research_lines.append(f"- {title}: {snippet}")
    research = "\n".join(research_lines)

    # 3. Build the seed message that kicks off the conversation.
    seed = HumanMessage(content=(
        f"Write a markdown article on: {state['topic']}\n\n"
        f"Use this research:\n{research}"
    ))

    return {
        "messages":  [seed],     # APPENDED  (add_messages reducer)
        "rating":    0,           # overwrite — start fresh
        "iteration": 0,           # overwrite — start fresh
    }


def writer_node(state: State) -> State:
    """Generate (or revise) the article from the current conversation."""

    # 1. Run the writer chain on the running conversation.
    #    On iteration 1 → it sees [seed] and writes a fresh article.
    #    On iteration 2+ → it sees [seed, draft1, critique1, …] and rewrites
    #                      addressing the critique.
    draft = writer_chain.invoke({"messages": state["messages"]})

    # 2. Append the new draft and bump the iteration counter.
    return {
        "messages":  [draft],                           # APPENDED
        "iteration": state["iteration"] + 1,            # overwritten
    }


def critic_node(state: State) -> State:
    """Critique the latest draft.  Score gates the loop; critique trains the next draft."""

    # 1. Run the critic chain — returns a Pydantic Critique(rating, critique).
    result = critic_chain.invoke({"messages": state["messages"]})

    # 2. Wrap the critique as a HumanMessage so that on the next writer loop
    #    it looks like a user request ("rewrite addressing this feedback").
    feedback = HumanMessage(content=(
        f"Editor score: {result.rating}/10\n"
        f"Feedback: {result.critique}\n\n"
        f"Rewrite the article addressing every point above."
    ))

    # 3. Append the feedback message and update the latest rating.
    return {
        "messages": [feedback],         # APPENDED
        "rating":   result.rating,      # overwritten
    }


def should_continue(state: State) -> Literal["writer", "save"]:
    """Conditional edge — decide whether to loop again or finish.

    Routers return a STRING that names the next node (not a partial state).
    LangGraph reads this and routes the graph accordingly.
    """

    # 1. The critic was happy enough — go save and end.
    if state["rating"] >= TARGET_RATING:
        return "save"

    # 2. Safety cap — we've tried enough times.
    if state["iteration"] >= MAX_ITERATIONS:
        return "save"

    # 3. Otherwise loop back to the writer with the critique in messages.
    return "writer"


def save_node(state: State) -> State:
    """Side-effect node — write the latest article to outputs/article.md."""

    # 1. Find the most recent AIMessage in the conversation —
    #    that's the latest article draft.
    article = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            article = msg.content
            break

    # 2. Make sure outputs/ exists and build the file path.
    Path("outputs").mkdir(exist_ok=True)
    path = Path("outputs") / "article.md"

    # 3. Write the markdown file: header + meta line + the article body.
    path.write_text(
        f"# {state['topic']}\n\n"
        f"_rating {state['rating']}/10 — {state['iteration']} iteration(s)_\n\n"
        f"{article}\n"
    )
    print(f"saved → {path}")

    # 4. Return an empty dict — save changes the FILESYSTEM, not the state.
    return {}


# ============================================================================
# Step 3 — BUILD THE GRAPH
# ============================================================================
builder = StateGraph(State)

builder.add_node("search", search_node)
builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)
builder.add_node("save",   save_node)

builder.add_edge(START,    "search")
builder.add_edge("search", "writer")
builder.add_edge("writer", "critic")
builder.add_conditional_edges("critic", should_continue,
                              {"writer": "writer", "save": "save"})
builder.add_edge("save", END)

graph = builder.compile()


# ============================================================================
# Step 4 — RUN
# ============================================================================
if __name__ == "__main__":
    initial: State = {
        "topic":     "Why Python is the best language for AI",
        "messages":  [],
        "rating":    0,
        "iteration": 0,
    }

    final = graph.invoke(initial)

    print(f"\nFinal rating: {final['rating']}/10")
    print(f"Iterations:   {final['iteration']}")


# ============================================================================
# Want to try more?  Uncomment any of these:
# ============================================================================
#
# # 1. Print the graph topology as Mermaid text
# print(graph.get_graph().draw_mermaid())
#
# # 2. Save the graph as a PNG (or just run:  python make_diagram.py)
# Path("graph.png").write_bytes(
#     graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
# )
#
# # 3. Stream every node update as it happens (great for watching the loop)
# for chunk in graph.stream(initial, stream_mode="updates"):
#     for node, update in chunk.items():
#         print(f"[{node}] -> {list(update.keys())}")
