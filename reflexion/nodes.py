"""Step 4 — Nodes.  Each function takes state, returns a partial update."""

import base64
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .state import ThumbnailState
from .prompts import PROMPT_WRITER_SYSTEM, PROMPT_WRITER_USER, REVISION_HINT, CRITIC_SYSTEM
from .tools import web_search


# --- Clients (env vars loaded by reflexion/__init__.py) ---------------------
oa     = OpenAI()
writer = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
critic = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

OUTPUTS = Path(__file__).parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)


class Critique(BaseModel):
    rating:   int = Field(ge=1, le=10)
    critique: str


# --- NODE 1 -----------------------------------------------------------------
def node_web_search(state: ThumbnailState) -> dict:
    summary = web_search(f"YouTube thumbnail ideas for: {state['topic']}")
    safe = "".join(c if c.isalnum() else "_" for c in state["topic"])[:40]
    run_dir = OUTPUTS / f"{datetime.now():%Y%m%d_%H%M%S}_{safe}"
    run_dir.mkdir()
    return {"search_summary": summary, "iteration": 0, "run_dir": str(run_dir)}


# --- NODE 2 -----------------------------------------------------------------
def node_prompt_writer(state: ThumbnailState) -> dict:
    feedback = ""
    if state.get("critique"):
        feedback = REVISION_HINT.format(rating=state["rating"], critique=state["critique"])
    user = PROMPT_WRITER_USER.format(
        topic=state["topic"],
        search_summary=state["search_summary"],
        feedback=feedback,
    )
    resp = writer.invoke([SystemMessage(content=PROMPT_WRITER_SYSTEM),
                          HumanMessage(content=user)])
    return {"current_prompt": resp.content.strip()}


# --- NODE 3 -----------------------------------------------------------------
def node_generator(state: ThumbnailState) -> dict:
    n = state["iteration"] + 1
    resp = oa.images.generate(
        model="dall-e-3",
        prompt=state["current_prompt"],
        size="1792x1024",
        n=1,
        response_format="b64_json",
    )
    path = Path(state["run_dir"]) / f"iter_{n}.png"
    path.write_bytes(base64.b64decode(resp.data[0].b64_json))
    return {"image_path": str(path), "iteration": n}


# --- NODE 4 -----------------------------------------------------------------
def node_critic(state: ThumbnailState) -> dict:
    img_b64 = base64.b64encode(Path(state["image_path"]).read_bytes()).decode()
    result = critic.with_structured_output(Critique).invoke([
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=[
            {"type": "text", "text": f"Topic: {state['topic']}. Rate this thumbnail."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]),
    ])
    return {
        "rating":   result.rating,
        "critique": result.critique,
        "history":  [{
            "iteration":  state["iteration"],
            "prompt":     state["current_prompt"],
            "image_path": state["image_path"],
            "rating":     result.rating,
            "critique":   result.critique,
        }],
    }


# --- CONDITIONAL EDGE -------------------------------------------------------
def should_continue(state: ThumbnailState):
    if state["rating"] >= state["target_rating"]:
        return "saver"
    if state["iteration"] >= state["max_iterations"]:
        return "saver"
    return "prompt_writer"


# --- NODE 5 -----------------------------------------------------------------
def node_saver(state: ThumbnailState) -> dict:
    best = max(state["history"], key=lambda h: h["rating"])
    run_dir = Path(state["run_dir"])

    lines = [
        f"# Thumbnail — {state['topic']}",
        f"**Final rating**: {best['rating']}/10",
        f"**Iterations**: {state['iteration']}",
        "",
        "## Research", state["search_summary"], "",
    ]
    for h in state["history"]:
        lines += [
            f"## Iteration {h['iteration']}  ({h['rating']}/10)",
            f"**Prompt**: {h['prompt']}",
            f"**Critique**: {h['critique']}",
            f"![iter_{h['iteration']}]({Path(h['image_path']).name})",
            "",
        ]

    (run_dir / "final.png").write_bytes(Path(best["image_path"]).read_bytes())
    lines += ["## Final", "![final](./final.png)"]

    report = run_dir / "report.md"
    report.write_text("\n".join(lines))
    return {"final_report": str(report)}
