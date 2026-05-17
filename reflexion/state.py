"""
Step 1 — State.  Shared by every node.

Two things to notice:

1.  total=False
    ───────────
    TypedDict normally requires EVERY key on every dict.  total=False makes
    each key optional, so the initial state we pass to graph.invoke() only
    needs to provide what we have on day one (topic, target_rating, etc.) —
    the rest of the fields appear as nodes run and write them.

2.  Annotated[list, add]   (only on `history`)
    ──────────────────────────────────────────
    By default, when a node returns {key: value}, LangGraph OVERWRITES the
    old value.  That's what we want for `current_prompt`, `rating`,
    `image_path` etc.  — each iteration only the LATEST value matters.

    But `history` is different.  Every critic step appends ONE row to it,
    and we want the rows from older iterations to stick around so we can
    write the final report.  `Annotated[list, add]` tells LangGraph to
    APPEND (using operator.add) instead of replace.
"""

from operator import add
from typing import Annotated, TypedDict


class ThumbnailState(TypedDict, total=False):
    # ---- Input (you pass these to graph.invoke) -----------------------
    topic:           str   # the YouTube video title / subject
    target_rating:   int   # stop the loop once critic gives at least this score (1-10)
    max_iterations:  int   # hard safety cap — stop after this many generator runs

    # ---- Written once by  node_web_search  ----------------------------
    search_summary:  str   # Tavily bullet-point research, fed into the prompt
    run_dir:         str   # folder under outputs/ where this run's files go

    # ---- Overwritten on EVERY loop iteration --------------------------
    current_prompt:  str   # latest DALL·E image prompt
    image_path:      str   # latest generated PNG on disk
    rating:          int   # latest critic score 1-10
    critique:        str   # latest critic feedback (drives the next rewrite)
    iteration:       int   # how many generator runs so far (1, 2, 3, …)

    # ---- Written once by  node_saver  ---------------------------------
    final_report:    str   # path to the final report.md

    # ---- Append-only log (one row per critic step) --------------------
    # The reducer (operator.add) appends instead of overwriting, so we can
    # walk every iteration's prompt + image + score in the final report.
    history:         Annotated[list, add]
