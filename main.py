"""
Aarav's World Trip — Simple Visual LangGraph Demo

Graph:
    home → airport ─── passport? ──► london → big_ben ─── budget? ──► END
              ▲             │                                │
              └─── forgot ──┘                                ├─► NYC   (>2L)
                                                             └─► Mumbai (≤2L)

Run:
    python main.py            # auto flow
    python main.py --pause    # press Enter between steps (live-class mode)
"""

import sys
from collections import Counter
from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, START, END

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich import box

console = Console(width=96)


# ===========================================================================
# 1. STATE
# ===========================================================================
class TripState(TypedDict):
    location:     str
    expenses:     int
    mood:         str
    visited:      Annotated[list, add]      # APPEND via reducer
    photos:       Annotated[list, add]      # APPEND via reducer
    budget:       int
    has_passport: bool
    home_visits:  int


# ===========================================================================
# 2. NODES
# ===========================================================================
def home(state: TripState):
    visits = state.get("home_visits", 0) + 1
    if visits == 1:                                    # first time — forgets passport
        return {
            "location": "Home, Delhi", "mood": "Excited",
            "home_visits": 1, "has_passport": False,
        }
    return {                                           # second time — taxi back + grab passport
        "location": "Home, Delhi", "mood": "Annoyed",
        "home_visits": visits, "has_passport": True,
        "expenses": state["expenses"] + 30_000,
        "budget":   state["budget"]   - 30_000,
        "visited":  ["Home (passport pickup)"],
    }


def airport(state: TripState):
    return {
        "location": "IGI Airport",
        "expenses": state["expenses"] + 50_000,
        "budget":   state["budget"]   - 50_000,
        "visited":  ["IGI Airport"],
        "photos":   ["S1"],
    }


def london(state: TripState):
    return {
        "location": "London Airport",
        "expenses": state["expenses"] + 1_00_000,
        "budget":   state["budget"]   - 1_00_000,
        "mood": "Tired",
        "visited": ["London Airport"],
        "photos":  ["S2"],
    }


def big_ben(state: TripState):
    return {
        "location": "Big Ben",
        "expenses": state["expenses"] + 50_000,
        "budget":   state["budget"]   - 50_000,
        "mood": "Adventurous",
        "visited": ["Big Ben"],
        "photos":  ["S3"],
    }


def new_york(state: TripState):
    return {
        "location": "Times Square, NYC",
        "expenses": state["expenses"] + 1_00_000,
        "budget":   state["budget"]   - 1_00_000,
        "mood": "Happy",
        "visited": ["NYC"],
        "photos":  ["S4"],
    }


def mumbai(state: TripState):
    return {
        "location": "Gateway of India, Mumbai",
        "expenses": state["expenses"] + 50_000,
        "budget":   state["budget"]   - 50_000,
        "mood": "Sad",
        "visited": ["Mumbai"],
    }


# ===========================================================================
# 3. CONDITIONAL EDGES
# ===========================================================================
def passport_check(state: TripState) -> Literal["london", "home"]:
    return "london" if state.get("has_passport", False) else "home"


def pick_next_city(state: TripState) -> Literal["new_york", "mumbai"]:
    return "new_york" if state["budget"] > 2_00_000 else "mumbai"


# ===========================================================================
# 4. BUILD THE GRAPH
# ===========================================================================
builder = StateGraph(TripState)
for name, fn in [
    ("home", home), ("airport", airport), ("london", london),
    ("big_ben", big_ben), ("new_york", new_york), ("mumbai", mumbai),
]:
    builder.add_node(name, fn)

builder.add_edge(START, "home")
builder.add_edge("home", "airport")
builder.add_conditional_edges("airport", passport_check,
                              {"london": "london", "home": "home"})     # 🔁
builder.add_edge("london", "big_ben")
builder.add_conditional_edges("big_ben", pick_next_city,
                              {"new_york": "new_york", "mumbai": "mumbai"})
builder.add_edge("new_york", END)
builder.add_edge("mumbai", END)

graph = builder.compile()


# ===========================================================================
#                            VISUAL HELPERS
# ===========================================================================
NODE_EMOJI = {
    "home": "🏠", "airport": "✈️ ", "london": "🛬", "big_ben": "🕰️ ",
    "new_york": "🗽", "mumbai": "🛕",
}
NODE_HEADLINE = {
    "home":     "Aarav at home",
    "airport":  "Reach IGI Airport",
    "london":   "Flight lands at London",
    "big_ben":  "Visit Big Ben",
    "new_york": "Times Square, NYC",
    "mumbai":   "Gateway of India, Mumbai",
}
# One color per node — used on BOTH the step-rule and the diary panel border.
NODE_COLOR = {
    "home":     "yellow",
    "airport":  "cyan",
    "london":   "blue",
    "big_ben":  "magenta",
    "new_york": "green",
    "mumbai":   "red",
}
MOOD_STYLE = {
    "Excited":     "bold yellow",
    "Annoyed":     "yellow",
    "Tired":       "yellow",
    "Adventurous": "bold cyan",
    "Happy":       "bold green",
    "Sad":         "bold red",
}
FIELD_ORDER = ["location", "expenses", "mood", "visited", "photos", "budget", "has_passport"]
FIELD_LABEL = {"has_passport": "passport"}


def fmt_inr(n: int) -> str:
    """Indian-style: 5,00,000 not 500,000."""
    sign, s = ("-" if n < 0 else ""), str(abs(n))
    if len(s) <= 3:
        return f"INR {sign}{s}"
    last3, rest, groups = s[-3:], s[:-3], []
    while len(rest) > 2:
        groups.append(rest[-2:]); rest = rest[:-2]
    if rest:
        groups.append(rest)
    return f"INR {sign}" + ",".join(reversed(groups)) + "," + last3


def budget_color(b: int) -> str:
    if b > 2_00_000: return "bold green"
    if b > 1_00_000: return "bold yellow"
    if b > 0:        return "bold red"
    return "bold red"


def render_value(field: str, state: dict) -> Text:
    v = state.get(field)
    if field in ("expenses", "budget"):
        return Text(fmt_inr(v), style=budget_color(v) if field == "budget" else "white")
    if field == "mood":
        return Text(v or "—", style=MOOD_STYLE.get(v, "white"))
    if field == "has_passport":
        return Text("✓ yes" if v else "✗ no", style="bold green" if v else "bold red")
    if isinstance(v, list):
        return Text(", ".join(v) if v else "—", style="white")
    return Text(v or "—", style="white")


def field_marker_note(field: str, update: dict, prev: dict) -> tuple[str, str]:
    """Returns (marker_markup, note_markup) for the diary row.
    Note column is reserved for numeric deltas only — the marker alone signals 'changed'."""
    if field not in update:
        return "  ", ""
    new, old = update[field], prev.get(field)
    if isinstance(new, list):
        return "[bold green]+[/]", ""
    if isinstance(old, int) and isinstance(new, int) and not isinstance(new, bool) and old != new:
        delta = new - old
        sign = "+" if delta > 0 else ""
        return "[bold yellow]►[/]", f"[dim]({sign}{delta:,})[/]"
    if old != new:
        return "[bold yellow]►[/]", ""
    return "  ", ""


def show_diary(title: str, state: dict, update: dict | None = None,
               prev: dict | None = None, color: str = "blue") -> None:
    update = update or {}
    prev = prev or {}

    table = Table(show_header=False, box=box.SIMPLE, pad_edge=False, expand=False, show_edge=False)
    table.add_column("m", width=2)
    table.add_column("k", style="bold cyan", width=10)
    table.add_column("v", min_width=24, max_width=50, overflow="fold")
    table.add_column("note", justify="right", width=11)

    for field in FIELD_ORDER:
        if field not in state:
            continue
        marker, note = field_marker_note(field, update, prev)
        label = FIELD_LABEL.get(field, field)
        table.add_row(marker, label, render_value(field, state), note)

    console.print(Panel(table, title=f"[bold white on {color}]  {title}  [/]",
                        border_style=color, expand=False, padding=(0, 1)))


def show_graph_tree() -> None:
    """Show the graph structure once, at the start of the demo."""
    tree = Tree("[bold cyan]🌍  Aarav's Trip Graph[/]", guide_style="dim")
    tree.add("[bold]START[/]")
    tree.add("[bold]🏠  home[/]    [dim](may forget passport on visit 1)[/]")
    tree.add("[bold]✈️   airport[/]    [dim](spend ₹50k)[/]")
    passport = tree.add("[yellow]🔍  passport_check (conditional edge)[/]")
    passport.add("[green]has_passport → continue to[/] [bold]🛬 london[/]")
    passport.add("[red]forgot → 🔁 loop back to[/] [bold]🏠 home[/]")
    tree.add("[bold]🛬  london[/]    [dim](spend ₹1L)[/]")
    tree.add("[bold]🕰️   big_ben[/]    [dim](spend ₹50k)[/]")
    budget = tree.add("[yellow]🔀  pick_next_city (conditional edge)[/]")
    budget.add("[green]budget > ₹2L  →[/] [bold]🗽 new_york[/]    [dim](spend ₹1L)[/]")
    budget.add("[red]budget ≤ ₹2L  →[/] [bold]🛕 mumbai[/]    [dim](spend ₹50k)[/]")
    tree.add("[bold]END[/]")
    console.print(Panel(tree, title="[bold cyan]The Graph You're Building[/]",
                        border_style="cyan", expand=False, padding=(0, 1)))


def step_color(node: str, visit: int) -> str:
    """Same color drives both the rule banner and the diary border."""
    return "yellow" if (node == "home" and visit > 1) else NODE_COLOR[node]


def show_step_rule(node: str, visit: int, step_num: int) -> None:
    looped = node == "home" and visit > 1
    emoji = "🔁" if looped else NODE_EMOJI[node]
    headline = "Aarav rushes home for passport!" if looped else NODE_HEADLINE[node]
    color = step_color(node, visit)
    console.line()                                           # breathing room
    console.rule(f"[bold {color}]Step {step_num}   {emoji}   {node}()   —   {headline}[/]",
                 style=color, characters="─")


def show_router(kind: str, *, took: str, budget: int | None = None) -> None:
    if kind == "passport":
        if took == "home":
            body = "has_passport = [red bold]False[/]    →    [yellow bold]🔁 loop back to HOME[/]"
            color = "yellow"
        else:
            body = "has_passport = [green bold]True[/]    →    [green bold]continue to LONDON ✈️[/]"
            color = "green"
        title = "🔍  passport_check  (conditional edge)"
    else:
        if took == "new_york":
            body = (f"budget = [bold]{fmt_inr(budget)}[/]   [dim]>[/]   "
                    f"[bold]{fmt_inr(2_00_000)}[/]    →    [green bold]🗽 NEW YORK[/]")
            color = "green"
        else:
            body = (f"budget = [bold]{fmt_inr(budget)}[/]   [dim]≤[/]   "
                    f"[bold]{fmt_inr(2_00_000)}[/]    →    [red bold]🛕 MUMBAI[/]")
            color = "red"
        title = "🔀  pick_next_city  (conditional edge)"
    console.line()
    console.print(Panel(body, title=f"[bold {color}]{title}[/]",
                        border_style=color, expand=False, padding=(0, 1)))


def show_summary(state: dict, path: list[str], counts: Counter) -> None:
    t = Table(show_header=False, box=box.SIMPLE, pad_edge=False, expand=False, show_edge=False)
    t.add_column(style="bold cyan", width=12)
    t.add_column()
    t.add_row("Path",   Text(" → ".join(path), style="white"))
    t.add_row("Loops",  Text(f"home ×{counts['home']},  airport ×{counts['airport']}", style="yellow"))
    t.add_row("Ended",  Text(state["location"], style="white"))
    t.add_row("Mood",   Text(state["mood"], style=MOOD_STYLE.get(state["mood"], "white")))
    t.add_row("Spent",  Text(fmt_inr(state["expenses"]), style="white"))
    t.add_row("Left",   Text(fmt_inr(state["budget"]),   style=budget_color(state["budget"])))
    t.add_row("Places", Text(", ".join(state["visited"]), style="magenta"))
    t.add_row("Photos", Text(", ".join(state["photos"]),  style="blue"))
    console.print(Panel(t, title="[bold green]  🎒  Trip Summary  [/]",
                        border_style="green", expand=False, padding=(0, 1)))


def apply_reducers(merged: dict, update: dict) -> None:
    """Same rule LangGraph uses internally: lists with `add` reducer append, others replace."""
    for k, v in update.items():
        if isinstance(merged.get(k), list) and isinstance(v, list):
            merged[k] = merged[k] + v
        else:
            merged[k] = v


# ===========================================================================
# 5. RUN
# ===========================================================================
if __name__ == "__main__":
    pause = "--pause" in sys.argv

    initial: TripState = {
        "location": "", "expenses": 0, "mood": "",
        "visited": [], "photos": [], "budget": 5_00_000,
        "has_passport": False, "home_visits": 0,
    }

    console.rule("[bold white on blue]  🌍  Aarav's World Trip  —  LangGraph Demo  [/]")
    show_graph_tree()
    show_diary("Initial state", initial)
    if pause: console.input("[dim]  press Enter to start the trip > [/]")

    merged: dict      = dict(initial)
    prev:   dict      = dict(initial)
    counts: Counter   = Counter()
    path:   list[str] = []
    prev_node:          str | None = None
    budget_at_big_ben:  int | None = None
    step_num = 0

    for chunk in graph.stream(initial, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                continue

            # Conditional-edge announcements (fire BEFORE the destination's diary)
            if prev_node == "airport":
                show_router("passport", took="home" if node == "home" else "london")
            if prev_node == "big_ben" and node in ("new_york", "mumbai"):
                show_router("budget", took=node, budget=budget_at_big_ben)

            step_num   += 1
            counts[node] += 1
            path.append(node)

            color = step_color(node, counts[node])
            show_step_rule(node, counts[node], step_num)
            apply_reducers(merged, update)
            label = f"diary after {node}()" + (f"   visit {counts[node]}" if counts[node] > 1 else "")
            show_diary(label, merged, update, prev, color=color)

            if node == "big_ben":
                budget_at_big_ben = merged["budget"]
            prev = dict(merged)
            prev_node = node
            if pause: console.input("[dim]  press Enter for next step > [/]")

    console.line()
    show_summary(merged, path, counts)
    console.rule("[bold green]  ✅  Trip complete  [/]", style="green", characters="═")
