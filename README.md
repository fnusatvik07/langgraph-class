# LangGraph Class — DataSense

A teaching repository for a live LangGraph class.  Four progressive examples
that build up from a pure-Python graph (no LLM) to a multi-modal agent
with web search and image generation.

The goal: make state, nodes, edges, conditional edges, and loops feel
concrete — students can read every example end-to-end in one sitting.

---

## What's inside

| File / folder | What it teaches |
|---|---|
| `main.py` | **Aarav's World Trip** — a deterministic LangGraph (no LLM).  Teaches state, nodes, linear edges, conditional edges, reducers.  Rich terminal output with a per-step diary panel. |
| `chains_demo.py` | **LangChain chain basics** (5 demos).  `prompt \| llm`, `StrOutputParser`, `MessagesPlaceholder`, structured Pydantic output, streaming. |
| `chains.py` | The writer + critic chains used by the article agent.  Triple-quoted prompts with a clear rubric for the critic. |
| `article_agent.py` | **Article Reflexion Agent** — topic in → Tavily search → write → critic → loop until rating ≥ 8 → save markdown.  Single file, ~200 lines. |
| `make_diagram.py` | Generate `graph.mmd` + `graph.png` from the article agent. |
| `article_agent.drawio` | Detailed teaching diagram showing how state updates at every node, openable in [draw.io](https://app.diagrams.net) or the VS Code Draw.io extension. |
| `reflexion/` | **Thumbnail Reflexion Agent** (assignment 1).  Multi-file project that extends the same loop pattern with DALL-E 3 image generation + GPT-4o vision critic.  Costs ~$0.24/run. |

---

## Setup

```bash
# 1. Install Python deps
uv sync                     # uses pyproject.toml + uv.lock
# or:  pip install -e .

# 2. Add your API keys to .env (gitignored)
echo "OPENAI_API_KEY=sk-..."         >  .env
echo "TAVILY_API_KEY=tvly-..."       >> .env
```

Requires Python 3.11+.

---

## Run the examples

### 1. Aarav's World Trip (no LLM — pure LangGraph mechanics)

```bash
python main.py                # auto flow
python main.py --pause        # press Enter between steps (live-class mode)
```

You'll see a colored "Aarav's Diary" panel after every node, the
journey map with the current node highlighted, conditional-edge panels
for the passport loop and budget fork, and a final trip summary.

### 2. LangChain chain demos

```bash
python chains_demo.py 1       # basic chain
python chains_demo.py 2       # output parser
python chains_demo.py 3       # MessagesPlaceholder  ← used by writer_chain
python chains_demo.py 4       # structured output    ← used by critic_chain
python chains_demo.py 5       # streaming
```

### 3. Article Reflexion Agent

```bash
python article_agent.py                       # default topic
python make_diagram.py                        # save graph.png
```

Edit the topic at the bottom of `article_agent.py` (in `__main__`).
Final article is saved to `outputs/article.md`.

Uncomment the blocks at the bottom of `article_agent.py` to try:
- Printing the graph as Mermaid text
- Streaming every node update as it happens

### 4. Thumbnail Reflexion Agent (assignment)

```bash
python -m reflexion.make_diagram                       # save the graph PNG
python -m reflexion.main "Why Python is the best"      # invoke once
python -m reflexion.main "Why Python is the best" --stream
```

This is the same reflexion loop as the article agent, extended with:
- DALL-E 3 image generation node
- GPT-4o vision critic that reads the actual generated image
- Multi-file Python package layout

Run cost ~$0.24 (up to 3 × DALL-E images @ $0.08).

---

## The graph (article agent)

```
START → search → writer → critic ──┐ (rating < 8: loop to writer)
                  ▲                 │
                  └─────────────────┘
                                    │ (rating ≥ 8)
                                    ▼
                                  save → END
```

See `article_agent.drawio` for the full annotated version with state-after
panels at each node.

---

## License

MIT.
