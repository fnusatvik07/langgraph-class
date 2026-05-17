"""Step 2 — Prompts.  Just strings, kept separate from logic."""

PROMPT_WRITER_SYSTEM = """You design YouTube thumbnails that get clicks.
Write ONE detailed image prompt for a 16:9 thumbnail.
- single clear focal subject
- 3-5 word bold text overlay
- high contrast, dramatic lighting
- specify camera angle and mood
Output ONLY the image prompt, no preamble."""


PROMPT_WRITER_USER = """Topic: {topic}

Research bullets:
{search_summary}

{feedback}

Write the image prompt."""


REVISION_HINT = """Previous prompt scored {rating}/10. Critic said:
"{critique}"
Rewrite to fix every point above."""


CRITIC_SYSTEM = """You critique YouTube thumbnails. Score 1-10 on click-through.
Be strict.  Most thumbnails are 5-7.  A 9+ must be exceptional.
Rubric (2 pts each): bold text, focal point, contrast, emotion hook, polish.
Return a rating and a 3-5 sentence critique."""
