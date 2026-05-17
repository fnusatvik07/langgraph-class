"""
============================================================================
 LangChain Chains — Quick Reference
============================================================================

A "chain" in LangChain is just three things piped together:

        prompt   |   llm   |   (optional) output parser

This file walks through 5 minimal chain patterns.
Run one demo at a time:

    python chains_demo.py 1     # the simplest chain
    python chains_demo.py 2     # add an output parser
    python chains_demo.py 3     # MessagesPlaceholder (conversation history)
    python chains_demo.py 4     # structured output (Pydantic)
    python chains_demo.py 5     # streaming tokens

Demos 3 and 4 are EXACTLY the patterns used in chains.py for the
article reflexion agent  →  teach these first, then the agent makes sense.
"""

import sys

from dotenv import load_dotenv
load_dotenv()                                       # OPENAI_API_KEY

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# Shared LLM — used in every demo below.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)


# ============================================================================
# Demo 1 — the simplest chain:   prompt  |  llm
# ============================================================================
def demo_1_basic():
    """
    A prompt with ONE variable, piped into an LLM.
    `.invoke()` takes a dict matching the prompt's variables.
    Output is an AIMessage object (not a plain string yet).
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a witty assistant."),
        ("human",  "Tell me a one-line joke about {topic}."),
    ])

    chain = prompt | llm                            # ← that pipe IS the chain

    response = chain.invoke({"topic": "Python"})

    print("type:    ", type(response).__name__)    # AIMessage
    print("content: ", response.content)


# ============================================================================
# Demo 2 — add an output parser:   prompt  |  llm  |  StrOutputParser()
# ============================================================================
def demo_2_with_parser():
    """
    `StrOutputParser()` pulls `.content` out of the AIMessage so the chain
    returns a plain string.  Useful when you don't need the message object.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You explain things in 1 sentence."),
        ("human",  "What is {thing}?"),
    ])

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"thing": "LangGraph"})
    print("answer (plain str):", answer)


# ============================================================================
# Demo 3 — MessagesPlaceholder:  inject a list of prior messages dynamically
# ============================================================================
def demo_3_messages_placeholder():
    """
    `MessagesPlaceholder("history")` is a slot in the prompt where you pass
    a LIST of messages at invoke time.  This is how the article writer sees
    the running conversation (drafts + critiques) and revises accordingly.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly Python tutor."),
        MessagesPlaceholder(variable_name="history"),
    ])

    chain = prompt | llm

    history = [
        HumanMessage(content="What's a list comprehension?"),
        AIMessage(content="A compact way to build lists, e.g. [x*2 for x in nums]."),
        HumanMessage(content="Show one for squaring even numbers from 0 to 10."),
    ]

    response = chain.invoke({"history": history})
    print(response.content)


# ============================================================================
# Demo 4 — Structured output via Pydantic
# ============================================================================
class MovieReview(BaseModel):
    """The shape the LLM must return — like a TypedDict for the model."""
    title:  str = Field(description="movie title")
    rating: int = Field(ge=1, le=10, description="1-10 score")
    review: str = Field(description="1-2 sentence review")


def demo_4_structured_output():
    """
    `llm.with_structured_output(Schema)` makes the model return a Pydantic
    object you can use like a regular Python object — no JSON parsing,
    no missing fields, validated types.  This is how the critic gets
    a guaranteed integer `rating` in the reflexion agent.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a film critic."),
        ("human",  "Review the movie: {movie}"),
    ])

    chain = prompt | llm.with_structured_output(MovieReview)

    review: MovieReview = chain.invoke({"movie": "Inception"})

    print("type:    ", type(review).__name__)      # MovieReview
    print("title:   ", review.title)
    print("rating:  ", review.rating, "/10")
    print("review:  ", review.review)


# ============================================================================
# Demo 5 — Streaming tokens as the model generates them
# ============================================================================
def demo_5_streaming():
    """
    `.stream()` yields chunks WHILE the model is still generating, instead
    of waiting for the full response.  Great for UIs that show text typing
    out in real time.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("human", "Write a haiku about {topic}."),
    ])

    chain = prompt | llm | StrOutputParser()

    for chunk in chain.stream({"topic": "rainy days"}):
        print(chunk, end="", flush=True)
    print()


# ============================================================================
# Entry point — pick a demo from the CLI
# ============================================================================
DEMOS = {
    "1": ("basic chain",            demo_1_basic),
    "2": ("with output parser",     demo_2_with_parser),
    "3": ("MessagesPlaceholder",    demo_3_messages_placeholder),
    "4": ("structured output",      demo_4_structured_output),
    "5": ("streaming",              demo_5_streaming),
}


if __name__ == "__main__":
    pick = sys.argv[1] if len(sys.argv) > 1 else None
    if pick in DEMOS:
        name, fn = DEMOS[pick]
        print(f"\n=== Demo {pick}: {name} ===\n")
        fn()
    else:
        print(__doc__)
        print(f"Pick a demo:  {', '.join(DEMOS)}")
