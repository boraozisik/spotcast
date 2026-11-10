"""The same agent as agent.py, built as a LangGraph graph.

    START → profile ──refused──▶ END
               │
               ▼
             agent ──tool calls──▶ tools
               │  ▲                 │
               │  └─────────────────┘
               ▼  no tool calls (or too many steps)
             finish → END

agent.py is a `for` loop we wrote ourselves. Here the same steps are NODES and the
decisions between them are EDGES. What the graph adds for free:
- a checkpointer: the conversation is remembered, so "peki pazar?" works as a follow-up,
- streaming: every node's output can be watched as it happens,
- a picture of the flow (`--diagram`).

Run: uv run spotcast-graph
"""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from spotcast.agent import (
    GAVE_UP,
    MAX_STEPS,
    SYSTEM_PROMPT,
    Step,
    UserProfile,
    check_call,
    date_table,
    decide_profile,
    finish_answer,
    read_profile,
    run_tool,
    schema_for_llm,
)
from spotcast.llm import CHAT_MODEL
from spotcast.tools import TOOLS


class AgentState(TypedDict, total=False):
    """Everything the graph remembers. Each node returns only the keys it changes."""

    messages: Annotated[list[BaseMessage], add_messages]  # new messages are APPENDED
    question: str
    sport: str
    level: str
    level_note: str
    steps: list[
        dict
    ]  # this turn's tool calls, as plain dicts the checkpointer can save
    answer: str


def build_graph(
    llm: Any | None = None,
    profile_reader=read_profile,
    checkpointer=None,
    clock: Callable[[], datetime] = datetime.now,
):
    """Build and compile the graph. `llm`, `profile_reader` and `clock` can be fakes in tests."""
    llm = llm or ChatOllama(model=CHAT_MODEL, temperature=0.2, reasoning=False)
    llm_with_tools = llm.bind_tools([schema_for_llm(tool) for tool in TOOLS])

    def profile(state: AgentState) -> dict:
        said = profile_reader(state["question"])
        # In a follow-up ("peki pazar?") nothing new is said: keep the earlier profile
        if said.sport == "unknown" and "sport" in state:
            said = UserProfile(sport=state["sport"], level=said.level)
        keep_level = said.level == "unknown" and "level" in state
        if keep_level:
            said = UserProfile(sport=said.sport, level=state["level"])
        who = decide_profile(said)
        if who.refusal:
            return {"answer": who.refusal}
        # Still unknown if the user never said it: keep the earlier note, too
        level_note = state.get("level_note", "") if keep_level else who.level_note

        new_messages: list[BaseMessage] = []
        if not state.get("messages"):  # first turn: the rules and today's dates
            prompt = SYSTEM_PROMPT.format(date_table=date_table(clock()))
            new_messages.append(SystemMessage(prompt))
        new_messages.append(HumanMessage(state["question"]))
        return {
            "messages": new_messages,
            "sport": who.sport,
            "level": who.level,
            "level_note": level_note,
            "steps": [],
            "answer": "",
        }

    def after_profile(state: AgentState) -> Literal["agent", "__end__"]:
        return END if state.get("answer") else "agent"

    def agent(state: AgentState) -> dict:
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def after_agent(state: AgentState) -> Literal["tools", "finish"]:
        last = state["messages"][-1]
        wants_tools = isinstance(last, AIMessage) and last.tool_calls
        return "tools" if wants_tools and len(state["steps"]) < MAX_STEPS else "finish"

    def tools(state: AgentState) -> dict:
        steps, replies = list(state["steps"]), []
        # Results of every earlier tool call, this turn and earlier ones ("peki pazar?")
        earlier = [
            json.loads(m.content)
            for m in state["messages"]
            if isinstance(m, ToolMessage)
        ]
        for call in state["messages"][-1].tool_calls:
            started = time.monotonic()
            result = check_call(
                call["name"], call["args"], state["question"], clock(), earlier
            ) or run_tool(call["name"], call["args"], state["sport"], state["level"])
            step = Step(call["name"], call["args"], result, time.monotonic() - started)
            steps.append(asdict(step))
            replies.append(
                ToolMessage(
                    json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )
        return {"messages": replies, "steps": steps}

    def finish(state: AgentState) -> dict:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:  # stopped by the step limit
            return {"answer": GAVE_UP}
        steps = [Step(**step) for step in state["steps"]]
        return {"answer": finish_answer(str(last.content), steps, state["level_note"])}

    graph = StateGraph(AgentState)
    graph.add_node("profile", profile)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("finish", finish)
    graph.add_edge(START, "profile")
    graph.add_conditional_edges("profile", after_profile)
    graph.add_conditional_edges("agent", after_agent)
    graph.add_edge("tools", "agent")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


def ask(app, question: str, thread_id: str = "cli") -> tuple[str, list[Step]]:
    """Run one turn. The same thread_id continues the same conversation."""
    config = {"configurable": {"thread_id": thread_id}}
    state = app.invoke({"question": question}, config)
    return state["answer"], [Step(**step) for step in state.get("steps", [])]


def main() -> None:
    parser = argparse.ArgumentParser(description="SpotCast agent built with LangGraph.")
    parser.add_argument(
        "--diagram", action="store_true", help="Print the graph as Mermaid"
    )
    args = parser.parse_args()

    app = build_graph(checkpointer=MemorySaver())
    if args.diagram:
        print(app.get_graph().draw_mermaid())
        return

    # Two turns in ONE conversation: the second only makes sense with memory
    conversation = [
        "Cumartesi Mersin sahilinde rüzgâr sörfü yapmak istiyorum, koşullar uygun mu?",
        "Peki pazar günü nasıl?",
    ]
    for question in conversation:
        print(f"\n❓ {question}")
        started = time.monotonic()
        for update in app.stream(
            {"question": question},
            {"configurable": {"thread_id": "demo"}},
            stream_mode="updates",  # one item per finished node
        ):
            for node, change in update.items():
                if node == "tools":
                    for step in change["steps"][-len(change["messages"]) :]:
                        print(
                            f"   🔧 {step['tool']}({step['arguments']})  "
                            f"({step['seconds']:.1f} s)"
                        )
                elif node == "finish" or (node == "profile" and change.get("answer")):
                    print(f"\n💬 {change['answer']}")
                else:
                    print(f"   · {node}")
        print(f"   ({time.monotonic() - started:.0f} s)")
