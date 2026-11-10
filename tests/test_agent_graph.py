"""Tests for the LangGraph version of the agent. A fake chat model replays scripted replies."""

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from spotcast.agent import GAVE_UP, MAX_STEPS, UserProfile
from spotcast.agent_graph import ask, build_graph
from spotcast.tools import TOOLS_BY_NAME, ProfileArgs


class FakeChatModel:
    """Just enough of a LangChain chat model: bind_tools() and invoke()."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


class FakeTool:
    def __init__(self, result):
        self.result, self.args, self.received = result, ProfileArgs, []

    def call(self, arguments):
        self.received.append(arguments)
        return self.result


def reads(sport="windsurf", level="intermediate"):
    return lambda question: UserProfile(sport=sport, level=level)


def calls_tool(tool_name, call_id="1", **arguments):
    call = {"name": tool_name, "args": arguments, "id": call_id}
    return AIMessage("", tool_calls=[call])


def test_tool_loop_and_finished_answer(monkeypatch):
    tool = FakeTool({"windows_best_first": []})
    monkeypatch.setitem(TOOLS_BY_NAME, "recommend_spots", tool)
    llm = FakeChatModel(
        calls_tool("recommend_spots", level="advanced"), AIMessage("Alaçatı.")
    )
    app = build_graph(
        llm=llm, profile_reader=reads(level="unknown"), checkpointer=MemorySaver()
    )

    answer, steps = ask(app, "Nereye gideyim?")

    assert answer.startswith("Alaçatı.")
    assert (
        "orta seviye rüzgâr sörfü içindir" in answer and "kendin kontrol et" in answer
    )
    assert [s.tool for s in steps] == ["recommend_spots"]
    # Code, not the LLM, sets the level
    assert tool.received[0]["level"] == "intermediate"


def test_unsupported_activity_never_reaches_the_llm():
    llm = FakeChatModel()  # would fail if invoked
    app = build_graph(
        llm=llm, profile_reader=reads(sport="other"), checkpointer=MemorySaver()
    )

    answer, steps = ask(app, "Yarın dalışa gideceğim")

    assert "henüz öneri veremiyorum" in answer
    assert llm.calls == [] and steps == []


def test_follow_up_remembers_the_conversation(monkeypatch):
    monkeypatch.setitem(
        TOOLS_BY_NAME, "check_conditions", FakeTool({"label_counts": {}})
    )
    llm = FakeChatModel(
        calls_tool(
            "check_conditions", "1", latitude=36.8, longitude=34.6, day="2026-11-07"
        ),
        AIMessage("Cumartesi uygun değil."),
        calls_tool(
            "check_conditions", "2", latitude=36.8, longitude=34.6, day="2026-11-08"
        ),
        AIMessage("Pazar da uygun değil."),
    )
    said = iter([UserProfile(sport="windsurf", level="unknown")] * 2)
    app = build_graph(
        llm=llm, profile_reader=lambda q: next(said), checkpointer=MemorySaver()
    )

    ask(app, "Cumartesi Mersin?", thread_id="t1")
    answer, _ = ask(app, "Peki pazar?", thread_id="t1")

    second_turn_messages = llm.calls[2]
    texts = [str(m.content) for m in second_turn_messages]
    # The first question is still in the conversation
    assert "Cumartesi Mersin?" in texts
    assert sum(isinstance(m, SystemMessage) for m in second_turn_messages) == 1
    assert (
        "orta seviye rüzgâr sörfü içindir" in answer
    )  # still unknown level → note again


def test_step_limit_stops_the_graph(monkeypatch):
    monkeypatch.setitem(TOOLS_BY_NAME, "find_location", FakeTool({"places": []}))
    llm = FakeChatModel(
        *[calls_tool("find_location", str(i), name="x") for i in range(MAX_STEPS + 1)]
    )
    app = build_graph(llm=llm, profile_reader=reads(), checkpointer=MemorySaver())

    answer, steps = ask(app, "?")

    assert answer == GAVE_UP
    assert len(steps) == MAX_STEPS
