"""Tests for the MCP server, talking to it in-process with the MCP client."""

import asyncio
import json

from mcp import Client

from spotcast.mcp_server import server
from spotcast.tools import TOOLS_BY_NAME


class FakeTool:
    def __init__(self, result, prints=None):
        self.result, self.prints, self.received = result, prints, None

    def call(self, arguments):
        self.received = arguments
        if self.prints:
            print(self.prints)  # a stray print must not reach the MCP channel
        return self.result


def call(name, arguments):
    async def go():
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(go())


def test_all_four_tools_are_offered():
    async def go():
        async with Client(server) as client:
            return await client.list_tools()

    tools = {tool.name: tool for tool in asyncio.run(go()).tools}

    assert set(tools) == {
        "find_location",
        "check_conditions",
        "recommend_spots",
        "search_knowledge",
    }
    level = tools["check_conditions"].input_schema["properties"]["level"]
    assert level["enum"] == ["beginner", "intermediate", "advanced"]


def test_unsupported_level_is_an_mcp_error(monkeypatch):
    fake = FakeTool({"label_counts": {}})
    monkeypatch.setitem(TOOLS_BY_NAME, "check_conditions", fake)

    result = call(
        "check_conditions",
        {"latitude": 36.8, "longitude": 34.6, "day": "2026-11-07", "level": "beginner"},
    )

    assert result.is_error
    assert "henüz öneri veremiyorum" in result.content[0].text
    assert fake.received is None  # never ran


def test_results_say_who_they_are_for_and_carry_the_caution(monkeypatch, capsys):
    monkeypatch.setitem(
        TOOLS_BY_NAME,
        "recommend_spots",
        FakeTool({"windows_best_first": []}, prints="noise"),
    )

    result = call("recommend_spots", {"weekend_only": True})

    data = json.loads(result.content[0].text)
    assert not result.is_error
    assert data["valid_for"] == "windsurf, intermediate level only"
    assert "kendin kontrol et" in data["caution"]
    assert "noise" in capsys.readouterr().err  # the print went to stderr, not stdout


def test_tool_errors_become_mcp_errors(monkeypatch):
    monkeypatch.setitem(
        TOOLS_BY_NAME, "find_location", FakeTool({"error": "No place called 'X'"})
    )

    result = call("find_location", {"name": "X"})

    assert result.is_error
    assert "No place called 'X'" in result.content[0].text
