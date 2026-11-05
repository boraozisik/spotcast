"""SpotCast as an MCP server, so other AI apps (Claude Desktop, IDEs, ...) can use its tools.

MCP (Model Context Protocol) is a standard way for an AI app to discover and call outside
tools: like REST or GraphQL, but for LLM tools. The app starts this server as a subprocess
and talks to it over stdin/stdout.

The tools are the same ones our own agent uses (tools.py). One big difference: here a
DIFFERENT AI decides how to call them, and we can't see the user's question. So the safety
rules that protect our agent must live in the tools themselves:
- the level is a parameter, and unsupported levels are refused with a clear message,
- every conditions result says which level it is valid for and carries the fixed caution.

Run: uv run spotcast-mcp   (normally an MCP client starts it, not you)
"""

import contextlib
import sys
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from spotcast.agent import UserProfile, decide_profile
from spotcast.explain import CAUTION
from spotcast.tools import TOOLS_BY_NAME

INSTRUCTIONS = """SpotCast checks wind, waves and suitability for water sports in Turkey,
using weather forecasts and SpotCast's own model. Today only windsurf at intermediate level
is supported. Always pass the user's real level: tools refuse unsupported levels, and results
for one level must never be used for someone at another level. Always tell the user the
'warnings' and the 'caution' that come with a result."""

server = MCPServer("spotcast", instructions=INSTRUCTIONS)

Level = Annotated[
    Literal["beginner", "intermediate", "advanced"],
    Field(
        description="The user's real windsurf level. Do not change it to get a result."
    ),
]
Day = Annotated[str, Field(description="YYYY-MM-DD, within the next 7 days")]


def _run(name: str, arguments: dict) -> dict:
    """Run one of our tools.

    - In stdio mode stdout IS the MCP channel: any print() would corrupt it, so everything
      our code prints goes to stderr instead.
    - Our tools return {"error": ...}; MCP has its own way (isError), which tells the
      client "this is a failure, not a result", so we turn one into the other.
    """
    with contextlib.redirect_stdout(sys.stderr):
        result = TOOLS_BY_NAME[name].call(arguments)
    if "error" in result:
        raise ToolError(result["error"])
    return result


def _check_level(level: str) -> None:
    who = decide_profile(UserProfile(sport="windsurf", level=level))
    if who.refusal:
        raise ToolError(who.refusal)


def _with_safety(result: dict, level: str) -> dict:
    return result | {"valid_for": f"windsurf, {level} level only", "caution": CAUTION}


@server.tool(description=TOOLS_BY_NAME["find_location"].description)
def find_location(
    name: Annotated[str, Field(description="Place name in Turkey, e.g. 'Mersin'")],
) -> dict:
    return _run("find_location", {"name": name})


@server.tool(description=TOOLS_BY_NAME["check_conditions"].description)
def check_conditions(
    latitude: float, longitude: float, day: Day, level: Level = "intermediate"
) -> dict:
    _check_level(level)
    arguments = {
        "latitude": latitude,
        "longitude": longitude,
        "day": day,
        "level": level,
    }
    return _with_safety(_run("check_conditions", arguments), level)


@server.tool(description=TOOLS_BY_NAME["recommend_spots"].description)
def recommend_spots(
    day: Day | None = None,
    weekend_only: bool = False,
    earliest_hour: Annotated[int, Field(ge=0, le=23)] = 9,
    latest_hour: Annotated[int, Field(ge=0, le=23)] = 18,
    level: Level = "intermediate",
) -> dict:
    _check_level(level)
    arguments = {
        "day": day,
        "weekend_only": weekend_only,
        "earliest_hour": earliest_hour,
        "latest_hour": latest_hour,
        "level": level,
    }
    return _with_safety(_run("recommend_spots", arguments), level)


@server.tool(description=TOOLS_BY_NAME["search_knowledge"].description)
def search_knowledge(
    question: Annotated[str, Field(description="What to look up, with context")],
    spot: Annotated[str | None, Field(description="Limit to one known spot")] = None,
) -> dict:
    return _run("search_knowledge", {"question": question, "spot": spot})


def main() -> None:
    server.run("stdio")
