"""SpotCast as an MCP server, so other AI apps (Claude Desktop, IDEs, ...) can use its tools.

MCP (Model Context Protocol) is a standard way for an AI app to discover and call outside
tools: like REST or GraphQL, but for LLM tools. The app starts this server as a subprocess
and talks to it over stdin/stdout.

The tools are the same ones our own agent uses (tools.py). One big difference: here a
DIFFERENT AI decides how to call them, and we can't see the user's question. So the safety
rules that protect our agent must live in the tools themselves:
- sport and level are parameters, and unsupported ones are refused with a clear message,
- every conditions result says which sport and level it is valid for and carries the fixed
  caution.

Run: uv run spotcast-mcp   (normally an MCP client starts it, not you)
"""

import contextlib
import sys
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from spotcast.agent import unsupported_answer
from spotcast.explain import CAUTION
from spotcast.profiles import LEVELS, PROFILES, SPORTS
from spotcast.tools import TOOLS_BY_NAME

INSTRUCTIONS = """SpotCast checks wind, waves and suitability for water sports in Turkey,
using weather forecasts and SpotCast's own model. Supported: windsurf, kitesurf, SUP and
kayak, each at beginner, intermediate and advanced level. Always pass the user's real sport
and level: results for one sport or level must never be used for another (20 knots is ideal
for windsurf and dangerous on a SUP). Always tell the user the 'warnings' and the 'caution'
that come with a result."""

server = MCPServer("spotcast", instructions=INSTRUCTIONS)

Sport = Annotated[
    Literal[SPORTS],
    Field(
        description="The user's sport. kayak = sea kayak (kano), sup = stand-up paddle."
    ),
]
Level = Annotated[
    Literal[LEVELS],
    Field(description="The user's real level. Do not change it to get a result."),
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


def _check_profile(sport: str, level: str) -> None:
    if (sport, level) not in PROFILES:
        raise ToolError(unsupported_answer(sport, level))


def _with_safety(result: dict, sport: str, level: str) -> dict:
    return result | {"valid_for": f"{sport}, {level} level only", "caution": CAUTION}


@server.tool(description=TOOLS_BY_NAME["find_location"].description)
def find_location(
    name: Annotated[str, Field(description="Place name in Turkey, e.g. 'Mersin'")],
) -> dict:
    return _run("find_location", {"name": name})


@server.tool(description=TOOLS_BY_NAME["check_conditions"].description)
def check_conditions(
    latitude: float,
    longitude: float,
    day: Day,
    sport: Sport = "windsurf",
    level: Level = "intermediate",
) -> dict:
    _check_profile(sport, level)
    arguments = {
        "latitude": latitude,
        "longitude": longitude,
        "day": day,
        "sport": sport,
        "level": level,
    }
    return _with_safety(_run("check_conditions", arguments), sport, level)


@server.tool(description=TOOLS_BY_NAME["recommend_spots"].description)
def recommend_spots(
    day: Day | None = None,
    weekend_only: bool = False,
    earliest_hour: Annotated[int, Field(ge=0, le=23)] = 9,
    latest_hour: Annotated[int, Field(ge=0, le=23)] = 18,
    sport: Sport = "windsurf",
    level: Level = "intermediate",
) -> dict:
    _check_profile(sport, level)
    arguments = {
        "day": day,
        "weekend_only": weekend_only,
        "earliest_hour": earliest_hour,
        "latest_hour": latest_hour,
        "sport": sport,
        "level": level,
    }
    return _with_safety(_run("recommend_spots", arguments), sport, level)


@server.tool(description=TOOLS_BY_NAME["recommend_near"].description)
def recommend_near(
    latitude: float,
    longitude: float,
    radius_km: Annotated[
        float,
        Field(gt=0, le=300, description="How far the user will go, straight-line km"),
    ] = 100,
    day: Day | None = None,
    weekend_only: bool = False,
    earliest_hour: Annotated[int, Field(ge=0, le=23)] = 9,
    latest_hour: Annotated[int, Field(ge=0, le=23)] = 18,
    sport: Sport = "windsurf",
    level: Level = "intermediate",
) -> dict:
    _check_profile(sport, level)
    arguments = {
        "latitude": latitude,
        "longitude": longitude,
        "radius_km": radius_km,
        "day": day,
        "weekend_only": weekend_only,
        "earliest_hour": earliest_hour,
        "latest_hour": latest_hour,
        "sport": sport,
        "level": level,
    }
    return _with_safety(_run("recommend_near", arguments), sport, level)


@server.tool(description=TOOLS_BY_NAME["search_knowledge"].description)
def search_knowledge(
    question: Annotated[str, Field(description="What to look up, with context")],
    spot: Annotated[str | None, Field(description="Limit to one known spot")] = None,
) -> dict:
    return _run("search_knowledge", {"question": question, "spot": spot})


def main() -> None:
    server.run("stdio")
