"""Which way does the shore face? Computed from the OpenStreetMap coastline.

    uv run python scripts/shore_facing.py              # every location in locations.toml
    uv run python scripts/shore_facing.py Alaçatı Foça # only these
    uv run python scripts/shore_facing.py --refresh    # download the coastline again

It downloads the coastline around every location in ONE request (Overpass API; the free
server is often busy, so one request beats thirty), then prints for each location:
- whether the point is on the sea or on land (a location must be on the water),
- the distance to the coast,
- `shore_facing`: the compass direction from the beach towards the open water,
- a suggested point: the nearest shore point moved 250 m out to sea, checked again. Use it
  when a location turns out to be on land or far out (wrong coordinates).

OSM draws coastlines with the land on the left and the sea on the right, so turning a
coastline segment 90 degrees clockwise points to the sea. The facing is the average of
those directions for the coast within 1 km of the nearest shore point, weighted by length.

Copy the values into locations.toml by hand, after a look at a map. The data comes from
OpenStreetMap: © OpenStreetMap contributors, ODbL (openstreetmap.org/copyright).
"""

import json
import math
import sys
import time

import requests

from spotcast.config import DATA_DIR
from spotcast.locations import LOCATIONS

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
SEARCH_RADIUS_M = 3000
LOCAL_COAST_M = (
    1000  # coast closer than this to the nearest shore point sets the facing
)
HEADERS = {"User-Agent": "SpotCast shore-facing script (open-source learning project)"}
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
CACHE = DATA_DIR / "coastline.json"  # data/ is not in git
OFFSHORE_STEP_M = 250  # how far out to sea a suggested point goes


def fetch_coastline(
    points: list[tuple[float, float]],
) -> list[list[tuple[float, float]]]:
    """Coastline ways near any of the points, as lists of (lat, lon)."""
    around = "".join(
        f'way["natural"="coastline"](around:{SEARCH_RADIUS_M},{lat},{lon});'
        for lat, lon in points
    )
    query = f"[out:json][timeout:180];({around});out geom;"
    for attempt in range(8):  # the public server is often busy: wait and retry
        response = requests.post(
            OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=240
        )
        if response.ok:
            ways = response.json()["elements"]
            return [[(p["lat"], p["lon"]) for p in way["geometry"]] for way in ways]
        print(f"  server busy ({response.status_code}), retrying...", flush=True)
        time.sleep(10 * (attempt + 1))
    response.raise_for_status()
    return []


def near(ways, lat: float, lon: float) -> list[list[tuple[float, float]]]:
    """Only the ways with a point within the search radius (plus a margin)."""
    project = to_metres(lat, lon)
    limit = SEARCH_RADIUS_M * 2
    return [w for w in ways if any(math.hypot(*project(*p)) < limit for p in w)]


def to_metres(lat0: float, lon0: float):
    """A flat x/y in metres around (lat0, lon0): accurate enough for a few kilometres."""
    kx = 111_320 * math.cos(math.radians(lat0))

    def project(lat: float, lon: float) -> tuple[float, float]:
        return (lon - lon0) * kx, (lat - lat0) * 111_320

    return project


def bearing(dx: float, dy: float) -> float:
    """Compass direction of a vector: 0 = north, 90 = east."""
    return math.degrees(math.atan2(dx, dy)) % 360


def analyse(lat: float, lon: float, ways) -> dict | None:
    project = to_metres(lat, lon)
    segments = []  # (x1, y1, x2, y2)
    for way in ways:
        points = [project(*p) for p in way]
        segments += [(*a, *b) for a, b in zip(points, points[1:], strict=False)]
    if not segments:
        return None

    def nearest_point(seg):
        x1, y1, x2, y2 = seg
        dx, dy = x2 - x1, y2 - y1
        t = max(0.0, min(1.0, -(x1 * dx + y1 * dy) / (dx * dx + dy * dy or 1)))
        return x1 + t * dx, y1 + t * dy

    distances = [math.hypot(*nearest_point(s)) for s in segments]
    closest = min(range(len(segments)), key=distances.__getitem__)
    x1, y1, x2, y2 = segments[closest]
    # Sea is on the right of the drawing direction: a negative cross product
    on_sea = (x2 - x1) * (0 - y1) - (y2 - y1) * (0 - x1) < 0
    cx, cy = nearest_point(segments[closest])

    # Average the seaward directions of the coast around the nearest shore point
    sx = sy = 0.0
    for seg in segments:
        px, py = nearest_point(seg)
        if math.hypot(px - cx, py - cy) > LOCAL_COAST_M:
            continue
        length = math.hypot(seg[2] - seg[0], seg[3] - seg[1])
        seaward = math.radians(bearing(seg[2] - seg[0], seg[3] - seg[1]) + 90)
        sx += length * math.sin(seaward)
        sy += length * math.cos(seaward)
    facing = bearing(sx, sy)
    # Suggested point: from the nearest shore point, a little way out to sea
    step = math.radians(facing)
    sx_m = cx + OFFSHORE_STEP_M * math.sin(step)
    sy_m = cy + OFFSHORE_STEP_M * math.cos(step)
    kx = 111_320 * math.cos(math.radians(lat))
    return {
        "on_sea": on_sea,
        "coast_m": round(distances[closest]),
        "facing": round(facing / 5) * 5,  # a 5-degree step is as exact as this gets
        "suggested": (round(lat + sy_m / 111_320, 4), round(lon + sx_m / kx, 4)),
    }


def main() -> None:
    wanted = {arg for arg in sys.argv[1:] if not arg.startswith("--")}
    chosen = [loc for loc in LOCATIONS if not wanted or loc.name in wanted]
    if CACHE.exists() and "--refresh" not in sys.argv:
        ways = json.loads(CACHE.read_text())
        print(f"Using the saved coastline ({CACHE.name}, {len(ways)} ways)\n")
    else:
        print(
            f"Downloading the coastline around {len(chosen)} locations...", flush=True
        )
        ways = fetch_coastline([(loc.latitude, loc.longitude) for loc in chosen])
        DATA_DIR.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(ways))
        print(f"{len(ways)} coastline ways\n", flush=True)
    for location in chosen:
        lat, lon = location.latitude, location.longitude
        result = analyse(lat, lon, near(ways, lat, lon))
        if result is None:
            print(f"{location.name:<22} no coastline within {SEARCH_RADIUS_M} m")
            continue
        where = "sea " if result["on_sea"] else "LAND"
        compass = COMPASS[round(result["facing"] / 45) % 8]
        line = (
            f"{location.name:<22} {where} coast {result['coast_m']:>5} m   "
            f"shore_facing = {result['facing']:>3} ({compass:<2})"
        )
        new_lat, new_lon = result["suggested"]
        check = analyse(new_lat, new_lon, near(ways, new_lat, new_lon))
        if check:
            new_compass = COMPASS[round(check["facing"] / 45) % 8]
            line += (
                f"   suggested {new_lat}, {new_lon}: "
                f"{'sea' if check['on_sea'] else 'LAND'} {check['coast_m']} m, "
                f"facing {check['facing']} ({new_compass})"
            )
        print(line, flush=True)


if __name__ == "__main__":
    main()
