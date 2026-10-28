import time

from flask import Flask, redirect, render_template, url_for
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# Works whether the app is loaded as the package module "app.main" (flask run, most hosts)
# or as a standalone file with app/ on the import path.
try:
    from app.db import engine
except ImportError:
    from db import engine

# Static files live in public/static so Vercel's CDN can serve them; Flask serves the same
# folder at /static when running locally.
app = Flask(__name__, static_folder="../public/static", static_url_path="/static")


FIRST_SEASON = 2000
REGULAR_SEASON_WEEKS = 18
GITHUB_URL = "https://github.com/BenjaminHowland55/Sports_Analytics_Project"

# Home page cards: (label, stat column, what it measures)
HOME_CARDS = [
    ("Overall Defense", "epa_per_play_allowed", "EPA per play allowed"),
    ("Pass Defense", "pass_epa_per_dropback_allowed", "EPA per dropback allowed"),
    ("Rush Defense", "rush_epa_per_carry_allowed", "EPA per carry allowed"),
    ("Pass Rush", "four_man_pressure_rate", "pressure rate with 4 or fewer rushers"),
    ("Coverage", "clean_pocket_epa_allowed", "EPA allowed on clean-pocket dropbacks"),
]


def get_arizona_history():
    query = text("""
        SELECT season, plays, last_week,
               epa_per_play_allowed, epa_per_play_allowed_rank,
               pass_epa_per_dropback_allowed, pass_epa_per_dropback_allowed_rank,
               rush_epa_per_carry_allowed, rush_epa_per_carry_allowed_rank,
               explosive_play_rate_allowed, explosive_play_rate_allowed_rank,
               pressure_rate, pressure_rate_rank,
               four_man_pressure_rate, four_man_pressure_rate_rank,
               clean_pocket_epa_allowed, clean_pocket_epa_allowed_rank,
               blitz_rate, man_coverage_rate
        FROM team_defense_history
        WHERE team = 'ARI' AND season >= :first_season
        ORDER BY season DESC
    """)
    with engine.connect() as conn:
        rows = [dict(row._mapping) for row in conn.execute(query, {"first_season": FIRST_SEASON})]
    # The newest season is still being played if it hasn't reached the last regular-season week
    if rows:
        rows[0]["in_progress"] = rows[0]["last_week"] < REGULAR_SEASON_WEEKS
    return rows


def get_home_cards():
    """Arizona's ranks in the most recent season that has charting data (pressure, coverage)."""
    columns = ", ".join(f"h.{stat}, h.{stat}_rank" for _, stat, _ in HOME_CARDS)
    query = text(f"""
        SELECT h.season, {columns},
               (SELECT COUNT(*) FROM team_defense_history t WHERE t.season = h.season) AS teams
        FROM team_defense_history h
        WHERE h.team = 'ARI' AND h.four_man_pressure_rate IS NOT NULL
        ORDER BY h.season DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(query).mappings().first()
    if row is None:
        return None, []
    cards = [{"label": label, "stat": stat, "measure": measure,
              "value": row[stat], "rank": row[stat + "_rank"]}
             for label, stat, measure in HOME_CARDS]
    return {"season": row["season"], "teams": row["teams"]}, cards


ROOM_ORDER = ["ED", "IDL", "LB", "CB", "S"]
TOP_PLAYERS = 10


def get_report_card():
    """Room grades for every team, Arizona's graded players, the league's best at each position, and the
    stats behind the grades."""
    with engine.connect() as conn:
        rooms = [dict(r._mapping) for r in conn.execute(text("SELECT * FROM room_grades"))]
        players = [dict(r._mapping) for r in conn.execute(
            text("SELECT * FROM player_grades WHERE team = 'ARI' ORDER BY snaps DESC"))]
        listed = [dict(r._mapping) for r in conn.execute(text(
            "SELECT * FROM player_grades WHERE listed = 1 ORDER BY composite DESC, snaps DESC"))]
        stats = [dict(r._mapping) for r in conn.execute(text("SELECT * FROM grade_stats"))]
    if not rooms:
        return None

    # A player traded this season has a row per team; rank him once, under the team he's played most for
    top = {room: [] for room in ROOM_ORDER}
    seen = set()
    for p in sorted(listed, key=lambda p: -p["snaps"]):
        if p["pfr_id"] not in seen:
            seen.add(p["pfr_id"])
            top[p["room"]].append(p)
    top = {room: sorted(ps, key=lambda p: -p["composite"])[:TOP_PLAYERS] for room, ps in top.items()}

    order = {room: i for i, room in enumerate(ROOM_ORDER)}
    by_team = {}
    for row in rooms:
        team = by_team.setdefault(row["team"], {"team": row["team"], "cap_pct": 0.0})
        team[row["room"]] = row["grade_rank"]
        team["cap_pct"] += row["cap_pct"]
    return {
        "season": rooms[0]["season"],
        "arizona": sorted((r for r in rooms if r["team"] == "ARI"), key=lambda r: order[r["room"]]),
        "league": sorted(by_team.values(), key=lambda t: t["team"]),
        "players": {room: [p for p in players if p["room"] == room] for room in ROOM_ORDER},
        "top": top,
        "room_names": {r["room"]: r["room_name"] for r in rooms},
        "used_stats": {room: [s["stat"] for s in stats if s["room"] == room and s["used"]] for room in ROOM_ORDER},
        # Within each room, the biggest weights first and unused stats last
        "stats": sorted(stats, key=lambda s: (order[s["room"]], -(s["weight_share"] or 0), -s["year_to_year_r"])),
        # The market study's sample and fit, one row per room
        "market": [next(s for s in stats if s["room"] == room) for room in ROOM_ORDER],
    }


# The footer's "data through" line, refreshed at most once an hour so pages don't all hit the database
_data_freshness = {"checked": 0.0, "value": None}


def get_data_freshness():
    if time.time() - _data_freshness["checked"] > 3600:
        try:
            with engine.connect() as conn:
                row = conn.execute(text("""
                    SELECT season, MAX(last_week) AS week FROM team_defense_history
                    WHERE season = (SELECT MAX(season) FROM team_defense_history)
                    GROUP BY season
                """)).mappings().first()
            _data_freshness["value"] = dict(row) if row else None
        except SQLAlchemyError:
            app.logger.exception("Could not load data freshness")
        _data_freshness["checked"] = time.time()
    return _data_freshness["value"]


@app.context_processor
def footer_context():
    return {"data_through": get_data_freshness(), "github_url": GITHUB_URL}


@app.route("/")
def home():
    try:
        card_season, cards = get_home_cards()
    except SQLAlchemyError:
        app.logger.exception("Could not load home page cards")
        card_season, cards = None, []
    return render_template("home.html", card_season=card_season, cards=cards)


@app.route("/stats")
def stats():
    try:
        rows = get_arizona_history()
    except SQLAlchemyError:
        app.logger.exception("Could not load team_defense_history rows")
        rows = []
    return render_template("stats.html", rows=rows)


@app.route("/report-card")
def report_card():
    try:
        card = get_report_card()
    except SQLAlchemyError:
        app.logger.exception("Could not load report card tables")
        card = None
    return render_template("report_card.html", card=card, room_order=ROOM_ORDER)


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/visualizations")
def visualizations():
    return render_template("visualizations.html")


@app.route("/source")
def source():
    return render_template("source.html")


# The old About page became Home; send old links there instead of a 404.
@app.route("/about")
def about():
    return redirect(url_for("home"), code=301)
