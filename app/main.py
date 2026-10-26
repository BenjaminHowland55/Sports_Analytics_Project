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


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/stats")
def stats():
    try:
        rows = get_arizona_history()
    except SQLAlchemyError:
        app.logger.exception("Could not load team_defense_history rows")
        rows = []
    return render_template("stats.html", rows=rows)


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
