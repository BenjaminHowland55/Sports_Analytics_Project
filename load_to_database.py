"""Upload the nflreadpy analysis tables to the project's TiDB Cloud (MySQL) database.

Creates (or replaces) two tables in the `nfl_analytics` database:
  plays         one row per regular-season run/pass play, 2023-2025, with QB pressure, blitz
                (5+ rushers), defensive package (base/nickel/dime by DB count) and man/zone coverage
  team_defense  one row per defense per season: the notebook's table (data/team_defense_2023_2025.csv)
                plus blitz, man-coverage and package rates

A separate mode builds a long-running history table instead:
  team_defense_history  one row per defense per season, 2000 to the current season, with plays,
                        last week played, EPA per play / per dropback / per carry allowed, explosive
                        play rate allowed, and (where charted) pressure, 4-or-fewer-rusher pressure,
                        clean-pocket EPA and blitz rates (2016 on) and man coverage rate (2018 on).
                        Each better-or-worse stat also gets a *_rank column (1 = best that season).

A third mode builds the player tables (every team, so Arizona's players can be compared with the league):
  player_defense_season  one row per defender per season, 2018 to the current season: Pro Football
                         Reference's charted stats (pressures, sacks, targets and yards allowed in
                         coverage, missed tackles) plus regular-season defensive snaps
  player_impact          one row per defender per season, 2016-2025 (seasons with participation data):
                         the defense's EPA allowed with the player on the field, and an adjusted
                         plus-minus that credits each play's EPA to the 11 defenders on the field
                         while accounting for teammates and the offense faced (ridge regression).
                         Positive impact = fewer EPA allowed. Very noisy: it barely repeats from one
                         season to the next, so treat it as a clue, not a verdict.

Run from this folder:  python load_to_database.py           (plays + team_defense)
                       python load_to_database.py history   (team_defense_history, ~3-5 minutes)
                       python load_to_database.py players   (player tables, ~5-10 minutes)
The password is read from .env (key: DATABASE_PASSWORD) and is never printed.
"""
import sys

import certifi
import nflreadpy as nfl
import numpy as np
import pandas as pd
import polars as pl
from dotenv import dotenv_values
from scipy import sparse
from sklearn.linear_model import Ridge
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

SEASONS = [2023, 2024, 2025]
HISTORY_FIRST_SEASON = 2000
CHARTING_FIRST_SEASON = 2016
PFR_FIRST_SEASON = 2018
# How strongly the plus-minus pulls players toward average; chosen by 5-fold cross-validation on 2025
IMPACT_SHRINKAGE = 1000

# Pro Football Reference column -> readable name
PFR_COLUMNS = {
    "season": "season", "player": "player", "pfr_id": "pfr_id", "tm": "team", "pos": "position",
    "age": "age", "g": "games", "gs": "games_started",
    "prss": "pressures", "hrry": "hurries", "qbkd": "qb_knockdowns", "sk": "sacks", "bltz": "blitzes",
    "tgt": "targets", "cmp": "completions_allowed", "yds": "yards_allowed", "td": "tds_allowed",
    "int": "interceptions", "rat": "passer_rating_allowed", "yds_tgt": "yards_per_target",
    "comb": "tackles", "m_tkl": "missed_tackles", "m_tkl_percent": "missed_tackle_rate",
}

# Stats ranked within each season (1 = best). True means a lower value is better.
RANKED_STATS = {
    "epa_per_play_allowed": True,
    "pass_epa_per_dropback_allowed": True,
    "rush_epa_per_carry_allowed": True,
    "explosive_play_rate_allowed": True,
    "pressure_rate": False,
    "four_man_pressure_rate": False,
    "clean_pocket_epa_allowed": True,
}
DB_USER = "2sjjjYZZAfxKqVv.root"
DB_HOST = "gateway01.us-west-2.prod.aws.tidbcloud.com"
DB_PORT = 4000
DB_NAME = "nfl_analytics"


def connect(database):
    password = dotenv_values(".env")["DATABASE_PASSWORD"]
    url = URL.create("mysql+pymysql", username=DB_USER, password=password,
                     host=DB_HOST, port=DB_PORT, database=database)
    # TiDB Cloud requires an encrypted (TLS) connection
    return create_engine(url, connect_args={"ssl": {"ca": certifi.where()}})


def build_tables():
    print("Downloading data from nflverse...")
    pbp = nfl.load_pbp(seasons=SEASONS).to_pandas()
    participation = nfl.load_participation(seasons=SEASONS).to_pandas()

    # Same cleaning as Step 4 of defense_pressure_vs_coverage.ipynb
    plays = pbp[
        (pbp["season_type"] == "REG")
        & (pbp["play_type"].isin(["pass", "run"]))
        & (pbp["epa"].notna())
    ][["game_id", "play_id", "season", "week", "posteam", "defteam",
       "play_type", "qb_dropback", "sack", "epa"]]
    plays = plays.merge(
        participation[["nflverse_game_id", "play_id", "was_pressure", "number_of_pass_rushers",
                       "defenders_in_box", "defense_personnel", "defense_man_zone_type",
                       "defense_coverage_type"]],
        left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="left",
    ).drop(columns="nflverse_game_id")
    plays["was_pressure"] = plays["was_pressure"].astype(float)

    # Blitz = 5+ pass rushers. Counts of 0 (spikes, missing charting) or over 11 (typos) are unknown.
    rushers = plays["number_of_pass_rushers"]
    plays["number_of_pass_rushers"] = rushers.where(rushers.between(1, 11))
    plays.loc[plays["qb_dropback"] != 1, "number_of_pass_rushers"] = None
    plays["is_blitz"] = (plays["number_of_pass_rushers"] >= 5).astype(float)
    plays.loc[plays["number_of_pass_rushers"].isna(), "is_blitz"] = None

    # Defensive package from the number of defensive backs, e.g. "3 CB, 2 DE, 2 DT, 1 FS, 1 MLB, 1 OLB, 1 SS"
    positions = plays["defense_personnel"].str.extractall(r"(\d+) (\w+)")
    positions.columns = ["count", "position"]
    is_db = positions["position"].isin(["CB", "FS", "SS", "S", "DB"])
    plays["n_defensive_backs"] = (positions.loc[is_db, "count"].astype(int)
                                  .groupby(level=0).sum()
                                  .reindex(plays.index, fill_value=0))
    plays["defense_package"] = pd.cut(plays["n_defensive_backs"], bins=[-1, 3, 4, 5, 6, 11],
                                      labels=["heavy", "base", "nickel", "dime", "quarter"]).astype(str)
    plays.loc[plays["defense_personnel"].isna(), ["n_defensive_backs", "defense_package"]] = None

    # Man vs. zone and coverage shell (pass plays only; blank text means not charted)
    plays["man_zone"] = plays["defense_man_zone_type"].map({"MAN_COVERAGE": "man", "ZONE_COVERAGE": "zone"})
    plays.loc[plays["qb_dropback"] != 1, "man_zone"] = None
    plays = plays.rename(columns={"defense_coverage_type": "coverage_type"}).drop(columns="defense_man_zone_type")

    dropbacks = plays[plays["qb_dropback"] == 1]
    overall = plays.groupby(["season", "defteam"]).agg(
        plays=("epa", "size"), epa_per_play_allowed=("epa", "mean"))
    rush = dropbacks.groupby(["season", "defteam"]).agg(
        dropbacks=("epa", "size"), pressure_rate=("was_pressure", "mean"),
        blitz_rate=("is_blitz", "mean"), avg_pass_rushers=("number_of_pass_rushers", "mean"))
    man_rate = (dropbacks.dropna(subset=["man_zone"]).assign(man=lambda d: d["man_zone"] == "man")
                .groupby(["season", "defteam"])["man"].mean().rename("man_coverage_rate"))
    packages = (pd.crosstab([plays["season"], plays["defteam"]], plays["defense_package"], normalize="index")
                [["base", "nickel", "dime"]].add_suffix("_rate"))
    clean = (dropbacks[dropbacks["was_pressure"] == 0].groupby(["season", "defteam"])["epa"]
             .mean().rename("clean_pocket_epa_allowed"))
    pressured = (dropbacks[dropbacks["was_pressure"] == 1].groupby(["season", "defteam"])["epa"]
                 .mean().rename("pressured_epa_allowed"))
    team_defense = (overall.join(rush).join(clean).join(pressured).join(man_rate).join(packages)
                    .reset_index().rename(columns={"defteam": "team"}))

    # Whole-number columns stored as integers (0/1 flags; blank stays blank)
    plays = plays.astype({"play_id": "int64", "season": "int64", "week": "int64",
                          "qb_dropback": "Int8", "sack": "Int8", "was_pressure": "Int8",
                          "is_blitz": "Int8", "number_of_pass_rushers": "Int8",
                          "defenders_in_box": "Int8", "n_defensive_backs": "Int8"})
    return plays, team_defense


def build_history():
    current_season = nfl.get_current_season()
    print(f"Downloading play-by-play {HISTORY_FIRST_SEASON}-{current_season} from nflverse...")

    # One season at a time, keeping only the needed columns, so 25+ seasons fit in memory.
    # Older seasons store play_id as a decimal, so cast it to match the participation table.
    plays = pd.concat([
        nfl.load_pbp(seasons=[season])
           .select(["game_id", "play_id", "season", "week", "season_type", "defteam",
                    "play_type", "qb_dropback", "yards_gained", "epa"])
           .filter((pl.col("season_type") == "REG")
                   & pl.col("play_type").is_in(["pass", "run"])
                   & pl.col("epa").is_not_null())
           .with_columns(pl.col("play_id").cast(pl.Int64))
           .to_pandas()
        for season in range(HISTORY_FIRST_SEASON, current_season + 1)
    ], ignore_index=True)

    # Charting starts in 2016; nflverse publishes it only for completed seasons
    charting = []
    for season in range(CHARTING_FIRST_SEASON, current_season + 1):
        try:
            charting.append(nfl.load_participation(seasons=[season])
                            .select(["nflverse_game_id", "play_id", "was_pressure",
                                     "number_of_pass_rushers", "defense_man_zone_type"])
                            .with_columns(pl.col("play_id").cast(pl.Int64))
                            .to_pandas())
        except ValueError:
            print(f"  No charting data for {season} yet")
    participation = pd.concat(charting, ignore_index=True)

    dropbacks = plays[plays["qb_dropback"] == 1].merge(
        participation, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="left")
    rushers = dropbacks["number_of_pass_rushers"].where(dropbacks["number_of_pass_rushers"].between(1, 11))
    dropbacks["pressure"] = dropbacks["was_pressure"].astype(float)
    dropbacks["blitz"] = (rushers >= 5).astype(float).where(rushers.notna())
    dropbacks["man"] = dropbacks["defense_man_zone_type"].map({"MAN_COVERAGE": 1.0, "ZONE_COVERAGE": 0.0})
    # Pressure created without blitzing, and EPA when the QB had a clean pocket
    dropbacks["four_man_pressure"] = dropbacks["pressure"].where(rushers <= 4)
    dropbacks["clean_epa"] = dropbacks["epa"].where(dropbacks["pressure"] == 0)

    # Dropbacks (passes, sacks, scrambles) vs. designed runs; explosive = 20+ yards passing, 10+ rushing
    is_dropback = plays["qb_dropback"] == 1
    is_run = (plays["play_type"] == "run") & ~is_dropback
    plays["pass_epa"] = plays["epa"].where(is_dropback)
    plays["rush_epa"] = plays["epa"].where(is_run)
    plays["explosive"] = ((is_dropback & (plays["yards_gained"] >= 20))
                          | (is_run & (plays["yards_gained"] >= 10))).astype(float)

    overall = plays.groupby(["season", "defteam"]).agg(
        plays=("epa", "size"), last_week=("week", "max"), epa_per_play_allowed=("epa", "mean"),
        pass_epa_per_dropback_allowed=("pass_epa", "mean"), rush_epa_per_carry_allowed=("rush_epa", "mean"),
        explosive_play_rate_allowed=("explosive", "mean"))
    # Rates use charted dropbacks only; seasons with no charting stay blank
    rates = (dropbacks.groupby(["season", "defteam"])
             [["pressure", "four_man_pressure", "clean_epa", "blitz", "man"]].mean()
             .rename(columns={"pressure": "pressure_rate", "four_man_pressure": "four_man_pressure_rate",
                              "clean_epa": "clean_pocket_epa_allowed", "blitz": "blitz_rate",
                              "man": "man_coverage_rate"}))
    history = (overall.join(rates).reset_index().rename(columns={"defteam": "team"})
               .astype({"season": "int64", "plays": "int64", "last_week": "int64"}))

    for stat, lower_is_better in RANKED_STATS.items():
        history[stat + "_rank"] = (history.groupby("season")[stat]
                                   .rank(ascending=lower_is_better, method="min").astype("Int64"))
    return history


def build_player_season():
    current_season = nfl.get_current_season()
    seasons = list(range(PFR_FIRST_SEASON, current_season + 1))
    print(f"Downloading player stats {PFR_FIRST_SEASON}-{current_season} from nflverse...")
    stats = (nfl.load_pfr_advstats(seasons=seasons, stat_type="def", summary_level="season")
             .to_pandas()[list(PFR_COLUMNS)].rename(columns=PFR_COLUMNS))

    # Regular-season defensive snaps; a traded player ("2TM") is listed under the last team played for
    snaps = (nfl.load_snap_counts(seasons=seasons).filter(pl.col("game_type") == "REG").to_pandas()
             .sort_values(["season", "week"]))
    per_player = snaps.groupby(["season", "pfr_player_id"]).agg(
        defense_snaps=("defense_snaps", "sum"), last_team=("team", "last")).reset_index()
    stats = stats.merge(per_player, left_on=["season", "pfr_id"], right_on=["season", "pfr_player_id"], how="left")
    stats["team"] = stats["team"].where(~stats["team"].str.endswith("TM"), stats["last_team"])

    # gsis_id links a player to the play-by-play and participation data
    ids = nfl.load_players().select(["pfr_id", "gsis_id"]).drop_nulls().unique("pfr_id").to_pandas()
    stats = stats.merge(ids, on="pfr_id", how="left").drop(columns=["pfr_player_id", "last_team"])

    # A few players have stats but no recorded defensive snaps; their per-snap rates stay blank
    snaps_played = stats["defense_snaps"].where(stats["defense_snaps"] > 0)
    stats["pressures_per_100_snaps"] = 100 * stats["pressures"] / snaps_played
    stats["targets_per_100_snaps"] = 100 * stats["targets"] / snaps_played
    return stats.astype({"season": "int64", "defense_snaps": "Int64"})


def build_player_impact():
    current_season = nfl.get_current_season()
    names = nfl.load_players().select(["gsis_id", "display_name", "position"]).to_pandas()
    seasons = []
    for season in range(CHARTING_FIRST_SEASON, current_season + 1):
        try:
            on_field = (nfl.load_participation(seasons=[season])
                        .select(["nflverse_game_id", "play_id", "defense_players", "n_defense"])
                        .with_columns(pl.col("play_id").cast(pl.Int64)).to_pandas())
        except ValueError:
            print(f"  No participation data for {season} yet")
            continue
        print(f"  Fitting player impact for {season}...")
        plays = (nfl.load_pbp(seasons=[season])
                 .select(["game_id", "play_id", "season_type", "posteam", "defteam", "play_type", "qb_dropback", "epa"])
                 .filter((pl.col("season_type") == "REG")
                         & pl.col("play_type").is_in(["pass", "run"])
                         & pl.col("epa").is_not_null())
                 .with_columns(pl.col("play_id").cast(pl.Int64))
                 .to_pandas())
        plays = plays.merge(on_field, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"])
        # Keep plays with a believable defender count (a few have 10 or 12 from charting errors or penalties)
        plays = plays[plays["n_defense"].between(10, 12)].reset_index(drop=True)
        plays["is_pass"] = plays["qb_dropback"] == 1

        # One row per defender per play
        player_plays = (plays[["defense_players", "defteam", "is_pass", "epa"]]
                        .assign(gsis_id=lambda d: d["defense_players"].str.split(";"))
                        .explode("gsis_id").drop(columns="defense_players"))
        player_plays = player_plays[player_plays["gsis_id"].str.len() > 0]

        # Plain on-field numbers: plays, main team, EPA allowed on dropbacks and on runs
        summary = player_plays.groupby("gsis_id").agg(
            team=("defteam", lambda t: t.mode().iloc[0]), plays=("epa", "size"), pass_plays=("is_pass", "sum"))
        summary["on_field_epa_per_dropback"] = player_plays[player_plays["is_pass"]].groupby("gsis_id")["epa"].mean()
        summary["on_field_epa_per_carry"] = player_plays[~player_plays["is_pass"]].groupby("gsis_id")["epa"].mean()

        # Adjusted plus-minus: EPA ~ the defenders on the field + the offense faced, fit separately for
        # pass and run plays. The shrinkage keeps players with few snaps near zero.
        players = pd.Index(summary.index)
        offenses = pd.Index(plays["posteam"].unique())
        rows = player_plays.index.to_numpy()
        on_field_matrix = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, players.get_indexer(player_plays["gsis_id"]))),
            shape=(len(plays), len(players)))
        offense_matrix = sparse.csr_matrix(
            (np.ones(len(plays)), (np.arange(len(plays)), offenses.get_indexer(plays["posteam"]))),
            shape=(len(plays), len(offenses)))
        X = sparse.hstack([on_field_matrix, offense_matrix]).tocsr()
        for name, mask in [("pass_impact", plays["is_pass"]), ("run_impact", ~plays["is_pass"])]:
            model = Ridge(alpha=IMPACT_SHRINKAGE).fit(X[mask.to_numpy()], plays.loc[mask, "epa"].to_numpy())
            # Flip the sign so a positive number means the defense allowed less EPA per play
            summary[name] = -model.coef_[:len(players)]
        summary["season"] = season
        seasons.append(summary.reset_index())

    impact = pd.concat(seasons, ignore_index=True).merge(names, on="gsis_id", how="left")
    impact = impact.rename(columns={"display_name": "player"})
    return impact[["season", "gsis_id", "player", "position", "team", "plays", "pass_plays",
                   "on_field_epa_per_dropback", "on_field_epa_per_carry", "pass_impact", "run_impact"]]


def upload(tables):
    with connect("sys").begin() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))

    engine = connect(DB_NAME)
    for name, df in tables:
        print(f"Uploading {name}: {len(df):,} rows...")
        df.to_sql(name, engine, if_exists="replace", index=False, chunksize=5000, method="multi")

    # Confirm every row arrived
    with engine.connect() as conn:
        for name, df in tables:
            in_db = conn.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar()
            status = "OK" if in_db == len(df) else "MISMATCH"
            print(f"[{status}] {name}: {in_db:,} rows in database (expected {len(df):,})")


def main():
    if sys.argv[1:] == ["history"]:
        upload([("team_defense_history", build_history())])
    elif sys.argv[1:] == ["players"]:
        upload([("player_defense_season", build_player_season()), ("player_impact", build_player_impact())])
    else:
        plays, team_defense = build_tables()
        upload([("team_defense", team_defense), ("plays", plays)])


if __name__ == "__main__":
    main()
