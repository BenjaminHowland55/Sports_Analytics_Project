"""Upload the nflreadpy analysis tables to the project's TiDB Cloud (MySQL) database.

Creates (or replaces) two tables in the `nfl_analytics` database:
  plays         one row per regular-season run/pass play, 2023-2025, with QB pressure, blitz
                (5+ rushers), defensive package (base/nickel/dime by DB count) and man/zone coverage
  team_defense  one row per defense per season: the notebook's table (data/team_defense_2023_2025.csv)
                plus blitz, man-coverage and package rates

Run from this folder:  python load_to_database.py
The password is read from .env (key: DATABASE_PASSWORD) and is never printed.
"""
import certifi
import nflreadpy as nfl
import pandas as pd
from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

SEASONS = [2023, 2024, 2025]
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


def main():
    plays, team_defense = build_tables()

    with connect("sys").begin() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))

    engine = connect(DB_NAME)
    for name, df in [("team_defense", team_defense), ("plays", plays)]:
        print(f"Uploading {name}: {len(df):,} rows...")
        df.to_sql(name, engine, if_exists="replace", index=False, chunksize=5000, method="multi")

    # Confirm every row arrived
    with engine.connect() as conn:
        for name, df in [("team_defense", team_defense), ("plays", plays)]:
            in_db = conn.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar()
            status = "OK" if in_db == len(df) else "MISMATCH"
            print(f"[{status}] {name}: {in_db:,} rows in database (expected {len(df):,})")


if __name__ == "__main__":
    main()
