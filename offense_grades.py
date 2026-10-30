"""Grade every offensive skill player against his position room (QB, RB, WR, TE), the same way
position_grades.py grades defenders: each player's stats over this season and his two most recent healthy seasons,
pulled toward average by how much each stat repeats, weighted by what veteran contracts pay for, then rolled up
into room grades, cap spending and value for all 32 teams.

Offensive linemen aren't graded: the free data has no individual blocking stats, and grading them from their team's
results on their snaps ranked good lines rather than good linemen.

Run with:  python offense_grades.py   (about 10 minutes; uploads offense_player_grades, offense_room_grades and
offense_grade_stats)
"""
import numpy as np
import pandas as pd
import polars as pl
import nflreadpy as nfl

import position_grades as grading
from load_to_database import upload

ROOMS = ["QB", "RB", "WR", "TE"]
ROOM_NAMES = {"QB": "Quarterbacks", "RB": "Running Backs", "WR": "Wide Receivers", "TE": "Tight Ends"}
POSITION_TO_ROOM = {"QB": "QB", "RB": "RB", "FB": "RB", "HB": "RB", "WR": "WR", "TE": "TE"}

# Each stat: (events column, opportunities column, minimum opportunities for the reliability test, lower is
# better, unused). Shares (carries, targets, air yards) are a player's piece of his team's total in the games he
# played, the offensive version of tackles: volume a team trusts him with.
STATS = {
    "epa_per_play": ("qb_epa", "qb_plays", 150, False, None),
    "cpoe": ("cpoe_x_attempts", "attempts", 150, False, None),
    "adot": ("passing_air_yards", "attempts", 150, False, None),
    "deep_completion_rate": ("deep_completions", "deep_attempts", 30, False, None),
    "pass_td_rate": ("passing_tds", "attempts", 150, False, None),
    "int_rate": ("passing_interceptions", "attempts", 150, True, None),
    "sack_rate": ("sacks_suffered", "dropbacks", 150, True, None),
    "bad_throw_rate": ("bad_throws", "pfr_pass_attempts", 150, True, None),
    "explosive_pass_rate": ("passing_20", "attempts", 150, False, None),
    "rush_epa": ("rushing_epa", "carries", 50, False, None),
    "rush_first_down_rate": ("rushing_first_downs", "carries", 50, False, None),
    "yards_after_contact": ("yards_after_contact", "pfr_carries", 50, False, None),
    "broken_tackle_rate": ("rush_broken_tackles", "pfr_carries", 50, False, None),
    "explosive_run_rate": ("rushing_20", "carries", 50, False, None),
    "carry_share": ("carries", "team_carries", 100, False, None),
    "scrimmage_yards_per_snap": ("rb_yards", "team_offense_snaps", 300, False, None),
    "fumble_rate": ("fumbles_lost", "touches", 100, True, None),
    "scrimmage_td_rate": ("scrimmage_tds", "touches", 100, False, None),
    "rec_epa": ("receiving_epa", "targets", 30, False, None),
    "yards_per_target": ("receiving_yards", "targets", 30, False, None),
    "target_share": ("targets", "team_targets", 100, False, None),
    "air_yards_share": ("receiving_air_yards", "team_air_yards", 500, False, None),
    "drop_rate": ("drops", "pfr_targets", 30, True, None),
    "yac_per_catch": ("receiving_yards_after_catch", "receptions", 20, False, None),
    "rec_broken_tackle_rate": ("rec_broken_tackles", "pfr_receptions", 20, False, None),
    "rec_first_down_rate": ("receiving_first_downs", "targets", 30, False, None),
    "rec_td_rate": ("receiving_tds", "targets", 30, False, None),
    "explosive_catch_rate": ("receiving_20", "targets", 30, False, None),
    "snap_share": ("offense_snaps", "team_offense_snaps", 100, False, None),
}
RECEIVER_STATS = ["rec_epa", "yards_per_target", "target_share", "air_yards_share", "drop_rate", "yac_per_catch",
                  "rec_broken_tackle_rate", "rec_first_down_rate", "rec_td_rate", "explosive_catch_rate",
                  "snap_share"]
ROOM_STATS = {
    # No snap share: starters play nearly every snap, so it mostly measures being rested in blowouts. EPA per play
    # combines passing and rushing over all of a quarterback's plays, so running counts by its real volume (graded
    # separately, pay gave rushing twice the weight of passing).
    # Depth of target and deep-ball accuracy raised the pay match (0.53 -> 0.55), the backtest (0.48 -> 0.49) and
    # next-season prediction (0.45 -> 0.46). Clutch stats were tested and left out: EPA late in close games barely
    # repeats and lowered the pay match, EPA on third and fourth down got no pay weight, and win probability added
    # made every check worse (it mostly reflects which moments a quarterback happened to play in).
    "QB": ["epa_per_play", "cpoe", "pass_td_rate", "int_rate", "sack_rate", "bad_throw_rate", "explosive_pass_rate",
           "adot", "deep_completion_rate"],
    # Scrimmage yards per team snap (in games he played) credits production, not just per-carry efficiency, which
    # favours backs sharing the load on fresher legs: pay match 0.78 -> 0.79, prediction of next season's scrimmage
    # yards 0.72 -> 0.73, backtest unchanged. Dropping rushing EPA, adding touch share, and guaranteeing production a
    # fixed share of the grade were also tested; none improved the checks, and the last made the backtest worse.
    "RB": ["snap_share", "rush_epa", "rush_first_down_rate", "yards_after_contact", "broken_tackle_rate",
           "explosive_run_rate", "carry_share", "target_share", "rec_epa", "fumble_rate", "scrimmage_td_rate",
           "scrimmage_yards_per_snap"],
    "WR": RECEIVER_STATS,
    "TE": RECEIVER_STATS,
}
# Kept although it repeats less than the minimum (r = 0.12), because veteran contracts pay for it (22% of running
# back pay, 90% range 13-27%) and adding it held the 2020-2025 backtest level while the pay match rose 0.75 -> 0.77.
# Running back touchdowns and fumbles, receiver drops and tight end touchdowns were tested the same way: pay gave them
# almost nothing, or the backtest got worse.
KEPT_DESPITE_LOW_REPEAT = {("RB", "rush_epa")}
# Also tested and left out: quarterbacks' interception-worthy throw rate (FTN charting; a small gain on every check,
# left out by choice). And because they didn't improve the checks: quarterbacks' catchable-ball rate, pressure-to-sack
# rate, time to throw and QB-fault sack rate; running backs' rush yards over expected and success rate; receivers'
# and tight ends' separation, yards after the catch over expected, contested catches and created receptions.
# Capping snap share's weight changed nothing for these rooms, so it isn't capped.
SNAP_SHARE_CAP = {room: None for room in ROOMS}
# Quarterbacks: efficiency (EPA per play and completion % over expected) is guaranteed half the grade, the rest
# keeping the market's proportions. Pay alone put about 60% on touchdown, interception and sack rates, which swing
# from season to season and lean on the line and play-calling. Elite quarterbacks are
# reliably far above the pack (see CAP_BELOW_REPEAT). This version predicted a quarterback's next-season EPA per
# play best of those tried (r = 0.45). Adjusting EPA for pressure faced and receivers' yards after the catch was also
# tested and dropped: it predicted worse (0.41-0.43) and matched team results worse.
GUARANTEED_SHARES = {"QB": (["epa_per_play", "cpoe"], 0.50)}
# Offense caps only its noisiest stats (repeat r below 0.20: interception rate, rushing EPA, fumbles, touchdown rates)
# at +/-2. Capping every stat tied the best players on stats they're reliably far ahead on (a tight end with 29% of his
# team's targets got the same credit as one with 21%); most offensive per-play stats repeat around 0.2-0.4, so the
# defense's 0.40 cut-off would cap nearly all of them and lowered the quarterback pay match (0.55 -> 0.52). With
# 0.20, every check matched no cap at all.
CAP_BELOW_REPEAT = 0.20
# Quarterbacks weight this season a little less (2.5 instead of 3): seven games is a small sample for the most
# important position. It raised the pay match with no change in prediction or the backtest.
ROOM_SEASON_WEIGHTS = {"QB": [2.5, 2, 1, 0.5, 0.25]}
# This season's team result in each room's main area, and whether a higher number is better
ROOM_AREA = {
    "QB": ("pass_epa", "EPA per dropback", True),
    "RB": ("run_epa", "EPA per designed run", True),
    "WR": ("wr_epa", "EPA per target to WRs", True),
    "TE": ("te_epa", "EPA per target to TEs", True),
}
# Backtest 2020-2025: each room graded from earlier seasons only, using that season's roster, then compared with the
# team's result that season (average correlation across the six seasons)
CONFIDENCE = {
    "QB": ("High", "Matched the team's EPA per dropback (r = 0.49) and overall offense (r = 0.48)"),
    "RB": ("Fair", "Matched the team's run EPA (r = 0.13) and overall offense (r = 0.14)"),
    "WR": ("Moderate", "Matched EPA on throws to receivers (r = 0.32) and overall offense (r = 0.28)"),
    "TE": ("Moderate", "Matched EPA on throws to tight ends (r = 0.30) and overall offense (r = 0.22)"),
}


def use_offense_settings():
    """Point the shared grading functions in position_grades.py at the offensive rooms, stats and columns."""
    for name, value in {"ROOMS": ROOMS, "ROOM_NAMES": ROOM_NAMES, "STATS": STATS, "ROOM_STATS": ROOM_STATS,
                        "KEPT_DESPITE_LOW_REPEAT": KEPT_DESPITE_LOW_REPEAT, "SNAP_SHARE_CAP": SNAP_SHARE_CAP,
                        "GUARANTEED_SHARES": GUARANTEED_SHARES, "ROOM_Z_CAP": {},
                        "ROOM_SEASON_WEIGHTS": ROOM_SEASON_WEIGHTS, "CAP_BELOW_REPEAT": CAP_BELOW_REPEAT,
                        "ROOM_AREA": ROOM_AREA, "CONFIDENCE": CONFIDENCE, "SNAPS": "offense_snaps",
                        "TEAM_SNAPS": "team_offense_snaps",
                        "SEASON_COUNTS": {"scrimmage_yards": "season_yards", "total_tds": "season_tds"}}.items():
        setattr(grading, name, value)


def load_offense_plays(seasons):
    columns = ["season", "game_id", "posteam", "play_type", "qb_dropback", "sack", "qb_hit", "rush", "pass_attempt",
               "epa", "receiver_player_id", "passer_player_id", "complete_pass", "air_yards"]
    return (nfl.load_pbp(seasons=seasons).filter(pl.col("season_type") == "REG").select(columns).to_pandas())


def deep_throws(plays):
    """Each passer's throws of 20+ air yards in each game, and how many were caught."""
    deep = plays[(plays["pass_attempt"] == 1) & (plays["air_yards"] >= 20) & plays["passer_player_id"].notna()]
    return (deep.assign(deep_completions=(deep["complete_pass"] == 1).astype(int))
            .groupby(["game_id", "passer_player_id"])
            .agg(deep_attempts=("deep_completions", "size"), deep_completions=("deep_completions", "sum"))
            .reset_index().rename(columns={"passer_player_id": "gsis_id"}))


def build_player_seasons(plays, seasons, rooms):
    """One row per player-season with every stat's events and opportunities, from game-level data."""
    all_snaps = nfl.load_snap_counts(seasons=seasons).filter(pl.col("game_type") == "REG").to_pandas()
    ids = (nfl.load_players().select(["pfr_id", "gsis_id", "display_name", "birth_date", "position"])
           .drop_nulls("pfr_id").unique("pfr_id").to_pandas())
    games = (all_snaps[all_snaps["offense_snaps"] > 0]
             .merge(ids[["pfr_id", "gsis_id", "display_name", "birth_date"]], left_on="pfr_player_id",
                    right_on="pfr_id", how="left").drop(columns="pfr_player_id"))
    
    # Box-score stats by game, and team totals in each game (for carry, target and air-yard shares)
    box = (nfl.load_player_stats(seasons=seasons, summary_level="week").filter(pl.col("season_type") == "REG")
           .to_pandas().rename(columns={"player_id": "gsis_id"}))
    box["cpoe_x_attempts"] = box["passing_cpoe"].fillna(0) * box["attempts"]
    box_columns = ["attempts", "completions", "passing_yards", "passing_air_yards", "passing_tds", "passing_interceptions",
                   "sacks_suffered", "passing_epa", "cpoe_x_attempts", "passing_20", "carries", "rushing_yards",
                   "rushing_tds", "rushing_first_downs", "rushing_epa", "rushing_20", "receptions", "targets",
                   "receiving_yards", "receiving_tds", "receiving_air_yards", "receiving_yards_after_catch",
                   "receiving_first_downs", "receiving_epa", "receiving_20", "rushing_fumbles_lost",
                   "receiving_fumbles_lost", "sack_fumbles_lost"]
    team_totals = (box.groupby(["game_id", "team"])[["carries", "targets", "receiving_air_yards"]].sum()
                   .rename(columns={"carries": "team_carries", "targets": "team_targets",
                                    "receiving_air_yards": "team_air_yards"}).reset_index())
    games = (games.merge(box[["game_id", "gsis_id"] + box_columns], on=["game_id", "gsis_id"], how="left")
             .merge(team_totals, on=["game_id", "team"], how="left"))

    # Pro Football Reference charting (drops, bad throws, contact, broken tackles); its denominators only count
    # charted games, so each rate's top and bottom cover the same games
    def pfr(stat_type, columns):
        return (nfl.load_pfr_advstats(seasons=seasons, stat_type=stat_type, summary_level="week")
                .filter(pl.col("game_type") == "REG").select(["game_id", "pfr_player_id"] + list(columns))
                .to_pandas().rename(columns=columns).rename(columns={"pfr_player_id": "pfr_id"})
                .drop_duplicates(["game_id", "pfr_id"]).assign(**{f"charted_{stat_type}": 1}))
    games = (games.merge(pfr("pass", {"passing_bad_throws": "bad_throws"}), on=["game_id", "pfr_id"], how="left")
             .merge(pfr("rush", {"rushing_yards_after_contact": "yards_after_contact",
                                 "rushing_broken_tackles": "rush_broken_tackles"}), on=["game_id", "pfr_id"], how="left")
             .merge(pfr("rec", {"receiving_drop": "drops", "receiving_broken_tackles": "rec_broken_tackles"}),
                    on=["game_id", "pfr_id"], how="left"))
    for stat_type, raw, charted in [("pass", "attempts", "pfr_pass_attempts"), ("rush", "carries", "pfr_carries"),
                                    ("rec", "targets", "pfr_targets"), ("rec", "receptions", "pfr_receptions")]:
        games[charted] = games[raw].fillna(0) * games[f"charted_{stat_type}"].fillna(0)

    games = games.merge(deep_throws(plays), on=["game_id", "gsis_id"], how="left")

    count_columns = box_columns + ["team_carries", "team_targets", "team_air_yards", "bad_throws",
                                   "yards_after_contact", "rush_broken_tackles", "drops", "rec_broken_tackles",
                                   "pfr_pass_attempts", "pfr_carries", "pfr_targets", "pfr_receptions",
                                   "deep_attempts", "deep_completions"]
    games[count_columns] = games[count_columns].fillna(0)
    players = (games.sort_values("week")
               .groupby(["season", "pfr_id"])
               .agg(gsis_id=("gsis_id", "first"), player=("display_name", "first"), birth_date=("birth_date", "first"),
                    roster_position=("position", "last"), offense_snaps=("offense_snaps", "sum"),
                    season_games=("game_id", "nunique"), **{c: (c, "sum") for c in count_columns})
               .reset_index())
    players["player"] = players["player"].fillna(players["pfr_id"])
    birth = pd.to_datetime(players["birth_date"], errors="coerce")
    players["age"] = ((pd.to_datetime(players["season"].astype(str) + "-09-01") - birth).dt.days // 365.25)
    players["dropbacks"] = players["attempts"] + players["sacks_suffered"]
    players["qb_epa"] = players["passing_epa"] + players["rushing_epa"]
    players["qb_plays"] = players["dropbacks"] + players["carries"]
    players["touches"] = players["carries"] + players["receptions"]
    players["fumbles_lost"] = (players["rushing_fumbles_lost"] + players["receiving_fumbles_lost"]
                               + players["sack_fumbles_lost"])
    players["scrimmage_tds"] = players["rushing_tds"] + players["receiving_tds"]
    players["rb_yards"] = players["rushing_yards"] + players["receiving_yards"]
    players["scrimmage_yards"] = players["passing_yards"] + players["rushing_yards"] + players["receiving_yards"]
    players["total_tds"] = players["passing_tds"] + players["rushing_tds"] + players["receiving_tds"]

    # Room: contract position, else roster position
    players = players.drop(columns="birth_date").merge(rooms, left_on="gsis_id", right_index=True, how="left")
    players["room"] = players["room"].map(POSITION_TO_ROOM).fillna(players["roster_position"].map(POSITION_TO_ROOM))
    players = players[players["room"].isin(ROOMS)].copy()

    # Snap share over the games he dressed for (played on offense or special teams)
    dressed = all_snaps.copy()
    dressed["game_offense_snaps"] = dressed.groupby(["game_id", "team"])["offense_snaps"].transform("max")
    dressed = dressed[(dressed["offense_snaps"] > 0) | (dressed["st_snaps"] > 0)]
    team_offense_snaps = (dressed.groupby(["season", "pfr_player_id"])["game_offense_snaps"].sum()
                          .rename("team_offense_snaps").rename_axis(["season", "pfr_id"]))
    players = players.merge(team_offense_snaps, left_on=["season", "pfr_id"], right_index=True, how="left")

    team_snaps = (games.groupby(["season", "team", "pfr_id"])
                  .agg(offense_snaps=("offense_snaps", "sum"), games_played=("game_id", "nunique")).reset_index()
                  .merge(all_snaps.groupby(["season", "team"])["game_id"].nunique().rename("team_games"),
                         left_on=["season", "team"], right_index=True, how="left"))
    return players, team_snaps


def team_area_results(plays, season):
    """This season's team result in each offensive room's main area."""
    plays = plays[(plays["season"] == season) & plays["epa"].notna() & plays["play_type"].isin(["pass", "run"])]
    positions = nfl.load_players().select(["gsis_id", "position"]).to_pandas().set_index("gsis_id")["position"]
    targets = plays[(plays["pass_attempt"] == 1) & (plays["sack"] == 0) & plays["receiver_player_id"].notna()]
    receiver = targets["receiver_player_id"].map(positions)
    runs = plays[(plays["rush"] == 1) & (plays["qb_dropback"] == 0)]
    dropbacks = plays[plays["qb_dropback"] == 1]
    by_team = lambda df: df.groupby("posteam")["epa"].mean()
    return pd.DataFrame({
        "pass_epa": by_team(dropbacks),
        "run_epa": by_team(runs),
        "wr_epa": by_team(targets[receiver == "WR"]),
        "te_epa": by_team(targets[receiver == "TE"]),
        "all_epa": by_team(plays),
    }).rename_axis("team")


def main():
    use_offense_settings()
    season = nfl.get_current_season()
    seasons = list(range(grading.FIRST_SEASON, season + 1))
    print(f"Downloading {grading.FIRST_SEASON}-{season} plays, snaps, player stats and contracts from nflverse...")
    plays = load_offense_plays(seasons)
    contracts, rooms, veteran_contracts = grading.load_contracts()
    players, team_snaps = build_player_seasons(plays, seasons, rooms)

    print("Testing which stats repeat year to year...")
    method = grading.estimate_reliability(players, season - 1)
    print("Measuring what the NFL pays for in each room...")
    method = grading.market_weights(players, method, veteran_contracts, season)
    print(method[["room", "stat", "year_to_year_r", "used", "weight_share", "share_low", "share_high"]]
          .round(3).to_string(index=False))

    print(f"Grading {season} players and rooms...")
    grades = (grading.grade_players(players, team_snaps, method, season, grading.load_roster(season))
              .merge(contracts, on="gsis_id", how="left"))
    areas = team_area_results(plays, season)
    rooms_table = (grading.grade_rooms(grades, areas)
                   .merge(grading.season_only_rooms(players, team_snaps, method, season), on=["team", "room"],
                          how="left"))
    rooms_table["team_epa_rank"] = rooms_table["team"].map(areas["all_epa"].rank(ascending=False).astype(int))
    grades = grades.astype({"snaps": "int64", "games_played": "int64"})
    upload([("offense_player_grades", grades), ("offense_room_grades", rooms_table),
            ("offense_grade_stats", grading.grading_caps(method))])


if __name__ == "__main__":
    main()
