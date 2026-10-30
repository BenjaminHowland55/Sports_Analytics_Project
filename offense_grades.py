"""Grade every offensive skill player against his position room (QB, RB, WR, TE), the same way
position_grades.py grades defenders: each player's stats over this season and his two most recent healthy seasons,
pulled toward average by how much each stat repeats, weighted by what veteran contracts pay for, then rolled up
into room grades, cap spending and value for all 32 teams.

Offensive linemen aren't graded individually (the free data has no individual blocking stats, and grading them from
their team's results on their snaps ranked good lines rather than good linemen). Each team's line is graded as a unit
on pass and run blocking instead (line_unit_grades).

Run with:  python offense_grades.py   (about 10 minutes; uploads offense_player_grades, offense_room_grades,
offense_grade_stats and offense_line_grades)
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
    "QB": ("High", "Matched the team's EPA per dropback (r = 0.43, after taking out half the supporting cast's help)"),
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
               "epa", "receiver_player_id", "passer_player_id", "complete_pass", "air_yards", "rusher_player_id",
               "yards_gained", "success"]
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


# Offensive lines are graded as units (no individual blocking stats exist in the free data). Tested on every team
# 2018-2025 for how much each stat repeats from season to season and how it matches the team's EPA: sack rate allowed
# (repeat 0.37, matches pass EPA 0.54) and pressure rate allowed (0.41, 0.44) for pass blocking; adjusted line yards
# (0.30, matches run EPA 0.74), yards before contact (0.30, 0.58) and stuff rate (0.28, 0.53) for run blocking.
# Pressure-to-sack rate (mostly the quarterback) and rushing success rate (mostly the back's result) were left out.
# Pass blocking is adjusted for the quarterback's average time to throw (Next Gen Stats): a quarterback who holds the
# ball longer takes more pressure (r = 0.25) and sacks (0.23) whatever the line does. Each rate is compared with what
# the league-wide fit expects at that time to throw; the adjusted rates matched the team's passing EPA better
# (pressure 0.44 -> 0.47, sacks 0.54 -> 0.57) and predicted next season's slightly better. Adjusting run blocking for
# stacked (8+ defender) boxes changed nothing (they barely relate to these stats), so it isn't used.
LINE_PASS_STATS = {"pressure_rate_adj": (0.5, True), "sack_rate_adj": (0.5, True)}  # stat: (weight, lower is better)
LINE_RUN_STATS = {"adj_line_yards": (0.4, False), "yards_before_contact": (0.3, False), "stuff_rate": (0.3, True)}
LINE_POSITIONS = ["T", "G", "C", "OT", "OG", "OL"]


def line_results(plays, seasons):
    """Each offense's blocking results by season. Adjusted line yards credit a running back's run to the line in
    steps: losses count 120%, the first 4 yards 100%, yards 5-10 50%, and nothing beyond 10."""
    plays = plays[plays["play_type"].isin(["pass", "run"])]
    dropbacks = plays[plays["qb_dropback"] == 1]
    pressured = (dropbacks["sack"].fillna(0) + dropbacks["qb_hit"].fillna(0)) > 0
    positions = nfl.load_players().select(["gsis_id", "position"]).to_pandas().set_index("gsis_id")["position"]
    runs = plays[(plays["rush"] == 1) & (plays["qb_dropback"] == 0)]
    back_runs = runs[runs["rusher_player_id"].map(positions).isin(["RB", "FB"])]
    gained = back_runs["yards_gained"].clip(lower=-20)
    line_yards = np.where(gained < 0, 1.2 * gained, np.minimum(gained, 4) + 0.5 * np.clip(gained - 4, 0, 6))
    keys = lambda df: [df["season"], df["posteam"]]
    pfr = (nfl.load_pfr_advstats(seasons=seasons, stat_type="rush", summary_level="week")
           .filter(pl.col("game_type") == "REG").to_pandas())
    pfr_positions = (nfl.load_players().select(["pfr_id", "position"]).drop_nulls().unique("pfr_id").to_pandas()
                     .set_index("pfr_id")["position"])
    pfr = pfr[pfr["pfr_player_id"].map(pfr_positions).isin(["RB", "FB"])]
    contact = pfr.groupby(["season", "team"])[["rushing_yards_before_contact", "carries"]].sum()
    return pd.DataFrame({
        "pressure_rate": pressured.groupby(keys(dropbacks)).mean(),
        "sack_rate": dropbacks["sack"].groupby(keys(dropbacks)).mean(),
        "adj_line_yards": pd.Series(line_yards, index=back_runs.index).groupby(keys(back_runs)).mean(),
        "stuff_rate": (back_runs["yards_gained"] <= 0).groupby(keys(back_runs)).mean(),
        "yards_before_contact": contact["rushing_yards_before_contact"] / contact["carries"],
    }).rename_axis(["season", "team"])


def team_time_to_throw(seasons):
    """Each offense's average time to throw (seconds), weighting each quarterback's weeks by his attempts."""
    ngs = (nfl.load_nextgen_stats(seasons=seasons, stat_type="passing")
           .filter((pl.col("season_type") == "REG") & (pl.col("week") > 0)).to_pandas())
    ngs = ngs.assign(seconds=ngs["avg_time_to_throw"] * ngs["attempts"], team=ngs["team_abbr"].replace({"LAR": "LA"}))
    totals = ngs.groupby(["season", "team"])[["seconds", "attempts"]].sum()
    return (totals["seconds"] / totals["attempts"]).rename("time_to_throw")


def blocking_results(plays, seasons):
    """line_results plus pressure and sack rates relative to what the league-wide fit expects at each offense's time
    to throw."""
    results = line_results(plays, seasons).join(team_time_to_throw(seasons))
    for stat in ["pressure_rate", "sack_rate"]:
        known = results[[stat, "time_to_throw"]].dropna()
        slope = np.polyfit(known["time_to_throw"], known[stat], 1)[0]
        results[stat + "_adj"] = results[stat] - slope * (results["time_to_throw"] - known["time_to_throw"].mean())
    return results


def block_scores(results, year):
    """Each team's pass- and run-blocking scores in a season: weighted averages of the stats' z-scores."""
    teams = results.loc[year]
    z = (teams - teams.mean()) / teams.std()
    part = lambda stats: sum(w * (-z[s] if lower else z[s]) for s, (w, lower) in stats.items())
    return pd.DataFrame({"pass_block_score": part(LINE_PASS_STATS), "run_block_score": part(LINE_RUN_STATS)})


def line_unit_grades(results, season, contracts):
    """Every team's offensive line this season: pass blocking, run blocking and overall (their average), plus last
    season's ranks, what the line costs, and value for money."""
    scores = lambda year: block_scores(results, year)
    now, before = scores(season), scores(season - 1)
    table = results.loc[season].join(now)
    table["line_score"] = (table["pass_block_score"] + table["run_block_score"]) / 2
    rank = lambda col: col.rank(ascending=False, method="min").astype(int)
    for part in ["pass_block", "run_block"]:
        table[f"{part}_rank"] = rank(table[f"{part}_score"])
        table[f"{part}_rank_last_season"] = rank(before[f"{part}_score"]).reindex(table.index)
    table["line_rank"] = rank(table["line_score"])
    table["line_rank_last_season"] = rank((before["pass_block_score"] + before["run_block_score"]) / 2).reindex(table.index)

    # Cap spent: the contracts of every lineman who has taken offensive snaps for the team this season
    snaps = nfl.load_snap_counts(seasons=[season]).filter((pl.col("game_type") == "REG") & (pl.col("offense_snaps") > 0)
                                                          & pl.col("position").is_in(LINE_POSITIONS)).to_pandas()
    ids = nfl.load_players().select(["pfr_id", "gsis_id"]).drop_nulls().unique("pfr_id").to_pandas()
    linemen = (snaps.groupby(["team", "pfr_player_id"])["offense_snaps"].sum().reset_index()
               .merge(ids, left_on="pfr_player_id", right_on="pfr_id").merge(contracts, on="gsis_id", how="left"))
    table["cap_pct"] = linemen.groupby("team")["apy_cap_pct"].sum().reindex(table.index).fillna(0)
    table["apy_millions"] = linemen.groupby("team")["apy"].sum().reindex(table.index).fillna(0)
    slope, intercept = np.polyfit(table["cap_pct"], table["line_score"], 1)
    table["value_rank"] = rank(table["line_score"] - (intercept + slope * table["cap_pct"]))
    table["spend_rank"] = rank(table["cap_pct"])
    table["season"] = season
    return table.rename_axis("team").reset_index()


# Quarterbacks are graded partly relative to their supporting cast: the team's pass blocking and its receiving weapons
# (its WR/TE/RB single-season grades, weighted by targets). Across QB seasons since 2019, one standard deviation better
# weapons added about 0.27 to a quarterback's grade and better blocking about 0.21. Half of that expected boost is taken
# out: the full amount raised the match with quarterback pay (0.55 -> 0.69) but cost next-season prediction
# (0.45 -> 0.37), and receivers' grades partly reflect their quarterback; half raised the pay match to 0.61 for a small
# cost (prediction 0.42, team passing match 0.46 -> 0.43).
QB_CAST_ADJUSTMENT = 0.5


def supporting_cast(players, team_snaps, method, results, seasons):
    """Each team's receiving weapons and pass blocking by season, both put on the same scale (z-scores across all
    team-seasons)."""
    weights = {room: method[(method["room"] == room) & method["used"]].set_index("stat")["weight"]
               for room in ["WR", "TE", "RB"]}
    main_team = (team_snaps.sort_values("offense_snaps").drop_duplicates(["season", "pfr_id"], keep="last")
                 .set_index(["season", "pfr_id"])["team"])
    weapons = {}
    for year in seasons:
        rows = []
        for room, w in weights.items():
            _, z, _ = grading.score_window(players, method, room, year, single_season=True)
            season_rows = players[(players["season"] == year) & (players["room"] == room)].set_index("pfr_id")
            rows.append(pd.DataFrame({"grade": ((z[w.index] * w).sum(axis=1) / w.sum()).reindex(season_rows.index),
                                      "targets": season_rows["targets"],
                                      "team": main_team.loc[year].reindex(season_rows.index)}))
        rows = pd.concat(rows).dropna()
        rows = rows[rows["targets"] > 0]
        weapons[year] = rows.groupby("team").apply(lambda t: np.average(t["grade"], weights=t["targets"]),
                                                   include_groups=False)
    blocking = {year: block_scores(results, year)["pass_block_score"] for year in seasons}
    cast = pd.concat([pd.concat(weapons).rename("weapons"), pd.concat(blocking).rename("blocking")], axis=1)
    return (cast - cast.mean()) / cast.std(), main_team


def quarterback_help(players, team_snaps, method, results, season):
    """How much each quarterback's grade is expected to be raised by his supporting cast over his grading window
    (seasons weighted as in the grade), from a fit of quarterbacks' single-season grades on their casts."""
    seasons = [s for s in results.index.get_level_values("season").unique() if s >= grading.FIRST_SEASON + 1]
    cast, main_team = supporting_cast(players, team_snaps, method, results, seasons)
    w = method[(method["room"] == "QB") & method["used"]].set_index("stat")["weight"]
    rows = []
    for year in [s for s in seasons if s < season]:
        _, z, _ = grading.score_window(players, method, "QB", year, single_season=True)
        qbs = players[(players["season"] == year) & (players["room"] == "QB")
                      & (players["season_games"] >= grading.HEALTHY_GAMES)].set_index("pfr_id")
        rows.append(pd.DataFrame({"grade": ((z[w.index] * w).sum(axis=1) / w.sum()).reindex(qbs.index),
                                  "team": main_team.loc[year].reindex(qbs.index), "season": year})
                    .join(cast, on=["season", "team"]))
    fit = pd.concat(rows).dropna()
    _, weapons_effect, blocking_effect = np.linalg.lstsq(
        np.column_stack([np.ones(len(fit)), fit["weapons"], fit["blocking"]]), fit["grade"], rcond=None)[0]
    print(f"Supporting cast: +1 SD weapons = {weapons_effect:+.2f}, +1 SD blocking = {blocking_effect:+.2f} QB grade")
    window = grading.select_window(players[players["room"] == "QB"], season, "QB")
    window = window.assign(team=[main_team.get((s, p)) for s, p in zip(window["season"], window["pfr_id"])])
    window = window.join(cast, on=["season", "team"]).dropna(subset=["weapons", "blocking"])
    window["help"] = weapons_effect * window["weapons"] + blocking_effect * window["blocking"]
    return window.groupby("pfr_id").apply(lambda d: np.average(d["help"], weights=d["weight"]), include_groups=False)


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

    print("Measuring each quarterback's supporting cast...")
    results = blocking_results(plays, seasons)
    grading.COMPOSITE_ADJUSTMENTS = {
        "QB": QB_CAST_ADJUSTMENT * quarterback_help(players, team_snaps, method, results, season)}

    print(f"Grading {season} players and rooms...")
    grades = (grading.grade_players(players, team_snaps, method, season, grading.load_roster(season))
              .merge(contracts, on="gsis_id", how="left"))
    areas = team_area_results(plays, season)
    rooms_table = (grading.grade_rooms(grades, areas)
                   .merge(grading.season_only_rooms(players, team_snaps, method, season), on=["team", "room"],
                          how="left"))
    rooms_table["team_epa_rank"] = rooms_table["team"].map(areas["all_epa"].rank(ascending=False).astype(int))
    grades = grades.astype({"snaps": "int64", "games_played": "int64"})
    print("Grading offensive lines as units...")
    lines = line_unit_grades(results, season, contracts)
    upload([("offense_player_grades", grades), ("offense_room_grades", rooms_table),
            ("offense_grade_stats", grading.grading_caps(method)), ("offense_line_grades", lines)])


if __name__ == "__main__":
    main()
