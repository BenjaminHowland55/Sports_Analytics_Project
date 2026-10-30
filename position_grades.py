"""Grade every team's five defensive position rooms for the current season, and compare what each room costs.

Rooms follow Over The Cap's contract positions: ED (edge), IDL (interior line), LB, CB, S.

How a player is graded (every number is recomputed on each run):
  1. Opportunities, game by game:  pass snaps = defensive snaps x the team's dropback share that game,
     run snaps = defensive snaps - pass snaps, coverage snaps = pass snaps - blitzes.
  2. Each stat is events / opportunities (e.g. pressures / pass snaps). A stat is used for a room only if
     it measures that room's job and repeats from one season to the next (year-to-year r >= MIN_RELIABILITY).
  3. Blended rate over the last three seasons, weighted 3-2-1 (most recent first), pulled toward the
     league rate:   (sum w*events + k*league rate) / (sum w*opportunities + k),   k = n * (1 - r) / r
     where r is the stat's year-to-year correlation at an average sample of n opportunities.
  4. z = (blended rate - room average) / room standard deviation (average and spread from the players with
     the most opportunities); the sign is flipped for lower-is-better stats so a positive z is always good,
     and z is capped at +/- Z_CAP so one stat can't dominate.
  5. Weight = what the NFL pays for. For every veteran contract signed since 2019 (rookie deals left out),
     the player's z-scores from the three seasons before signing are compared with the contract's yearly value
     as a share of the cap, holding age fixed (regression, weights kept non-negative). Snap share is one of the
     stats, so playing time counts. A stat's weight is the extra share of the cap paid for being one standard
     deviation better at it. Snap share is capped in some rooms, and lockdown coverage gets
     CORNER_LOCKDOWN_SHARE of the corner grade (see set_weights).
  6. Composite = sum(weight * z) / sum(weight).
  7. Player grade (0-100) = percentile of the composite among this season's regulars in the same room.
  (Each stat's impact in expected points is still measured and stored for comparison with the market.)

How a room is graded:
  room score  = this season's snap-weighted average composite (players with no history count as average)
  room spend  = sum of the players' contract APY as a share of the salary cap in the year signed
  value       = room score - the score expected for that spend (straight-line fit across the 32 teams)

Creates (or replaces) three tables in nfl_analytics: player_grades, room_grades, grade_stats.
Run from this folder:  python position_grades.py   (~5-10 minutes)
"""
import nflreadpy as nfl
import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import nnls

from load_to_database import upload

FIRST_SEASON = 2018  # Pro Football Reference's charted stats start here
# A player's grade uses this season and his two most recent healthy seasons before it. Each season's weight
# depends on how many seasons ago it was (this season, last season, 2 ago, 3 ago, 4 ago). Weights also set how
# much each season counts as evidence when pulling a player toward average (a season 2 ago counts a third as much
# as this one), so a player whose best play is years old is pulled further toward average.
SEASON_WEIGHTS = [3, 2, 1, 0.5, 0.25]
# Rooms with their own season weights (offense_grades.py gives quarterbacks a lighter current season); none on defense
ROOM_SEASON_WEIGHTS = {}
SEASONS_IN_WINDOW = 3
HEALTHY_GAMES = 8  # games on defense for a past season to count as healthy
LOOKBACK_SEASONS = 4  # how far back to look for healthy seasons
# Trend: players over YOUNG_AGE are Improving if their grade rose every season, Declining if it fell every season,
# Steady otherwise. Young players get room on the way down (the typical season-to-season change is 20 points):
# any rise counts as improvement, and so does a season his snap share rose by YOUNG_PLAYING_TIME_RISE (coaches
# trusting him with more) unless his grade fell by more than YOUNG_DECLINE_MARGIN; a smaller drop counts as no
# change. A young player is Improving with any improvement and no real drop.
YOUNG_AGE = 25
YOUNG_DECLINE_MARGIN = 20
YOUNG_PLAYING_TIME_RISE = 0.25
MIN_RELIABILITY = 0.15
REGULAR_SNAPS = 100  # this season's snaps needed to be in the 0-100 player grade comparison
# Players under REGULAR_SNAPS this season (hurt, or not yet back) stay listed if they have this many snaps over
# the three-season window and are on a roster, so an injury doesn't drop a proven player from the rankings
LIMITED_MIN_WINDOW_SNAPS = 800
MARKET_MIN_SNAPS = 300  # snaps over the three seasons before signing for a contract to count in the market study
# z-scores are capped at +/- this many standard deviations, so one stat can't dominate a grade. Noisy stats are
# pulled close together, so dividing by their small spread can turn a few lucky plays (six forced fumbles)
# into a +3.5 outlier; the cap left every room's match with veteran pay the same or better.
Z_CAP = 2.0
# The cap applies only to noisy stats, ones that repeat from season to season less than this (year-to-year r), like
# forced fumbles and interceptions; reliable ones (snap share, stops, pressures) can show how far ahead a player is.
# Every room's checks matched capping every stat; lower cut-offs, or no cap at all, hurt linebackers and safeties.
# None = cap every stat.
CAP_BELOW_REPEAT = 0.40
# Amounts subtracted from a room's composite, by pfr_id (offense_grades.py sets quarterbacks' supporting cast); none on
# defense
COMPOSITE_ADJUSTMENTS = {}
# (room, stat) pairs never capped even if they're under the cut-off. Edge rushers' sack rate (r = 0.22) is the core of
# the job: uncapped, the pay match rose 0.81 -> 0.82 and the team sack-rate backtest 0.19 -> 0.20, with next-season
# prediction within noise (0.63 -> 0.62). Also uncapping tackles for loss, or every edge stat but forced fumbles and
# missed tackles, added nothing.
NEVER_CAPPED = {("ED", "sack_rate")}
MARKET_BOOTSTRAPS = 200
# Stats kept although they repeat less than MIN_RELIABILITY, because two separate checks back them: veteran
# contracts pay for them, and adding them made the room's grades match team results better in the 2020-2025
# backtest. Linebacker completion % allowed: market weight 25% (90% range 13-33%); LB grade vs. EPA on throws to
# tight ends and running backs r = 0.19 -> 0.22. Their large k keeps small samples close to average.
# Forced fumbles (worth about 3 expected points; the offense loses the ball 55% of the time): market weight
# 6-29% depending on the room, with every bootstrap range above zero; adding them improved the edge, interior
# and safety backtests and moved the corner one within noise. Linebackers pass the repeat test on their own.
KEPT_DESPITE_LOW_REPEAT = {("LB", "completion_rate"), ("ED", "forced_fumble_rate"), ("IDL", "forced_fumble_rate"),
                           ("CB", "forced_fumble_rate"), ("S", "forced_fumble_rate")}
# Corners: interceptions swing a lot from season to season, so rather than follow pay alone, lockdown coverage
# (not allowing yards or completions) gets this share of the corner grade. The market alone gives it about 31%.
# Within the lockdown stats, and within the rest, stats keep the market's proportions.
CORNER_LOCKDOWN_STATS = ["yards_per_coverage_snap", "completion_rate"]
CORNER_LOCKDOWN_SHARE = 0.50
# Stats guaranteed a share of a room's grade (after snap share), the rest keeping the market's proportions
GUARANTEED_SHARES = {"CB": (CORNER_LOCKDOWN_STATS, CORNER_LOCKDOWN_SHARE)}
# Rooms with their own z-score cap, or None for no cap (offense_grades.py uses this); none on defense
ROOM_Z_CAP = {}
# Snap share (how much of the defense's snaps a player is trusted to play in the games he dressed for) is what coaches
# who watch every snap think, and a big part of pay. Uncapped, it put the best-paid players at the top in every
# room (rank match with veteran pay: edge 0.56 -> 0.84, corners 0.23 -> 0.66, linebackers 0.17 -> 0.68) and
# improved the edge and corner backtests, but it crowded out the stats that separate linebackers and safeties on
# the field. So its weight is capped where the 2020-2025 backtest started to drop (None = no cap). Edge rushers:
# the market gives it 44%, which let a team's rotation (Houston splitting snaps between its edge rushers) outweigh
# sacks and tackles for loss; capping it at 35% slightly improved the backtest (sack rate 0.19 -> 0.20, overall
# 0.23 -> 0.24) for a small drop in the pay match (0.85 -> 0.81); lower caps kept costing pay match for no gain.
SNAP_SHARE_CAP = {"ED": 0.35, "IDL": 0.30, "LB": 0.20, "CB": None, "S": 0.20}
ROOMS = ["ED", "IDL", "LB", "CB", "S"]
# Columns the shared grading functions read; offense_grades.py points these at its own columns
SNAPS = "defense_snaps"
TEAM_SNAPS = "team_defense_snaps"
SEASON_COUNTS = {"tackles": "season_tackles", "stops": "season_stops"}  # this season's totals shown on the page
ROOM_NAMES = {"ED": "Edge Rushers", "IDL": "Interior Line", "LB": "Linebackers", "CB": "Cornerbacks", "S": "Safeties"}

# Each stat: (events column, opportunities column, minimum opportunities for the reliability test,
#             lower is better, the event whose expected-points value sets its impact)
STATS = {
    "stop_rate": ("stops", "expected_stops", 8, False, "stop"),
    "sack_rate": ("sacks", "pass_snaps", 150, False, "sack"),
    "non_sack_pressure_rate": ("non_sack_pressures", "pass_snaps", 150, False, "non_sack_pressure"),
    "pressure_per_blitz": ("pressures", "blitzes", 20, False, "pressure"),
    "tfl_rate": ("tackles_for_loss", "run_snaps", 100, False, "tackle_for_loss"),
    "missed_tackle_rate": ("missed_tackles", "tackle_attempts", 20, True, "missed_tackle"),
    "yards_per_coverage_snap": ("yards_allowed", "coverage_snaps", 150, True, "coverage_yard"),
    "completion_rate": ("completions_allowed", "targets", 25, True, "completion"),
    "interception_rate": ("interceptions", "targets", 25, False, "interception"),
    "breakup_rate": ("breakups", "targets", 25, False, "breakup"),
    "forced_fumble_rate": ("forced_fumbles", "defense_snaps", 300, False, "forced_fumble"),
    "snap_share": ("defense_snaps", "team_defense_snaps", 100, False, "snap"),
}
# Stats that describe each room's job. Sacks and other pressures are separate so a sack isn't counted twice,
# and stops leave out tackles for loss for the same reason. Pressure rate is left out for LB/CB/S because it
# mostly reflects how often they are sent on blitzes (a coaching choice), not skill; pressures per blitz
# covers that instead. How much each stat counts comes from what NFL contracts pay for (market_weights).
ROOM_STATS = {
    "ED": ["snap_share", "sack_rate", "non_sack_pressure_rate", "tfl_rate", "stop_rate", "forced_fumble_rate",
           "missed_tackle_rate"],
    "IDL": ["snap_share", "sack_rate", "non_sack_pressure_rate", "tfl_rate", "stop_rate", "forced_fumble_rate",
            "missed_tackle_rate"],
    "LB": ["snap_share", "stop_rate", "pressure_per_blitz", "tfl_rate", "forced_fumble_rate", "missed_tackle_rate",
           "yards_per_coverage_snap", "completion_rate", "interception_rate", "breakup_rate"],
    "CB": ["snap_share", "yards_per_coverage_snap", "completion_rate", "interception_rate", "breakup_rate",
           "forced_fumble_rate", "stop_rate", "missed_tackle_rate"],
    "S": ["snap_share", "yards_per_coverage_snap", "completion_rate", "interception_rate", "breakup_rate",
          "forced_fumble_rate", "stop_rate", "tfl_rate", "missed_tackle_rate"],
}
# Play-by-play tackle credits: a solo tackle or a tackle with assistance counts 1, an assist counts 0.5
FULL_TACKLE_COLUMNS = ["solo_tackle_1_player_id", "solo_tackle_2_player_id",
                       "tackle_with_assist_1_player_id", "tackle_with_assist_2_player_id"]
ASSIST_COLUMNS = ["assist_tackle_1_player_id", "assist_tackle_2_player_id",
                  "assist_tackle_3_player_id", "assist_tackle_4_player_id"]
# Backtest 2020-2025: each room graded from earlier seasons only, using that season's roster, then compared
# with the team's result that season (average correlation across the six seasons)
CONFIDENCE = {
    "ED": ("Moderate", "Matched overall EPA allowed (r = 0.24) and team sack rate (r = 0.20)"),
    "IDL": ("Moderate", "Matched team sack rate (r = 0.24)"),
    "LB": ("Fair", "Matched overall EPA allowed (r = 0.14) and throws to tight ends and running backs (r = 0.13)"),
    "CB": ("Moderate", "Matched overall EPA allowed (r = 0.25) and outside throws (r = 0.21)"),
    "S": ("Low", "Matched overall EPA allowed (r = 0.16) and run defense (r = 0.13) only weakly"),
}
# This season's team result in each room's main area, and whether a higher number is better
ROOM_AREA = {
    "ED": ("sack_rate", "Sacks per dropback", True),
    "IDL": ("run_epa", "EPA per designed run allowed", False),
    "LB": ("te_rb_epa", "EPA per target to TEs and RBs", False),
    "CB": ("wr_outside_epa", "EPA per outside target to WRs", False),
    "S": ("deep_epa", "EPA per deep target (15+ air yards)", False),
}
# Roster positions -> room, for the few players with no contract on file
POSITION_TO_ROOM = {"DE": "ED", "OLB": "ED", "DT": "IDL", "NT": "IDL", "DL": "IDL", "LB": "LB", "ILB": "LB",
                    "MLB": "LB", "CB": "CB", "DB": "CB", "S": "S", "FS": "S", "SS": "S", "SAF": "S"}


def load_plays(seasons):
    """Regular-season run/pass plays, keeping only the columns needed here."""
    return pd.concat([
        nfl.load_pbp(seasons=[season])
           .select(["game_id", "play_id", "season", "season_type", "defteam", "play_type", "qb_dropback",
                    "pass_attempt", "sack", "interception", "complete_pass", "incomplete_pass", "yards_gained",
                    "pass_location", "air_yards", "receiver_player_id", "epa", "success", "fumble_forced"]
                   + FULL_TACKLE_COLUMNS + ASSIST_COLUMNS)
           .filter((pl.col("season_type") == "REG") & pl.col("play_type").is_in(["pass", "run"]))
           .with_columns(pl.col("play_id").cast(pl.Int64))
           .to_pandas()
        for season in seasons
    ], ignore_index=True)


def tackled_plays(plays):
    """Plays that ended in a tackle (not sacks), and whether each was a stop: the offense lost expected points
    (EPA <= 0) without losing yards. Tackles for loss are their own stat, so they aren't stops."""
    tackled = plays[plays[FULL_TACKLE_COLUMNS + ASSIST_COLUMNS].notna().any(axis=1)
                    & (plays["sack"] == 0) & plays["epa"].notna()].copy()
    tackled["stop"] = (tackled["success"] == 0) & (tackled["yards_gained"] >= 0)
    return tackled


def stops_by_game(plays):
    """Each player's stops in each game on run plays and on pass plays, with tackle credit (1 for a tackle,
    0.5 for an assist)."""
    tackled = tackled_plays(plays)
    credits = pd.concat([
        tackled.loc[tackled[column].notna(), ["game_id", column, "stop", "qb_dropback"]]
               .rename(columns={column: "gsis_id"}).assign(credit=1.0 if column in FULL_TACKLE_COLUMNS else 0.5)
        for column in FULL_TACKLE_COLUMNS + ASSIST_COLUMNS
    ], ignore_index=True)
    credits["stops"] = credits["credit"] * credits["stop"]
    credits["kind"] = np.where(credits["qb_dropback"] == 1, "pass_stops", "run_stops")
    return (credits.pivot_table(index=["game_id", "gsis_id"], columns="kind", values="stops", aggfunc="sum",
                                fill_value=0).reset_index().rename_axis(columns=None))


def measure_event_values(plays, seasons):
    """Expected points each defensive event is worth, from every completed season's plays.

    Each value is the swing against what usually happens instead, e.g. a sack against a clean-pocket dropback,
    an interception against an average target. A missed tackle has no play-by-play flag, so its cost comes from
    a team-game regression: EPA allowed on missed tackles, holding plays faced and tackles made fixed."""
    plays = plays[plays["season"].isin(seasons) & plays["epa"].notna()]
    pressure = pd.concat([nfl.load_participation(seasons=[s]).select(["nflverse_game_id", "play_id", "was_pressure"])
                          .with_columns(pl.col("play_id").cast(pl.Int64)).to_pandas() for s in seasons])
    dropbacks = plays[plays["qb_dropback"] == 1].merge(
        pressure, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"])
    dropbacks = dropbacks[dropbacks["was_pressure"].notna()]
    clean = dropbacks.loc[dropbacks["was_pressure"] == 0, "epa"].mean()
    pressured = dropbacks[dropbacks["was_pressure"] == 1]

    targets = plays[(plays["pass_attempt"] == 1) & (plays["sack"] == 0) & plays["receiver_player_id"].notna()]
    average_target = targets["epa"].mean()
    completions = targets[targets["complete_pass"] == 1]
    incompletions = targets[(targets["incomplete_pass"] == 1) & (targets["interception"] == 0)]
    runs = plays[(plays["play_type"] == "run") & (plays["qb_dropback"] == 0)]

    charted = (nfl.load_pfr_advstats(seasons=seasons, stat_type="def", summary_level="week")
               .filter(pl.col("game_type") == "REG").to_pandas()
               .groupby(["game_id", "team"])[["def_missed_tackles", "def_tackles_combined"]].sum())
    games = (plays.groupby(["game_id", "defteam"]).agg(epa=("epa", "sum"), plays=("epa", "size"))
             .rename_axis(["game_id", "team"]).join(charted, how="inner"))
    X = np.column_stack([np.ones(len(games)), games["def_missed_tackles"], games["plays"], games["def_tackles_combined"]])
    missed_tackle_cost = np.linalg.lstsq(X, games["epa"].to_numpy(), rcond=None)[0][1]
    tackled = tackled_plays(plays)
    no_loss = tackled[tackled["yards_gained"] >= 0]

    return {
        "stop": no_loss["epa"].mean() - no_loss.loc[no_loss["stop"], "epa"].mean(),
        "forced_fumble": tackled["epa"].mean() - plays.loc[plays["fumble_forced"] == 1, "epa"].mean(),
        "sack": clean - pressured.loc[pressured["sack"] == 1, "epa"].mean(),
        "non_sack_pressure": clean - pressured.loc[pressured["sack"] == 0, "epa"].mean(),
        "pressure": clean - pressured["epa"].mean(),
        "tackle_for_loss": runs["epa"].mean() - runs.loc[runs["yards_gained"] < 0, "epa"].mean(),
        "missed_tackle": missed_tackle_cost,
        "coverage_yard": np.polyfit(completions["yards_gained"].clip(-10, 80), completions["epa"], 1)[0],
        "completion": completions["epa"].mean() - incompletions["epa"].mean(),
        "interception": average_target - targets.loc[targets["interception"] == 1, "epa"].mean(),
        "breakup": average_target - incompletions["epa"].mean(),
    }


def load_contracts():
    """Each player's current contract (the most recently signed active one), each player's latest contract
    position, and every veteran contract (not a rookie deal) for the market study."""
    contracts = nfl.load_contracts().to_pandas().dropna(subset=["gsis_id"])
    active = (contracts[contracts["is_active"]].sort_values("year_signed").drop_duplicates("gsis_id", keep="last"))
    active["contract_through"] = active["year_signed"] + active["years"] - 1
    latest_position = contracts.sort_values("year_signed").groupby("gsis_id")["position"].last().rename("room")

    # A rookie deal is any contract signed in or before the player's first NFL season (drafted or not)
    people = nfl.load_players().select(["gsis_id", "pfr_id", "rookie_season", "birth_date"]).to_pandas()
    veteran = contracts.merge(people, on="gsis_id", how="left")
    first_season = veteran["rookie_season"].fillna(veteran["draft_year"])
    veteran = veteran[(veteran["year_signed"] > first_season) & veteran["apy_cap_pct"].notna()
                      & veteran["pfr_id"].notna()].copy()
    veteran["age"] = veteran["year_signed"] - pd.to_datetime(veteran["birth_date"], errors="coerce").dt.year
    return (active[["gsis_id", "apy", "apy_cap_pct", "year_signed", "contract_through"]], latest_position,
            veteran[["pfr_id", "year_signed", "apy_cap_pct", "age"]])


def build_player_seasons(plays, seasons, rooms):
    # Each game's dropback share for each defense turns a player's snaps into pass and run snaps
    share = (plays.groupby(["game_id", "defteam"])["qb_dropback"].mean()
             .rename("dropback_share").reset_index().rename(columns={"defteam": "team"}))
    all_snaps = nfl.load_snap_counts(seasons=seasons).filter(pl.col("game_type") == "REG").to_pandas()
    snaps = all_snaps[all_snaps["defense_snaps"] > 0].merge(share, on=["game_id", "team"], how="left")
    snaps["pass_snaps"] = snaps["defense_snaps"] * snaps["dropback_share"]
    snaps["run_snaps"] = snaps["defense_snaps"] - snaps["pass_snaps"]
    # Snaps and games played on defense by team, so a player traded this season counts for each team that played
    # them, plus how many games each team has played
    team_snaps = (snaps.groupby(["season", "team", "pfr_player_id"])
                  .agg(defense_snaps=("defense_snaps", "sum"), games_played=("game_id", "nunique"))
                  .reset_index().rename(columns={"pfr_player_id": "pfr_id"})
                  .merge(all_snaps.groupby(["season", "team"])["game_id"].nunique().rename("team_games"),
                         left_on=["season", "team"], right_index=True, how="left"))

    # Charted stats game by game. Pro Football Reference's charting can lag the snap counts by a week or
    # more, so only games that have been charted count, keeping each rate's events and snaps in step.
    charted = (nfl.load_pfr_advstats(seasons=seasons, stat_type="def", summary_level="week")
               .filter(pl.col("game_type") == "REG").to_pandas()
               .rename(columns={"def_pressures": "pressures", "def_sacks": "sacks", "def_times_blitzed": "blitzes",
                                "def_targets": "targets", "def_completions_allowed": "completions_allowed",
                                "def_yards_allowed": "yards_allowed", "def_ints": "interceptions",
                                "def_tackles_combined": "tackles", "def_missed_tackles": "missed_tackles"}))
    charted_games = charted[["game_id", "team"]].drop_duplicates()
    games = snaps.merge(charted_games, on=["game_id", "team"]).merge(
        charted[["game_id", "pfr_player_id", "pressures", "sacks", "blitzes", "targets", "completions_allowed",
                 "yards_allowed", "interceptions", "tackles", "missed_tackles"]],
        on=["game_id", "pfr_player_id"], how="left")

    # Tackles for loss and passes defended come from the official box score, matched by player and week
    ids = nfl.load_players().select(["pfr_id", "gsis_id", "display_name", "birth_date"]).drop_nulls("pfr_id") \
             .unique("pfr_id").to_pandas()
    games = (games.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="left")
             .drop(columns="pfr_id").rename(columns={"pfr_player_id": "pfr_id"}))
    box = (nfl.load_player_stats(seasons=seasons, summary_level="week").filter(pl.col("season_type") == "REG")
           .select(["season", "week", "player_id", "def_tackles_for_loss", "def_pass_defended", "def_fumbles_forced"])
           .to_pandas()
           .rename(columns={"player_id": "gsis_id", "def_tackles_for_loss": "tackles_for_loss",
                            "def_pass_defended": "passes_defended", "def_fumbles_forced": "forced_fumbles"}))
    games = (games.merge(box, on=["season", "week", "gsis_id"], how="left")
             .merge(stops_by_game(plays), on=["game_id", "gsis_id"], how="left"))

    count_columns = ["pressures", "sacks", "blitzes", "targets", "completions_allowed", "yards_allowed",
                     "interceptions", "tackles", "missed_tackles", "tackles_for_loss", "passes_defended",
                     "forced_fumbles", "run_stops", "pass_stops"]
    games[count_columns] = games[count_columns].fillna(0)  # no row for a game = nothing recorded
    players = (games.sort_values("week")
               .groupby(["season", "pfr_id"])
               .agg(gsis_id=("gsis_id", "first"), player=("display_name", "first"),
                    birth_date=("birth_date", "first"), roster_position=("position", "last"),
                    defense_snaps=("defense_snaps", "sum"), pass_snaps=("pass_snaps", "sum"),
                    run_snaps=("run_snaps", "sum"), **{c: (c, "sum") for c in count_columns})
               .reset_index())
    players["player"] = players["player"].fillna(players["pfr_id"])
    # Age on September 1 of the season
    birth = pd.to_datetime(players["birth_date"], errors="coerce")
    players["age"] = ((pd.to_datetime(players["season"].astype(str) + "-09-01") - birth).dt.days // 365.25)
    players = players.drop(columns="birth_date").merge(rooms, left_on="gsis_id", right_index=True, how="left")
    players["room"] = players["room"].fillna(players["roster_position"].map(POSITION_TO_ROOM))
    players = assign_role_rooms(players[players["room"].isin(ROOMS)].copy())

    # Stops come about three times as often on run snaps as on pass snaps for a linebacker, so a player's stops
    # are compared with what an average player in the room would make in the same mix of run and pass snaps
    players["stops"] = players["run_stops"] + players["pass_stops"]
    complete = players[players["season"] < players["season"].max()]
    for room in ROOMS:
        r = complete[complete["room"] == room]
        in_room = players["room"] == room
        players.loc[in_room, "expected_stops"] = (
            players.loc[in_room, "run_snaps"] * r["run_stops"].sum() / r["run_snaps"].sum()
            + players.loc[in_room, "pass_snaps"] * r["pass_stops"].sum() / r["pass_snaps"].sum())
    players["coverage_snaps"] = (players["pass_snaps"] - players["blitzes"]).clip(lower=0)
    players["tackle_attempts"] = players["tackles"] + players["missed_tackles"]
    # Pressures include sacks, and the official passes-defended count includes interceptions
    players["non_sack_pressures"] = (players["pressures"] - players["sacks"]).clip(lower=0)
    players["breakups"] = (players["passes_defended"] - players["interceptions"]).clip(lower=0)

    # Snap share: the player's snaps over the defense's snaps in the charted games they dressed for (played on
    # defense or special teams), so games missed hurt don't count against them but being benched does.
    # A defense's snaps in a game are the most any of its players played.
    dressed = all_snaps.merge(charted_games, on=["game_id", "team"])
    dressed["game_defense_snaps"] = dressed.groupby(["game_id", "team"])["defense_snaps"].transform("max")
    dressed = dressed[(dressed["defense_snaps"] > 0) | (dressed["st_snaps"] > 0)]
    team_defense_snaps = (dressed.groupby(["season", "pfr_player_id"])["game_defense_snaps"].sum()
                          .rename("team_defense_snaps").rename_axis(["season", "pfr_id"]))
    players = players.merge(team_defense_snaps, left_on=["season", "pfr_id"], right_index=True, how="left")
    # Games played on defense that season (every game, charted or not), to tell healthy seasons from injured ones
    season_games = (snaps.groupby(["season", "pfr_player_id"])["game_id"].nunique()
                    .rename("season_games").rename_axis(["season", "pfr_id"]))
    players = players.merge(season_games, left_on=["season", "pfr_id"], right_index=True, how="left")
    return players, team_snaps


def load_nickels(season):
    """gsis_ids of the defensive backs listed at nickel (NB) on each team's latest depth chart this season. Shown as
    a tag only: depth charts before 2025 don't list a nickel, and hybrids are listed inconsistently, so it isn't
    reliable enough to grade by."""
    charts = nfl.load_depth_charts(seasons=[season]).to_pandas()
    latest = charts[charts["dt"] == charts.groupby("team")["dt"].transform("max")]
    return set(latest.loc[latest["pos_abb"] == "NB", "gsis_id"].dropna())


def load_roster(season):
    """Each player's team and status in the latest week of this season's rosters (RES = injured reserve)."""
    roster = (nfl.load_rosters_weekly(seasons=[season]).to_pandas().dropna(subset=["pfr_id"])
              .sort_values("week").drop_duplicates("pfr_id", keep="last"))
    roster = roster[roster["status"].isin(["ACT", "RES", "INA"])]
    return roster[["pfr_id", "team", "status"]].rename(columns={"status": "roster_status"})


def assign_role_rooms(players):
    """Contract positions can lag a role change, so edge rushers and linebackers are grouped by what they do
    on passing plays, using each player's latest season with at least 150 pass snaps. Linebackers are
    thrown at on 6%+ of pass snaps; edge rushers on under 2% and they create far more pressure."""
    recent = (players[players["pass_snaps"] >= 150].sort_values("season").groupby("pfr_id").last())
    targeted = recent["targets"] / recent["pass_snaps"]
    pressuring = recent["pressures"] / recent["pass_snaps"]
    rushes = recent.index[(recent["room"] == "LB") & (targeted < 0.04) & (pressuring >= 0.025)]
    covers = recent.index[(recent["room"] == "ED") & (targeted >= 0.06) & (pressuring < 0.025)]
    players.loc[players["pfr_id"].isin(rushes), "room"] = "ED"
    players.loc[players["pfr_id"].isin(covers), "room"] = "LB"
    return players


def estimate_reliability(players, last_complete_season):
    """Year-to-year correlation of each room stat among players with enough opportunities in both seasons,
    plus how many opportunities the room gets per defensive snap (used to put impact on a per-snap scale)."""
    rows = []
    for room, stats in ROOM_STATS.items():
        in_room = players[players["room"] == room]
        complete = in_room[in_room["season"] <= last_complete_season]
        for stat in stats:
            events, opps, minimum, lower_is_better, event = STATS[stat]
            # Every player's team has a full season of snaps, so snap share's minimum is on the player's own snaps
            sample = events if stat == "snap_share" else opps
            correlations, sample_sizes = [], []
            for season in range(FIRST_SEASON, last_complete_season):
                both = (in_room[(in_room["season"] == season) & (in_room[sample] >= minimum)].set_index("pfr_id")
                        [[events, opps]].join(in_room[(in_room["season"] == season + 1) & (in_room[sample] >= minimum)]
                                              .set_index("pfr_id")[[events, opps]], lsuffix="_a", rsuffix="_b",
                                              how="inner"))
                if len(both) < 20:
                    continue
                correlations.append((both[events + "_a"] / both[opps + "_a"]).corr(both[events + "_b"] / both[opps + "_b"]))
                sample_sizes.append(both[[opps + "_a", opps + "_b"]].to_numpy().mean())
            r, n = np.mean(correlations), np.mean(sample_sizes)
            rows.append({"room": room, "stat": stat, "events": events, "opportunities": opps,
                         "lower_is_better": lower_is_better, "event": event, "year_to_year_r": r,
                         "avg_opportunities": n, "k": n * (1 - r) / r if r > 0 else np.nan,
                         "opportunities_per_snap": complete[opps].sum() / complete[SNAPS].sum(),
                         "used": bool(r >= MIN_RELIABILITY or (room, stat) in KEPT_DESPITE_LOW_REPEAT)})
    return pd.DataFrame(rows)


def select_window(room_players, end_season, room=None):
    """Each player's seasons for the grade: end_season always, then his most recent healthy seasons before it
    (HEALTHY_GAMES+ games) from the LOOKBACK_SEASONS before, so a season lost to injury is skipped rather than
    counted. Players without enough healthy seasons fill in with their most recent other ones. Each season is
    weighted by how many seasons ago it was (SEASON_WEIGHTS)."""
    now = room_players[room_players["season"] == end_season]
    before = room_players[room_players["season"].between(end_season - LOOKBACK_SEASONS, end_season - 1)].copy()
    before["healthy"] = before["season_games"] >= HEALTHY_GAMES
    before = before.sort_values(["healthy", "season"], ascending=False)
    before = before[before.groupby("pfr_id").cumcount() < SEASONS_IN_WINDOW - 1]
    window = pd.concat([now, before])
    # Each season's weight depends on how long ago it was, not its place in the window, so skipping an injured
    # season doesn't promote an older one to a recent season's weight
    window["weight"] = (end_season - window["season"]).map(dict(enumerate(ROOM_SEASON_WEIGHTS.get(room, SEASON_WEIGHTS))))
    return window


def score_window(players, method, room, end_season, single_season=False):
    """Blended rates and z-scores for every player in a room over his window ending at end_season (or that
    season alone). Returns (blended rates with window snaps, z-scores, each stat's standard deviation)."""
    used = method[(method["room"] == room) & method["used"]]
    in_room = players[players["room"] == room]
    if single_season:
        window = in_room[in_room["season"] == end_season].assign(weight=1)
    else:
        window = select_window(in_room, end_season, room)
    blended = pd.DataFrame({"window_snaps": window.groupby("pfr_id")[SNAPS].sum()})
    z = pd.DataFrame(index=blended.index)
    spreads = {}
    for _, row in used.iterrows():
        stat = row["stat"]
        weighted = (window.assign(e=window[row["events"]] * window["weight"],
                                  o=window[row["opportunities"]] * window["weight"], raw=window[row["opportunities"]],
                                  evidence=window[row["opportunities"]] * window["weight"] / ROOM_SEASON_WEIGHTS.get(room, SEASON_WEIGHTS)[0])
                    .groupby("pfr_id")[["e", "o", "raw", "evidence"]].sum())
        league_rate = window[row["events"]].sum() / window[row["opportunities"]].sum()
        # The season weights set which seasons count most. The pull toward average uses the opportunities scaled
        # by how recent they are (this season's count in full, older ones less), so a player with only this season
        # isn't treated as having three times the sample, and old play counts as less evidence of today's level.
        # A player with no opportunities (or events but no opportunities) is at the league rate
        recent_rate = (weighted["e"] / weighted["o"]).replace([np.inf, -np.inf], np.nan).fillna(league_rate)
        # A single season (for trends) pulls every player toward average by the same amount, as if each had the
        # room's typical sample, so a player who missed games isn't pulled harder and made to look like he declined
        sample = (pd.Series(weighted["raw"][weighted["raw"] > 0].median(), index=weighted.index) if single_season
                  else weighted["evidence"])
        blended[stat] = (recent_rate * sample + row["k"] * league_rate) / (sample + row["k"])
        # The room's average and spread come from its busier half, so backups don't set the scale
        busy = weighted["raw"] >= weighted["raw"].median()
        spreads[stat] = blended.loc[busy, stat].std()
        z[stat] = (blended[stat] - blended.loc[busy, stat].mean()) / spreads[stat]
        if row["lower_is_better"]:
            z[stat] = -z[stat]
        cap = z_cap(room, stat, row["year_to_year_r"])
        if cap is not None:
            z[stat] = z[stat].clip(-cap, cap)
    return blended, z, spreads


def z_cap(room, stat, repeat):
    """The cap on a stat's z-score, or None if it isn't capped: the room's cap applies only to noisy stats (repeat
    below CAP_BELOW_REPEAT) not listed in NEVER_CAPPED."""
    cap = ROOM_Z_CAP.get(room, Z_CAP)
    noisy = (CAP_BELOW_REPEAT is None or not repeat >= CAP_BELOW_REPEAT) and (room, stat) not in NEVER_CAPPED
    return cap if noisy else None


def grading_caps(method):
    """Record each stat's z-score cap and snap share's weight cap in the method table, for the site."""
    method["z_cap"] = [z_cap(room, stat, r) for room, stat, r in
                       zip(method["room"], method["stat"], method["year_to_year_r"])]
    method["weight_cap"] = [SNAP_SHARE_CAP.get(room) if stat == "snap_share" else None
                            for room, stat in zip(method["room"], method["stat"])]
    return method


def add_impact(reliability, players, event_values, season):
    """Each stat's impact: expected points per 100 snaps between an average player and one a standard deviation
    better. Kept for comparison with the market weights, which are what the grades use."""
    method = reliability.copy()
    method["event_value"] = method["event"].map(event_values)
    for room in ROOMS:
        _, _, spreads = score_window(players, method, room, season)
        for stat, spread in spreads.items():
            i = method.index[(method["room"] == room) & (method["stat"] == stat)]
            method.loc[i, "spread"] = spread
            method.loc[i, "impact"] = method.loc[i, "event_value"] * spread * method.loc[i, "opportunities_per_snap"] * 100
        used = (method["room"] == room) & method["used"]
        impact_weight = method.loc[used, "year_to_year_r"] * method.loc[used, "impact"]
        method.loc[used, "impact_share"] = impact_weight / impact_weight.sum()
    return method


def fit_pay(stats, pay, controls):
    """Non-negative stat weights for pay after removing what the controls explain (Frisch-Waugh)."""
    residual = lambda v: v - controls @ np.linalg.lstsq(controls, v, rcond=None)[0]
    stat_residuals = np.column_stack([residual(stats[:, j]) for j in range(stats.shape[1])])
    pay_residual = residual(pay)
    coefficients, _ = nnls(stat_residuals, pay_residual)
    return coefficients, pay_residual, stat_residuals


def market_weights(players, method, veteran_contracts, season):
    """What the NFL pays for in each room: veteran contracts signed 2019 on, against the player's blended stats
    over the three seasons before signing. Pay (APY as % of the cap) is regressed on the stats' z-scores, holding
    playing time (log of window snaps) and age fixed, with weights kept non-negative. Each stat's weight is the
    extra share of the cap the market pays for one standard deviation better; 200 bootstrap resamples give a
    90% range for each stat's share."""
    rng = np.random.default_rng(0)
    for room in ROOMS:
        used = method.index[(method["room"] == room) & method["used"]]
        stats = method.loc[used, "stat"].tolist()
        signed = []
        for year in range(FIRST_SEASON + 1, season + 1):
            blended, z, _ = score_window(players, method, room, year - 1)
            signed.append(veteran_contracts[veteran_contracts["year_signed"] == year]
                          .merge(z.join(blended["window_snaps"]), left_on="pfr_id", right_index=True))
        deals = pd.concat(signed, ignore_index=True)
        deals = deals[deals["window_snaps"] >= MARKET_MIN_SNAPS].dropna(subset=stats + ["age"])

        pay = deals["apy_cap_pct"].to_numpy() * 100
        # Playing time is a stat itself (snap share), so only age is held fixed
        controls = np.column_stack([np.ones(len(deals)), deals["age"]])
        X = deals[stats].to_numpy()
        coefficients, pay_residual, stat_residuals = fit_pay(X, pay, controls)
        shares = []
        for _ in range(MARKET_BOOTSTRAPS):
            pick = rng.integers(0, len(deals), len(deals))
            c, _, _ = fit_pay(X[pick], pay[pick], controls[pick])
            shares.append(c / c.sum() if c.sum() > 0 else c)
        shares = np.array(shares)

        method.loc[used, "market_cap_pct_per_sd"] = coefficients
        method.loc[used, "market_share"] = coefficients / coefficients.sum()
        method.loc[used, "share_low"] = np.percentile(shares, 5, axis=0)
        method.loc[used, "share_high"] = np.percentile(shares, 95, axis=0)
        method.loc[method["room"] == room, "market_contracts"] = len(deals)
        method.loc[method["room"] == room, "market_r2_age"] = 1 - pay_residual.var() / pay.var()
        method.loc[method["room"] == room, "market_r2_with_stats"] = (
            1 - (pay_residual - stat_residuals @ coefficients).var() / pay.var())
    return set_weights(method)


def set_weights(method):
    """The weights the grades use: the market's, with two changes. Snap share is held to SNAP_SHARE_CAP in rooms
    where more of it hurt the backtest. For corners, lockdown coverage then gets CORNER_LOCKDOWN_SHARE of the
    rest of the grade. The other stats keep the market's proportions."""
    method["weight_share"] = method["market_share"]
    for room in ROOMS:
        used = method.index[(method["room"] == room) & method["used"]]
        market = method.loc[used, "market_share"]
        is_snap = method.loc[used, "stat"] == "snap_share"
        snap = market[is_snap].sum()
        if SNAP_SHARE_CAP[room] is not None:
            snap = min(snap, SNAP_SHARE_CAP[room])
        rest = market[~is_snap]
        if room in GUARANTEED_SHARES:
            guaranteed_stats, share = GUARANTEED_SHARES[room]
            guaranteed = method.loc[rest.index, "stat"].isin(guaranteed_stats)
            if rest[guaranteed].sum() == 0:  # the market gave them nothing: split the guaranteed share evenly
                rest[guaranteed] = 1
            rest = pd.concat([rest[guaranteed] / rest[guaranteed].sum() * share,
                              rest[~guaranteed] / rest[~guaranteed].sum() * (1 - share)])
        else:
            rest = rest / rest.sum()
        method.loc[rest.index, "weight_share"] = rest * (1 - snap)
        method.loc[market[is_snap].index, "weight_share"] = snap
    method["weight"] = method["weight_share"]
    return method


def season_trends(players, method, room, season):
    """Each player's grade in each season of his window on its own (percentile among that season's players in the
    room with a full sample: HEALTHY_GAMES+ games, or REGULAR_SNAPS+ snaps and half the games so far in a season
    still being played; every player pulled toward average by the same amount), and whether it rose each time
    (Improving), fell each time (Declining) or neither (Steady), with more room for young players."""
    weights = method[(method["room"] == room) & method["used"]].set_index("stat")["weight"]
    in_room = players[players["room"] == room]
    points = []
    for s in range(season - LOOKBACK_SEASONS, season + 1):
        _, z, _ = score_window(players, method, room, s, single_season=True)
        composite = (z[weights.index] * weights).sum(axis=1) / weights.sum()
        rows = in_room[in_room["season"] == s].set_index("pfr_id")
        if s == season:  # this season counts once he's played half the games so far
            full = (rows[SNAPS] >= REGULAR_SNAPS) & (rows["season_games"] >= rows["season_games"].max() / 2)
        else:
            full = rows["season_games"] >= HEALTHY_GAMES
        scores = composite.loc[composite.index.intersection(rows.index[full])]
        points.append(pd.DataFrame({"pfr_id": scores.index.to_numpy(), "season": s,
                                    "season_grade": (scores.rank(pct=True) * 100).round().astype(int).to_numpy(),
                                    "age": rows.loc[scores.index, "age"].to_numpy(),
                                    "snap_share": (rows.loc[scores.index, SNAPS]
                                                   / rows.loc[scores.index, TEAM_SNAPS]).to_numpy()}))
    points = pd.concat(points).sort_values("season", ascending=False)
    points = points[points.groupby("pfr_id").cumcount() < SEASONS_IN_WINDOW].sort_values("season")

    def step(change, young, snap_rise):
        """+1 up, -1 down, 0 no real change"""
        if young:
            if change < -YOUNG_DECLINE_MARGIN:
                return -1
            return 1 if change > 0 or snap_rise >= YOUNG_PLAYING_TIME_RISE else 0
        return int(np.sign(change))

    def label(p):
        young = p["age"].iloc[-1] <= YOUNG_AGE
        steps = [step(c, young, r) for c, r in zip(np.diff(p["season_grade"]), np.diff(p["snap_share"]))]
        if not steps:
            return None
        if young and any(s > 0 for s in steps) and not any(s < 0 for s in steps):
            return "Improving"
        return "Improving" if all(s > 0 for s in steps) else "Declining" if all(s < 0 for s in steps) else "Steady"

    by_player = points.groupby("pfr_id")
    return pd.DataFrame({"trend": by_player[["season_grade", "age", "snap_share"]].apply(label),
                         "trend_seasons": by_player["season"].apply(lambda s: ",".join(map(str, s))),
                         "trend_grades": by_player["season_grade"].apply(lambda g: ",".join(map(str, g)))})


def grade_players(players, team_snaps, method, season, roster=None):
    """This season's player grades, using each stat's weight from the method table. With this season's roster,
    players with a track record who have barely played or not played this season (hurt) are added and listed,
    and each player's season-by-season trend is added (skipped without it, to keep the backtests quick)."""
    current = (team_snaps[team_snaps["season"] == season]
               .merge(players[players["season"] == season]
                      [["pfr_id", "gsis_id", "player", "age", "room", *SEASON_COUNTS]]
                      .rename(columns=SEASON_COUNTS), on="pfr_id")
               .rename(columns={SNAPS: "snaps"}))
    latest = players[players["season"] <= season].sort_values("season").drop_duplicates("pfr_id", keep="last")

    graded = []
    for room in ROOMS:
        blended, z, _ = score_window(players, method, room, season)
        weights = method[(method["room"] == room) & method["used"]].set_index("stat")["weight"]
        blended["composite"] = (z[weights.index] * weights).sum(axis=1) / weights.sum()
        if room in COMPOSITE_ADJUSTMENTS:  # e.g. quarterbacks' supporting cast (offense_grades.py)
            blended["composite"] -= COMPOSITE_ADJUSTMENTS[room].reindex(blended.index).fillna(0)

        room_now = current[current["room"] == room]
        if roster is not None:
            # Players whose latest season was in this room, with a track record, on a roster, but no snaps yet
            proven = blended.index[blended["window_snaps"] >= LIMITED_MIN_WINDOW_SNAPS]
            missing = (latest[(latest["room"] == room) & latest["pfr_id"].isin(proven)
                              & ~latest["pfr_id"].isin(current["pfr_id"])]
                       .merge(roster[["pfr_id", "team"]], on="pfr_id"))
            missing = missing.assign(season=season, snaps=0, games_played=0, **{c: 0 for c in SEASON_COUNTS.values()},
                                     age=missing["age"] + season - missing["season"])
            missing = missing.merge(team_snaps[team_snaps["season"] == season].groupby("team")["team_games"].max(),
                                    left_on="team", right_index=True, how="left")
            room_now = pd.concat([room_now, missing[room_now.columns]], ignore_index=True)
            room_now = room_now.merge(roster[["pfr_id", "roster_status"]], on="pfr_id", how="left")
            room_now = room_now.merge(season_trends(players, method, room, season), left_on="pfr_id",
                                      right_index=True, how="left")
        room_now = room_now.merge(blended, left_on="pfr_id", right_index=True, how="left")
        room_now["composite"] = room_now["composite"].fillna(0)
        season_snaps = room_now.groupby("pfr_id")["snaps"].transform("sum")
        regulars = room_now.groupby("pfr_id")["snaps"].sum() >= REGULAR_SNAPS
        regular_scores = room_now.drop_duplicates("pfr_id").set_index("pfr_id").loc[regulars[regulars].index, "composite"]
        room_now["grade"] = room_now["composite"].apply(lambda c: round(100 * (regular_scores < c).mean()))
        # Listed in the rankings: regulars, plus proven players held under the snap minimum this season
        room_now["limited"] = (season_snaps < REGULAR_SNAPS) & (room_now["window_snaps"] >= LIMITED_MIN_WINDOW_SNAPS)
        room_now["listed"] = (season_snaps >= REGULAR_SNAPS) | room_now["limited"]
        graded.append(room_now)

    grades = pd.concat(graded, ignore_index=True)
    grades["season"] = season
    return grades


def season_only_rooms(players, team_snaps, method, season):
    """Each team's room score and rank from this season's stats alone (same stats, weights and pull toward
    average), to show who is playing best right now next to the blended grades."""
    global SEASONS_IN_WINDOW
    saved, SEASONS_IN_WINDOW = SEASONS_IN_WINDOW, 1
    try:
        grades = grade_players(players, team_snaps, method, season)
    finally:
        SEASONS_IN_WINDOW = saved
    grades = grades[grades["snaps"] > 0]
    scores = (grades.groupby(["team", "room"])
              .apply(lambda t: np.average(t["composite"], weights=t["snaps"]), include_groups=False)
              .rename("season_only_score").reset_index())
    scores["season_only_rank"] = (scores.groupby("room")["season_only_score"].rank(ascending=False, method="min")
                                  .astype(int))
    return scores


def team_area_results(plays, season):
    """This season's team result in each room's main area of responsibility."""
    plays = plays[(plays["season"] == season) & plays["epa"].notna()]
    positions = nfl.load_players().select(["gsis_id", "position"]).to_pandas().set_index("gsis_id")["position"]
    targets = plays[(plays["pass_attempt"] == 1) & (plays["sack"] == 0) & plays["receiver_player_id"].notna()].copy()
    receiver = targets["receiver_player_id"].map(positions)
    by_team = lambda df: df.groupby("defteam")["epa"].mean()
    dropbacks = plays[plays["qb_dropback"] == 1]
    return pd.DataFrame({
        "sack_rate": dropbacks.groupby("defteam")["sack"].mean(),
        "run_epa": by_team(plays[(plays["play_type"] == "run") & (plays["qb_dropback"] == 0)]),
        "te_rb_epa": by_team(targets[receiver.isin(["TE", "RB", "FB"])]),
        "wr_outside_epa": by_team(targets[(receiver == "WR") & targets["pass_location"].isin(["left", "right"])]),
        "deep_epa": by_team(targets[targets["air_yards"] >= 15]),
    })


def grade_rooms(grades, areas):
    rooms = []
    for room in ROOMS:
        g = grades[(grades["room"] == room) & (grades["snaps"] > 0)]  # players yet to play add nothing or cost
        team = g.groupby("team").apply(
            lambda t: pd.Series({"room_score": np.average(t["composite"], weights=t["snaps"]),
                                 "players": t["pfr_id"].nunique(),
                                 "cap_pct": t.drop_duplicates("pfr_id")["apy_cap_pct"].fillna(0).sum(),
                                 "apy_millions": t.drop_duplicates("pfr_id")["apy"].fillna(0).sum()}),
            include_groups=False)
        # Value: how far the room is above or below the score expected for what it costs
        slope, intercept = np.polyfit(team["cap_pct"], team["room_score"], 1)
        team["value"] = team["room_score"] - (intercept + slope * team["cap_pct"])
        team["grade_rank"] = team["room_score"].rank(ascending=False, method="min").astype(int)
        team["spend_rank"] = team["cap_pct"].rank(ascending=False, method="min").astype(int)
        team["value_rank"] = team["value"].rank(ascending=False, method="min").astype(int)

        area_stat, area_label, higher_is_better = ROOM_AREA[room]
        team["area_label"] = area_label
        team["area_value"] = areas[area_stat]
        team["area_rank"] = areas[area_stat].rank(ascending=not higher_is_better, method="min").astype(int)
        team["confidence"], team["confidence_note"] = CONFIDENCE[room]
        team["room"], team["room_name"] = room, ROOM_NAMES[room]
        rooms.append(team.reset_index())
    out = pd.concat(rooms, ignore_index=True)
    out["season"] = grades["season"].iloc[0]
    return out.astype({"players": "int64"})


def main():
    season = nfl.get_current_season()
    seasons = list(range(FIRST_SEASON, season + 1))
    print(f"Downloading {FIRST_SEASON}-{season} plays, snaps, player stats and contracts from nflverse...")
    plays = load_plays(seasons)
    contracts, rooms, veteran_contracts = load_contracts()
    players, team_snaps = build_player_seasons(plays, seasons, rooms)

    print("Testing which stats repeat year to year, and measuring what each event is worth...")
    reliability = estimate_reliability(players, season - 1)
    event_values = measure_event_values(plays, seasons[:-1])
    print({event: round(value, 3) for event, value in event_values.items()})
    method = add_impact(reliability, players, event_values, season)

    print("Measuring what the NFL pays for in each room...")
    method = market_weights(players, method, veteran_contracts, season)
    print(method[["room", "stat", "year_to_year_r", "impact_share", "weight_share", "share_low", "share_high"]]
          .round(3).to_string(index=False))

    print(f"Grading {season} players and rooms...")
    grades = (grade_players(players, team_snaps, method, season, load_roster(season))
              .merge(contracts, on="gsis_id", how="left"))
    grades["nickel"] = grades["gsis_id"].isin(load_nickels(season)) & grades["room"].isin(["CB", "S"])
    rooms_table = (grade_rooms(grades, team_area_results(plays, season))
                   .merge(season_only_rooms(players, team_snaps, method, season), on=["team", "room"], how="left"))
    grades = grades.astype({"snaps": "int64", "games_played": "int64"})
    upload([("player_grades", grades), ("room_grades", rooms_table), ("grade_stats", grading_caps(method))])


if __name__ == "__main__":
    main()
