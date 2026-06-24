"""Source-blind example tests for issue #31 — World Football Elo engine.

Tests are derived from the issue #31 acceptance criteria only. No implementation
source was read. DataFrames are built in-memory; no network or file I/O.

Criteria tested (mapping to the three runtime-verifiable ACs):

  AC1 — EloEngine.run(df) returns per-team per-date post-match EloRating history
         AND exactly one pre-match MatchEloDiff per input row (incl. unplayed).
  AC2 — elo_diff = home_elo − away_elo (excludes home_advantage);
         home_adv = 0 for neutral matches does NOT change the reported elo_diff.
  AC4 — seed_wc2026(result, teams) returns each team's LATEST Elo; defaults to
         EloConfig.initial_rating for teams absent from the run history.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from worldcup_playoff.data.elo import (
    EloConfig,
    EloEngine,
    EloRating,
    MatchEloDiff,
    seed_wc2026,
)


# ---------------------------------------------------------------------------
# Helpers — build martj42-schema DataFrames in-memory
# ---------------------------------------------------------------------------


def _make_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "DATE",
                "HOME_TEAM",
                "AWAY_TEAM",
                "HOME_GOALS",
                "AWAY_GOALS",
                "TOURNAMENT",
                "NEUTRAL",
            ]
        )
    df = pd.DataFrame(rows)
    for col in ("HOME_GOALS", "AWAY_GOALS"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    if "NEUTRAL" not in df.columns:
        df["NEUTRAL"] = False
    return df


def _one_match(
    *,
    home: str = "TeamA",
    away: str = "TeamB",
    home_goals: int | None = 1,
    away_goals: int | None = 0,
    date: str = "2020-06-01",
    tournament: str = "Friendly",
    neutral: bool = True,
) -> pd.DataFrame:
    return _make_df(
        [
            {
                "DATE": date,
                "HOME_TEAM": home,
                "AWAY_TEAM": away,
                "HOME_GOALS": home_goals,
                "AWAY_GOALS": away_goals,
                "TOURNAMENT": tournament,
                "NEUTRAL": neutral,
            }
        ]
    )


def _played_rows(n: int) -> list[dict]:
    """Return n played-match rows with distinct dates, cycling through 4 teams."""
    teams = ["Alpha", "Beta", "Gamma", "Delta"]
    return [
        {
            "DATE": f"202{i // 12 + 0}-{(i % 12) + 1:02d}-01",
            "HOME_TEAM": teams[i % 4],
            "AWAY_TEAM": teams[(i + 1) % 4],
            "HOME_GOALS": 1,
            "AWAY_GOALS": 0,
            "TOURNAMENT": "Friendly",
            "NEUTRAL": True,
        }
        for i in range(n)
    ]


def _n_match_df(n: int) -> pd.DataFrame:
    return _make_df(_played_rows(n))


def _get_history(result) -> list | pd.DataFrame:
    if hasattr(result, "history"):
        return result.history
    return result.history_frame


# ---------------------------------------------------------------------------
# AC1 — EloRating and MatchEloDiff are importable named types
# ---------------------------------------------------------------------------


class TestWhenImportedThenNamedTypesExist:
    def test_when_elo_rating_is_imported_then_it_is_not_none(self):
        assert EloRating is not None

    def test_when_match_elo_diff_is_imported_then_it_is_not_none(self):
        assert MatchEloDiff is not None


# ---------------------------------------------------------------------------
# AC1 — EloRating history: per-team per-date post-match records
# ---------------------------------------------------------------------------


class TestWhenRunThenEloRatingHistoryIsReturned:
    def test_when_one_match_played_then_history_contains_at_least_two_entries(self):
        """Both teams get a post-match EloRating entry after one played match."""
        result = EloEngine(EloConfig()).run(_one_match())
        history = _get_history(result)
        assert len(history) >= 2

    def test_when_team_plays_two_matches_then_history_has_two_entries_for_that_team(self):
        """Per-team per-date: one entry per match played, for each participant."""
        rows = [
            {
                "DATE": "2020-01-01",
                "HOME_TEAM": "TeamA",
                "AWAY_TEAM": "TeamB",
                "HOME_GOALS": 1,
                "AWAY_GOALS": 0,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
            {
                "DATE": "2020-06-01",
                "HOME_TEAM": "TeamA",
                "AWAY_TEAM": "TeamC",
                "HOME_GOALS": 2,
                "AWAY_GOALS": 1,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
        ]
        result = EloEngine(EloConfig()).run(_make_df(rows))
        history = _get_history(result)
        if isinstance(history, list):
            a_entries = [r for r in history if getattr(r, "team", None) == "TeamA"]
        else:
            col = "team" if "team" in history.columns else history.columns[0]
            a_entries = history[history[col] == "TeamA"]
        assert len(a_entries) == 2

    def test_when_unplayed_fixture_then_history_has_no_entry_for_that_fixture(self):
        """Unplayed fixtures (NA goals) must NOT add a post-match EloRating entry."""
        played = _one_match(date="2020-01-01", home_goals=1, away_goals=0)
        unplayed = _make_df(
            [
                {
                    "DATE": "2026-06-20",
                    "HOME_TEAM": "TeamA",
                    "AWAY_TEAM": "TeamB",
                    "HOME_GOALS": None,
                    "AWAY_GOALS": None,
                    "TOURNAMENT": "FIFA World Cup",
                    "NEUTRAL": True,
                }
            ]
        )
        df_played_only = pd.concat([played], ignore_index=True)
        df_with_unplayed = pd.concat([played, unplayed], ignore_index=True)
        result_played = EloEngine(EloConfig()).run(df_played_only)
        result_with_unplayed = EloEngine(EloConfig()).run(df_with_unplayed)
        # The history entry count must not grow by adding an unplayed fixture
        assert len(_get_history(result_played)) == len(_get_history(result_with_unplayed))

    def test_when_elo_rating_instances_in_history_then_each_has_team_and_rating(self):
        """Each EloRating record must expose the team name and a numeric post-match rating."""
        result = EloEngine(EloConfig()).run(_one_match())
        history = _get_history(result)
        if isinstance(history, list):
            record = history[0]
            assert hasattr(record, "team") or hasattr(record, "team_name")
            assert hasattr(record, "rating") or hasattr(record, "elo")
        else:
            assert len(history.columns) >= 2


# ---------------------------------------------------------------------------
# AC1 — MatchEloDiff: exactly one per input row (including unplayed)
# ---------------------------------------------------------------------------


class TestWhenRunThenExactlyOneMatchEloDiffPerRow:
    def test_when_one_row_in_df_then_one_match_diff_returned(self):
        result = EloEngine(EloConfig()).run(_one_match())
        assert len(result.match_diffs) == 1

    def test_when_five_rows_in_df_then_five_match_diffs_returned(self):
        result = EloEngine(EloConfig()).run(_n_match_df(5))
        assert len(result.match_diffs) == 5

    def test_when_df_contains_unplayed_fixture_then_match_diff_count_equals_total_rows(self):
        """Unplayed fixtures must also emit a pre-match MatchEloDiff."""
        played = _one_match(date="2020-01-01")
        unplayed = _make_df(
            [
                {
                    "DATE": "2026-06-20",
                    "HOME_TEAM": "TeamA",
                    "AWAY_TEAM": "TeamB",
                    "HOME_GOALS": None,
                    "AWAY_GOALS": None,
                    "TOURNAMENT": "FIFA World Cup",
                    "NEUTRAL": True,
                }
            ]
        )
        df = pd.concat([played, unplayed], ignore_index=True)
        result = EloEngine(EloConfig()).run(df)
        assert len(result.match_diffs) == 2

    def test_when_all_rows_are_unplayed_then_match_diff_count_equals_row_count(self):
        """Even a fully-unplayed schedule emits one MatchEloDiff per fixture."""
        rows = [
            {
                "DATE": f"2026-06-{20 + i:02d}",
                "HOME_TEAM": f"Team{i}",
                "AWAY_TEAM": f"Team{i + 1}",
                "HOME_GOALS": None,
                "AWAY_GOALS": None,
                "TOURNAMENT": "FIFA World Cup",
                "NEUTRAL": True,
            }
            for i in range(4)
        ]
        result = EloEngine(EloConfig()).run(_make_df(rows))
        assert len(result.match_diffs) == 4

    @given(st.integers(min_value=1, max_value=12))
    @settings(max_examples=20)
    def test_property_when_n_rows_then_exactly_n_match_diffs(self, n: int) -> None:
        """Invariant: one MatchEloDiff per input row for any n ≥ 1."""
        result = EloEngine(EloConfig()).run(_n_match_df(n))
        assert len(result.match_diffs) == n


# ---------------------------------------------------------------------------
# AC1 — MatchEloDiff structure: named fields
# ---------------------------------------------------------------------------


class TestWhenMatchEloDiffStructureThenNamedFieldsArePresent:
    def test_when_run_then_each_match_diff_is_match_elo_diff_instance(self):
        result = EloEngine(EloConfig()).run(_one_match())
        assert isinstance(result.match_diffs[0], MatchEloDiff)

    def test_when_run_then_match_diff_has_elo_diff_field(self):
        diff = EloEngine(EloConfig()).run(_one_match()).match_diffs[0]
        assert hasattr(diff, "elo_diff")

    def test_when_run_then_match_diff_has_home_elo_field(self):
        diff = EloEngine(EloConfig()).run(_one_match()).match_diffs[0]
        assert hasattr(diff, "home_elo")

    def test_when_run_then_match_diff_has_away_elo_field(self):
        diff = EloEngine(EloConfig()).run(_one_match()).match_diffs[0]
        assert hasattr(diff, "away_elo")

    def test_when_run_then_match_diff_elo_diff_is_numeric(self):
        diff = EloEngine(EloConfig()).run(_one_match()).match_diffs[0]
        assert isinstance(diff.elo_diff, float)

    def test_when_run_then_match_diff_home_elo_is_numeric(self):
        diff = EloEngine(EloConfig()).run(_one_match()).match_diffs[0]
        assert isinstance(diff.home_elo, float)

    def test_when_run_then_match_diff_away_elo_is_numeric(self):
        diff = EloEngine(EloConfig()).run(_one_match()).match_diffs[0]
        assert isinstance(diff.away_elo, float)


# ---------------------------------------------------------------------------
# AC1 — MatchEloDiff is PRE-match (captures ratings before the update)
# ---------------------------------------------------------------------------


class TestWhenMatchEloDiffIsPreMatchThenValuesReflectPreUpdateRatings:
    def test_when_both_teams_debut_then_home_elo_and_away_elo_equal_initial_rating(self):
        """Both teams are new → pre-match ratings are initial_rating for both."""
        config = EloConfig()
        diff = EloEngine(config).run(_one_match()).match_diffs[0]
        assert diff.home_elo == pytest.approx(config.initial_rating)
        assert diff.away_elo == pytest.approx(config.initial_rating)

    def test_when_two_matches_played_then_second_diff_uses_post_first_match_ratings(self):
        """Second MatchEloDiff must use the ratings after the first match updated them."""
        config = EloConfig()
        rows = [
            {
                "DATE": "2020-01-01",
                "HOME_TEAM": "TeamA",
                "AWAY_TEAM": "TeamB",
                "HOME_GOALS": 1,
                "AWAY_GOALS": 0,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
            {
                "DATE": "2020-06-01",
                "HOME_TEAM": "TeamA",
                "AWAY_TEAM": "TeamB",
                "HOME_GOALS": 1,
                "AWAY_GOALS": 0,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
        ]
        result = EloEngine(config).run(_make_df(rows))
        diff1, diff2 = result.match_diffs[0], result.match_diffs[1]
        # After match 1: TeamA gained, TeamB lost → match 2 pre-match home_elo > away_elo
        assert diff1.home_elo == pytest.approx(config.initial_rating)
        assert diff2.home_elo > config.initial_rating
        assert diff2.away_elo < config.initial_rating

    def test_when_unplayed_fixture_follows_played_match_then_diff_uses_updated_ratings(self):
        """Unplayed fixture MatchEloDiff must still reflect ratings updated by prior matches."""
        config = EloConfig()
        rows = [
            {
                "DATE": "2020-01-01",
                "HOME_TEAM": "TeamA",
                "AWAY_TEAM": "TeamB",
                "HOME_GOALS": 1,
                "AWAY_GOALS": 0,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
            {
                "DATE": "2026-06-20",
                "HOME_TEAM": "TeamA",
                "AWAY_TEAM": "TeamB",
                "HOME_GOALS": None,
                "AWAY_GOALS": None,
                "TOURNAMENT": "FIFA World Cup",
                "NEUTRAL": True,
            },
        ]
        result = EloEngine(config).run(_make_df(rows))
        diff_unplayed = result.match_diffs[1]
        # TeamA's pre-unplayed-match rating must reflect the earlier win
        assert diff_unplayed.home_elo > config.initial_rating
        assert diff_unplayed.away_elo < config.initial_rating


# ---------------------------------------------------------------------------
# AC2 — elo_diff = home_elo − away_elo (definition check)
# ---------------------------------------------------------------------------


class TestWhenEloDiffDefinitionThenItEqualsHomeMinusAway:
    def test_when_equal_ratings_then_elo_diff_is_zero(self):
        """home_elo = away_elo = initial_rating → elo_diff = 0."""
        diff = EloEngine(EloConfig()).run(_one_match()).match_diffs[0]
        assert diff.elo_diff == pytest.approx(0.0)

    def test_when_elo_diff_computed_then_it_equals_home_elo_minus_away_elo_exactly(self):
        """elo_diff is defined as home_elo − away_elo; must equal the arithmetic result."""
        diff = EloEngine(EloConfig()).run(_one_match()).match_diffs[0]
        assert diff.elo_diff == pytest.approx(diff.home_elo - diff.away_elo)

    def test_when_home_team_has_higher_rating_then_elo_diff_is_positive(self):
        """TeamA has won a prior match → higher rating → positive elo_diff when home again."""
        config = EloConfig()
        rows = [
            {
                "DATE": "2019-01-01",
                "HOME_TEAM": "TeamA",
                "AWAY_TEAM": "TeamB",
                "HOME_GOALS": 3,
                "AWAY_GOALS": 0,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
            {
                "DATE": "2020-01-01",
                "HOME_TEAM": "TeamA",
                "AWAY_TEAM": "TeamB",
                "HOME_GOALS": 1,
                "AWAY_GOALS": 0,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
        ]
        result = EloEngine(config).run(_make_df(rows))
        diff2 = result.match_diffs[1]
        assert diff2.elo_diff > 0.0
        assert diff2.elo_diff == pytest.approx(diff2.home_elo - diff2.away_elo)

    def test_when_away_team_has_higher_rating_then_elo_diff_is_negative(self):
        """TeamB won prior match as home team; now plays away → TeamA's elo_diff < 0."""
        config = EloConfig()
        rows = [
            {
                "DATE": "2019-01-01",
                "HOME_TEAM": "TeamB",
                "AWAY_TEAM": "TeamA",
                "HOME_GOALS": 3,
                "AWAY_GOALS": 0,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
            {
                "DATE": "2020-01-01",
                "HOME_TEAM": "TeamA",
                "AWAY_TEAM": "TeamB",
                "HOME_GOALS": 1,
                "AWAY_GOALS": 0,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
        ]
        result = EloEngine(config).run(_make_df(rows))
        diff2 = result.match_diffs[1]
        assert diff2.elo_diff < 0.0
        assert diff2.elo_diff == pytest.approx(diff2.home_elo - diff2.away_elo)

    @given(
        home_goals=st.integers(min_value=0, max_value=8),
        away_goals=st.integers(min_value=0, max_value=8),
    )
    def test_property_when_any_result_then_elo_diff_equals_home_minus_away(
        self, home_goals: int, away_goals: int
    ) -> None:
        """Invariant: elo_diff == home_elo - away_elo for any scoreline."""
        diff = (
            EloEngine(EloConfig())
            .run(_one_match(home_goals=home_goals, away_goals=away_goals))
            .match_diffs[0]
        )
        assert diff.elo_diff == pytest.approx(diff.home_elo - diff.away_elo, abs=1e-9)


# ---------------------------------------------------------------------------
# AC2 — elo_diff EXCLUDES home_advantage (neutral and non-neutral give same diff)
# ---------------------------------------------------------------------------


class TestWhenEloDiffExcludesHomeAdvantage:
    def test_when_equal_ratings_neutral_match_then_elo_diff_is_zero_regardless_of_home_adv(self):
        """Neutral: home_adv = 0 in the update formula; elo_diff = 0 (both at initial_rating)."""
        config = EloConfig(home_advantage=300.0)
        diff = EloEngine(config).run(_one_match(neutral=True)).match_diffs[0]
        assert diff.elo_diff == pytest.approx(0.0)

    def test_when_equal_ratings_non_neutral_match_then_elo_diff_is_still_zero(self):
        """Non-neutral: home_advantage shifts We but elo_diff = home_elo - away_elo = 0."""
        config = EloConfig(home_advantage=300.0)
        diff = EloEngine(config).run(_one_match(neutral=False)).match_diffs[0]
        # elo_diff must NOT include home_advantage (300 would make it non-zero if included)
        assert diff.elo_diff == pytest.approx(0.0)

    def test_when_same_teams_neutral_vs_non_neutral_then_elo_diff_field_is_identical(self):
        """The neutral flag must not change the reported elo_diff value."""
        config = EloConfig(home_advantage=150.0)
        diff_neutral = EloEngine(config).run(_one_match(neutral=True)).match_diffs[0]
        diff_non_neutral = EloEngine(config).run(_one_match(neutral=False)).match_diffs[0]
        assert diff_neutral.elo_diff == pytest.approx(diff_non_neutral.elo_diff)

    @given(
        home_advantage=st.floats(
            min_value=0.0, max_value=600.0, allow_nan=False, allow_infinity=False
        )
    )
    @settings(max_examples=30)
    def test_property_when_any_home_advantage_then_elo_diff_equals_home_minus_away(
        self, home_advantage: float
    ) -> None:
        """Invariant: elo_diff = home_elo − away_elo for any home_advantage, neutral or not."""
        config = EloConfig(home_advantage=home_advantage)
        for neutral in (True, False):
            diff = EloEngine(config).run(_one_match(neutral=neutral)).match_diffs[0]
            assert diff.elo_diff == pytest.approx(diff.home_elo - diff.away_elo, abs=1e-9), (
                f"neutral={neutral}, home_advantage={home_advantage}"
            )


# ---------------------------------------------------------------------------
# AC4 — seed_wc2026: latest Elo + initial_rating default for absent teams
# ---------------------------------------------------------------------------


class TestWhenSeedWc2026ThenLatestEloIsReturned:
    def test_when_team_plays_multiple_matches_then_seed_is_most_recent_rating(self):
        """seed_wc2026 must return the rating after all matches, not an earlier snapshot."""
        config = EloConfig()
        rows = [
            {
                "DATE": "2020-01-01",
                "HOME_TEAM": "TeamA",
                "AWAY_TEAM": "Fodder1",
                "HOME_GOALS": 1,
                "AWAY_GOALS": 0,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
            {
                "DATE": "2021-01-01",
                "HOME_TEAM": "TeamA",
                "AWAY_TEAM": "Fodder2",
                "HOME_GOALS": 1,
                "AWAY_GOALS": 0,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
            {
                "DATE": "2022-01-01",
                "HOME_TEAM": "TeamA",
                "AWAY_TEAM": "Fodder3",
                "HOME_GOALS": 1,
                "AWAY_GOALS": 0,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
        ]
        result = EloEngine(config).run(_make_df(rows))
        seeds = seed_wc2026(result, ["TeamA"])
        assert seeds["TeamA"] == pytest.approx(result.final_ratings["TeamA"])

    def test_when_team_absent_from_history_then_seed_is_initial_rating(self):
        config = EloConfig()
        result = EloEngine(config).run(_one_match(home="X", away="Y"))
        seeds = seed_wc2026(result, ["NotInHistory"])
        assert seeds["NotInHistory"] == pytest.approx(config.initial_rating)

    def test_when_mix_of_known_and_absent_teams_then_known_use_latest_absent_use_default(self):
        config = EloConfig()
        result = EloEngine(config).run(
            _one_match(home="France", away="Germany", home_goals=1, away_goals=0, neutral=True)
        )
        seeds = seed_wc2026(result, ["France", "Germany", "Brazil"])
        assert seeds["France"] == pytest.approx(result.final_ratings["France"])
        assert seeds["Germany"] == pytest.approx(result.final_ratings["Germany"])
        assert seeds["Brazil"] == pytest.approx(config.initial_rating)

    def test_when_seed_called_then_result_is_dict_covering_all_requested_teams(self):
        config = EloConfig()
        result = EloEngine(config).run(_one_match(home="A", away="B"))
        teams = ["A", "B", "C", "D"]
        seeds = seed_wc2026(result, teams)
        assert set(seeds.keys()) == set(teams)

    def test_when_seed_called_with_empty_teams_then_empty_dict_is_returned(self):
        result = EloEngine(EloConfig()).run(_one_match())
        seeds = seed_wc2026(result, [])
        assert seeds == {}

    @given(st.integers(min_value=1, max_value=15))
    @settings(max_examples=20)
    def test_property_when_n_absent_teams_then_all_default_to_initial_rating(self, n: int) -> None:
        """Invariant: seed for any team absent from history is always initial_rating."""
        config = EloConfig()
        result = EloEngine(config).run(_one_match(home="Known1", away="Known2"))
        absent = [f"Unknown{i}" for i in range(n)]
        seeds = seed_wc2026(result, absent)
        for team in absent:
            assert seeds[team] == pytest.approx(config.initial_rating)
