"""Source-blind example tests for issue #5 — World Football Elo engine.

Tests are derived from acceptance criteria only.  No implementation source was
read.  DataFrames are built in-memory; no network or file I/O.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from pydantic import ValidationError

from worldcup_playoff.config import AppConfig
from worldcup_playoff.data.elo import EloConfig, EloEngine, compute_elo, seed_wc2026


# ---------------------------------------------------------------------------
# Helpers — build martj42-schema DataFrames in-memory
# ---------------------------------------------------------------------------


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Return a DataFrame matching the martj42 results schema."""
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


def _gain_for_tournament(tournament: str, neutral: bool = True) -> float:
    """Return TeamA's rating gain from a 1-0 home win with the given tournament string."""
    config = EloConfig()
    result = EloEngine(config).run(
        _one_match(home_goals=1, away_goals=0, neutral=neutral, tournament=tournament)
    )
    return result.final_ratings["TeamA"] - config.initial_rating


# ---------------------------------------------------------------------------
# AC1 — EloEngine.run returns EloResult with history, match_diffs, final_ratings
# ---------------------------------------------------------------------------


class TestWhenRunCalledThenResultHasRequiredComponents:
    def test_when_run_then_result_exposes_final_ratings(self):
        result = EloEngine(EloConfig()).run(_one_match())
        assert hasattr(result, "final_ratings")

    def test_when_run_then_final_ratings_is_dict_with_both_teams(self):
        result = EloEngine(EloConfig()).run(_one_match(home="Alpha", away="Beta"))
        assert isinstance(result.final_ratings, dict)
        assert "Alpha" in result.final_ratings
        assert "Beta" in result.final_ratings

    def test_when_run_then_final_ratings_values_are_floats(self):
        result = EloEngine(EloConfig()).run(_one_match())
        for v in result.final_ratings.values():
            assert isinstance(v, float)

    def test_when_run_then_result_exposes_match_diffs(self):
        result = EloEngine(EloConfig()).run(_one_match())
        assert hasattr(result, "match_diffs")

    def test_when_one_played_match_then_match_diffs_has_one_entry(self):
        result = EloEngine(EloConfig()).run(_one_match())
        assert len(result.match_diffs) == 1

    def test_when_two_played_matches_then_match_diffs_has_two_entries(self):
        df = _make_df(
            [
                {
                    "DATE": "2020-01-01",
                    "HOME_TEAM": "A",
                    "AWAY_TEAM": "B",
                    "HOME_GOALS": 1,
                    "AWAY_GOALS": 0,
                    "TOURNAMENT": "Friendly",
                    "NEUTRAL": True,
                },
                {
                    "DATE": "2020-02-01",
                    "HOME_TEAM": "C",
                    "AWAY_TEAM": "D",
                    "HOME_GOALS": 2,
                    "AWAY_GOALS": 1,
                    "TOURNAMENT": "Friendly",
                    "NEUTRAL": True,
                },
            ]
        )
        result = EloEngine(EloConfig()).run(df)
        assert len(result.match_diffs) == 2

    def test_when_run_then_result_exposes_history_attribute_or_helper(self):
        result = EloEngine(EloConfig()).run(_one_match())
        assert hasattr(result, "history") or hasattr(result, "history_frame")


# ---------------------------------------------------------------------------
# AC2 — Expectation math: We == 0.5 for equal ratings on neutral; dr=400 → We≈0.909
# ---------------------------------------------------------------------------


class TestWhenExpectationMathThenValuesAreExact:
    def test_when_equal_ratings_on_neutral_then_delta_implies_we_is_0_5(self):
        """Equal ratings + neutral → We=0.5 → ΔR = k_friendly × 1.0 × 0.5."""
        config = EloConfig()
        result = EloEngine(config).run(_one_match(home_goals=1, away_goals=0, neutral=True))
        expected_delta = config.k_friendly * 0.5
        assert result.final_ratings["TeamA"] == pytest.approx(
            config.initial_rating + expected_delta
        )

    def test_when_dr_400_via_home_advantage_then_delta_implies_we_approx_0_909(self):
        """home_advantage=400, equal initial ratings, non-neutral → dr=400 → We=10/11.

        ΔR = k_friendly × G=1.0 × (1 − 10/11) = k_friendly / 11.
        """
        config = EloConfig(home_advantage=400.0)
        result = EloEngine(config).run(_one_match(home_goals=1, away_goals=0, neutral=False))
        expected_we = 1.0 / (1.0 + 10.0 ** (-400.0 / 400.0))  # ≈ 0.90909
        expected_delta = config.k_friendly * 1.0 * (1.0 - expected_we)
        assert result.final_ratings["TeamA"] == pytest.approx(
            config.initial_rating + expected_delta, rel=1e-5
        )


# ---------------------------------------------------------------------------
# AC3 — Hand-computed case: both 1500, home wins 1-0, friendly, neutral → 1510/1490 + zero-sum
# ---------------------------------------------------------------------------


class TestWhenHandComputedCaseThenRatingsAndZeroSumCorrect:
    def test_when_both_1500_neutral_friendly_home_wins_1_0_then_home_1510_away_1490(self):
        """Canonical hand check: K=20, G=1.0 (margin 1), We=0.5 → ±10."""
        config = EloConfig()
        result = EloEngine(config).run(_one_match(home_goals=1, away_goals=0, neutral=True))
        assert result.final_ratings["TeamA"] == pytest.approx(1510.0)
        assert result.final_ratings["TeamB"] == pytest.approx(1490.0)

    def test_when_played_match_then_rating_change_is_zero_sum(self):
        config = EloConfig()
        result = EloEngine(config).run(_one_match(home_goals=2, away_goals=1, neutral=True))
        delta_home = result.final_ratings["TeamA"] - config.initial_rating
        delta_away = result.final_ratings["TeamB"] - config.initial_rating
        assert delta_home + delta_away == pytest.approx(0.0, abs=1e-10)

    @given(
        home_goals=st.integers(min_value=0, max_value=10),
        away_goals=st.integers(min_value=0, max_value=10),
    )
    def test_property_when_any_result_then_rating_change_is_zero_sum(
        self, home_goals: int, away_goals: int
    ) -> None:
        """For any scoreline, Δhome + Δaway = 0 (conservation invariant)."""
        config = EloConfig()
        result = EloEngine(config).run(
            _one_match(home_goals=home_goals, away_goals=away_goals, neutral=True)
        )
        delta_home = result.final_ratings["TeamA"] - config.initial_rating
        delta_away = result.final_ratings["TeamB"] - config.initial_rating
        assert delta_home + delta_away == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# AC4 — home_advantage = 0 when NEUTRAL; non-neutral adds home_advantage to dr
# ---------------------------------------------------------------------------


class TestWhenNeutralFlagThenHomeAdvantageIsApplied:
    def test_when_neutral_true_home_wins_then_gain_is_larger_than_non_neutral(self):
        """Neutral removes home_advantage → We=0.5 → larger gain than non-neutral where We>0.5."""
        gain_neutral = _gain_for_tournament("Friendly", neutral=True)
        gain_non_neutral = _gain_for_tournament("Friendly", neutral=False)
        assert gain_neutral > gain_non_neutral

    def test_when_neutral_false_then_home_wins_gain_is_less_than_half_k(self):
        """Non-neutral: home_advantage > 0 → We > 0.5 → gain < k_friendly × 0.5."""
        config = EloConfig()
        assert config.home_advantage > 0.0, "home_advantage must be positive for this test"
        result = EloEngine(config).run(_one_match(home_goals=1, away_goals=0, neutral=False))
        gain = result.final_ratings["TeamA"] - config.initial_rating
        assert gain < config.k_friendly * 0.5

    def test_when_home_advantage_zero_then_neutral_and_non_neutral_give_same_result(self):
        """home_advantage=0 → the neutral flag has no effect on the update."""
        config = EloConfig(home_advantage=0.0)
        result_neutral = EloEngine(config).run(_one_match(home_goals=1, away_goals=0, neutral=True))
        result_non_neutral = EloEngine(config).run(
            _one_match(home_goals=1, away_goals=0, neutral=False)
        )
        assert result_neutral.final_ratings["TeamA"] == pytest.approx(
            result_non_neutral.final_ratings["TeamA"]
        )
        assert result_neutral.final_ratings["TeamB"] == pytest.approx(
            result_non_neutral.final_ratings["TeamB"]
        )


# ---------------------------------------------------------------------------
# AC6 — K classified by tournament importance with correct precedence
# ---------------------------------------------------------------------------


class TestWhenTournamentClassifiedThenKFactorReflectsImportance:
    def test_when_friendly_then_lowest_gain(self):
        assert _gain_for_tournament("Friendly") < _gain_for_tournament(
            "FIFA World Cup qualification"
        )

    def test_when_world_cup_then_highest_gain(self):
        assert _gain_for_tournament("FIFA World Cup") > _gain_for_tournament(
            "FIFA World Cup qualification"
        )

    def test_when_qualification_contains_world_cup_substring_qualifier_tier_takes_precedence(self):
        """'FIFA World Cup qualification' must not match world-cup tier (precedence check)."""
        qual_gain = _gain_for_tournament("FIFA World Cup qualification")
        wc_gain = _gain_for_tournament("FIFA World Cup")
        assert qual_gain < wc_gain

    def test_when_tournament_name_is_lowercase_then_same_k_as_titlecase(self):
        """K classification is case-insensitive."""
        assert _gain_for_tournament("friendly") == pytest.approx(_gain_for_tournament("Friendly"))
        assert _gain_for_tournament("fifa world cup") == pytest.approx(
            _gain_for_tournament("FIFA World Cup")
        )

    def test_when_unknown_tournament_then_defaults_to_friendly_k(self):
        assert _gain_for_tournament("Obscure Regional Invitational") == pytest.approx(
            _gain_for_tournament("Friendly")
        )

    def test_when_continental_tournament_then_k_between_qualifier_and_world_cup(self):
        """Continental tier K must be > qualifier K and < world-cup K."""
        config = EloConfig()
        # Drive the keyword from the config's own defaults so the test is not brittle.
        continental_kw = config.continental_keywords[0]
        continental_gain = _gain_for_tournament(continental_kw)
        qualifier_gain = _gain_for_tournament("FIFA World Cup qualification")
        wc_gain = _gain_for_tournament("FIFA World Cup")
        assert qualifier_gain < continental_gain < wc_gain


# ---------------------------------------------------------------------------
# AC7 — Unplayed fixtures (<NA> goals): no rating change, diff emitted, no raise
# ---------------------------------------------------------------------------


class TestWhenUnplayedFixtureThenBehaviorIsCorrect:
    def _na_df(self) -> pd.DataFrame:
        return _make_df(
            [
                {
                    "DATE": "2026-06-15",
                    "HOME_TEAM": "TeamA",
                    "AWAY_TEAM": "TeamB",
                    "HOME_GOALS": None,
                    "AWAY_GOALS": None,
                    "TOURNAMENT": "FIFA World Cup",
                    "NEUTRAL": True,
                }
            ]
        )

    def test_when_na_goals_then_engine_does_not_raise(self):
        EloEngine(EloConfig()).run(self._na_df())

    def test_when_na_goals_then_ratings_remain_at_initial_rating(self):
        config = EloConfig()
        result = EloEngine(config).run(self._na_df())
        assert result.final_ratings.get("TeamA", config.initial_rating) == pytest.approx(
            config.initial_rating
        )
        assert result.final_ratings.get("TeamB", config.initial_rating) == pytest.approx(
            config.initial_rating
        )

    def test_when_na_goals_then_pre_match_elo_diff_is_still_emitted(self):
        result = EloEngine(EloConfig()).run(self._na_df())
        assert len(result.match_diffs) == 1

    def test_when_played_match_precedes_na_match_then_played_ratings_preserved(self):
        """A played match updates ratings; the subsequent unplayed fixture must not alter them."""
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
                "DATE": "2026-06-15",
                "HOME_TEAM": "TeamA",
                "AWAY_TEAM": "TeamB",
                "HOME_GOALS": None,
                "AWAY_GOALS": None,
                "TOURNAMENT": "FIFA World Cup",
                "NEUTRAL": True,
            },
        ]
        result = EloEngine(config).run(_make_df(rows))
        # Played: 1500 each, K=20, G=1.0, We=0.5 → TeamA=1510, TeamB=1490
        assert result.final_ratings["TeamA"] == pytest.approx(1510.0)
        assert result.final_ratings["TeamB"] == pytest.approx(1490.0)
        # Both matches emit a pre-match diff
        assert len(result.match_diffs) == 2


# ---------------------------------------------------------------------------
# AC8 — Team with no prior matches enters at initial_rating
# ---------------------------------------------------------------------------


class TestWhenTeamHasNoPriorMatchesThenEntersAtInitialRating:
    def test_when_debut_teams_play_then_start_at_initial_rating(self):
        """Both teams debut → start at 1500; 1-0 neutral friendly → 1510 / 1490."""
        config = EloConfig()
        result = EloEngine(config).run(
            _one_match(home="NewA", away="NewB", home_goals=1, away_goals=0, neutral=True)
        )
        assert result.final_ratings["NewA"] == pytest.approx(
            config.initial_rating + config.k_friendly * 0.5
        )
        assert result.final_ratings["NewB"] == pytest.approx(
            config.initial_rating - config.k_friendly * 0.5
        )

    def test_when_debut_team_faces_established_team_then_debut_enters_at_initial_rating(self):
        """A brand-new team entering mid-history uses initial_rating, not any stale value."""
        config = EloConfig()
        rows = [
            # Give TeamA a boosted rating via prior wins.
            {
                "DATE": "2019-01-01",
                "HOME_TEAM": "TeamA",
                "AWAY_TEAM": "Fodder",
                "HOME_GOALS": 3,
                "AWAY_GOALS": 0,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
            # BrandNew team debuts — must enter at initial_rating = 1500.
            {
                "DATE": "2020-01-01",
                "HOME_TEAM": "TeamA",
                "AWAY_TEAM": "BrandNew",
                "HOME_GOALS": 1,
                "AWAY_GOALS": 0,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
        ]
        result = EloEngine(config).run(_make_df(rows))
        # BrandNew entered at 1500; lost to stronger TeamA → negative gain
        brand_new_delta = result.final_ratings["BrandNew"] - config.initial_rating
        assert brand_new_delta < 0.0


# ---------------------------------------------------------------------------
# AC9 — Chronological determinism: shuffled input → identical final ratings
# ---------------------------------------------------------------------------


class TestWhenInputShuffledThenFinalRatingsAreIdentical:
    def test_when_rows_shuffled_then_final_ratings_match(self):
        rows = [
            {
                "DATE": "2020-01-01",
                "HOME_TEAM": "A",
                "AWAY_TEAM": "B",
                "HOME_GOALS": 2,
                "AWAY_GOALS": 1,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
            {
                "DATE": "2020-01-01",
                "HOME_TEAM": "C",
                "AWAY_TEAM": "D",
                "HOME_GOALS": 0,
                "AWAY_GOALS": 3,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
            {
                "DATE": "2020-03-15",
                "HOME_TEAM": "A",
                "AWAY_TEAM": "C",
                "HOME_GOALS": 1,
                "AWAY_GOALS": 1,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
            {
                "DATE": "2020-06-01",
                "HOME_TEAM": "B",
                "AWAY_TEAM": "D",
                "HOME_GOALS": 0,
                "AWAY_GOALS": 0,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            },
        ]
        shuffled = rows[:]
        random.shuffle(shuffled)

        config = EloConfig()
        result_a = EloEngine(config).run(_make_df(rows))
        result_b = EloEngine(config).run(_make_df(shuffled))

        for team in ("A", "B", "C", "D"):
            assert result_a.final_ratings[team] == pytest.approx(
                result_b.final_ratings[team], rel=1e-6
            )

    @given(
        st.lists(
            st.fixed_dictionaries(
                {
                    "home_goals": st.integers(min_value=0, max_value=5),
                    "away_goals": st.integers(min_value=0, max_value=5),
                    "month": st.integers(min_value=1, max_value=12),
                    "day": st.integers(min_value=1, max_value=28),
                }
            ),
            min_size=2,
            max_size=8,
        )
    )
    @settings(max_examples=50)
    def test_property_when_rows_shuffled_then_final_ratings_are_deterministic(
        self, params: list[dict]
    ) -> None:
        """Chronological sort must produce the same final ratings regardless of input row order."""
        teams = ["Alpha", "Beta", "Gamma", "Delta"]
        rows = [
            {
                "DATE": f"2020-{p['month']:02d}-{p['day']:02d}",
                "HOME_TEAM": teams[i % len(teams)],
                "AWAY_TEAM": teams[(i + 1) % len(teams)],
                "HOME_GOALS": p["home_goals"],
                "AWAY_GOALS": p["away_goals"],
                "TOURNAMENT": "Friendly",
                "NEUTRAL": True,
            }
            for i, p in enumerate(params)
        ]
        shuffled = rows[:]
        random.shuffle(shuffled)

        config = EloConfig()
        result_a = EloEngine(config).run(_make_df(rows))
        result_b = EloEngine(config).run(_make_df(shuffled))

        for team in teams:
            if team in result_a.final_ratings and team in result_b.final_ratings:
                assert result_a.final_ratings[team] == pytest.approx(
                    result_b.final_ratings[team], rel=1e-6
                )


# ---------------------------------------------------------------------------
# AC10 — seed_wc2026 returns latest rating; absent teams default to initial_rating
# ---------------------------------------------------------------------------


class TestWhenSeedWc2026CalledThenRatingsAreCorrect:
    def test_when_team_has_history_then_seed_matches_final_rating(self):
        config = EloConfig()
        df = _one_match(home="Brazil", away="Argentina", home_goals=1, away_goals=0, neutral=True)
        result = EloEngine(config).run(df)
        seeds = seed_wc2026(result, ["Brazil", "Argentina"])
        assert seeds["Brazil"] == pytest.approx(result.final_ratings["Brazil"])
        assert seeds["Argentina"] == pytest.approx(result.final_ratings["Argentina"])

    def test_when_team_absent_from_history_then_seed_is_initial_rating(self):
        config = EloConfig()
        df = _one_match(home="Brazil", away="Argentina", home_goals=1, away_goals=0, neutral=True)
        result = EloEngine(config).run(df)
        seeds = seed_wc2026(result, ["Brazil", "Argentina", "UnknownTeam"])
        assert seeds["UnknownTeam"] == pytest.approx(config.initial_rating)

    def test_when_all_teams_absent_then_all_seeds_are_initial_rating(self):
        config = EloConfig()
        df = _one_match(home="TeamX", away="TeamY", home_goals=2, away_goals=0, neutral=True)
        result = EloEngine(config).run(df)
        seeds = seed_wc2026(result, ["Unknown1", "Unknown2", "Unknown3"])
        for v in seeds.values():
            assert v == pytest.approx(config.initial_rating)

    def test_when_seed_called_then_returns_dict_with_exactly_the_requested_teams(self):
        config = EloConfig()
        df = _one_match(home="A", away="B")
        result = EloEngine(config).run(df)
        teams = ["A", "B", "C"]
        seeds = seed_wc2026(result, teams)
        assert set(seeds.keys()) == set(teams)


# ---------------------------------------------------------------------------
# AC11 — EloConfig defaults + validation + AppConfig.elo
# ---------------------------------------------------------------------------


class TestWhenEloConfigValidationThenCorrectBehavior:
    def test_when_default_elo_config_then_no_error_and_positive_values(self):
        config = EloConfig()
        assert config.initial_rating > 0.0
        assert config.home_advantage >= 0.0
        assert config.k_friendly > 0
        assert config.k_qualifier > 0
        assert config.k_continental > 0
        assert config.k_world_cup > 0

    def test_when_initial_rating_zero_then_validation_error(self):
        with pytest.raises(ValidationError):
            EloConfig(initial_rating=0.0)

    def test_when_initial_rating_negative_then_validation_error(self):
        with pytest.raises(ValidationError):
            EloConfig(initial_rating=-1.0)

    def test_when_k_friendly_zero_then_validation_error(self):
        with pytest.raises(ValidationError):
            EloConfig(k_friendly=0)

    def test_when_k_friendly_negative_then_validation_error(self):
        with pytest.raises(ValidationError):
            EloConfig(k_friendly=-10)

    def test_when_k_qualifier_zero_then_validation_error(self):
        with pytest.raises(ValidationError):
            EloConfig(k_qualifier=0)

    def test_when_k_continental_zero_then_validation_error(self):
        with pytest.raises(ValidationError):
            EloConfig(k_continental=0)

    def test_when_k_world_cup_zero_then_validation_error(self):
        with pytest.raises(ValidationError):
            EloConfig(k_world_cup=0)

    def test_when_home_advantage_negative_then_validation_error(self):
        with pytest.raises(ValidationError):
            EloConfig(home_advantage=-0.001)

    def test_when_home_advantage_zero_then_no_validation_error(self):
        """home_advantage=0 is valid (all-neutral-venue setup)."""
        EloConfig(home_advantage=0.0)  # must not raise

    def test_when_app_config_constructed_then_elo_field_is_elo_config_instance(self):
        app = AppConfig()
        assert isinstance(app.elo, EloConfig)


# ---------------------------------------------------------------------------
# compute_elo factory function
# ---------------------------------------------------------------------------


class TestWhenComputeEloFactoryCalledThenResultIsEquivalentToEngine:
    def test_when_compute_elo_called_without_config_then_result_has_final_ratings(self):
        result = compute_elo(_one_match())
        assert hasattr(result, "final_ratings")

    def test_when_compute_elo_called_with_config_then_result_matches_engine(self):
        config = EloConfig()
        df = _one_match(home_goals=1, away_goals=0, neutral=True)
        result = compute_elo(df, config)
        # Same hand-computed result: both 1500, friendly, K=20, We=0.5 → 1510/1490
        assert result.final_ratings["TeamA"] == pytest.approx(1510.0)
        assert result.final_ratings["TeamB"] == pytest.approx(1490.0)

    def test_when_compute_elo_called_with_custom_initial_rating_then_config_is_used(self):
        config = EloConfig(initial_rating=1000.0)
        df = _one_match(home_goals=1, away_goals=0, neutral=True)
        result = compute_elo(df, config)
        # Both start at 1000; k_friendly * 0.5 gain for winner
        assert result.final_ratings["TeamA"] == pytest.approx(
            config.initial_rating + config.k_friendly * 0.5
        )
