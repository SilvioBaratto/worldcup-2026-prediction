# Graph Report - worldcup-2026-prediction  (2026-06-23)

## Corpus Check
- 63 files · ~49,014 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1627 nodes · 3056 edges · 116 communities (68 shown, 48 thin omitted)
- Extraction: 70% EXTRACTED · 30% INFERRED · 0% AMBIGUOUS · INFERRED: 930 edges (avg confidence: 0.66)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `441f4dc3`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 86|Community 86]]
- [[_COMMUNITY_Community 87|Community 87]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 91|Community 91]]
- [[_COMMUNITY_Community 92|Community 92]]
- [[_COMMUNITY_Community 93|Community 93]]
- [[_COMMUNITY_Community 94|Community 94]]
- [[_COMMUNITY_Community 95|Community 95]]
- [[_COMMUNITY_Community 96|Community 96]]
- [[_COMMUNITY_Community 97|Community 97]]
- [[_COMMUNITY_Community 98|Community 98]]
- [[_COMMUNITY_Community 99|Community 99]]
- [[_COMMUNITY_Community 105|Community 105]]
- [[_COMMUNITY_Community 106|Community 106]]
- [[_COMMUNITY_Community 107|Community 107]]
- [[_COMMUNITY_Community 108|Community 108]]
- [[_COMMUNITY_Community 109|Community 109]]
- [[_COMMUNITY_Community 110|Community 110]]
- [[_COMMUNITY_Community 111|Community 111]]
- [[_COMMUNITY_Community 112|Community 112]]
- [[_COMMUNITY_Community 113|Community 113]]
- [[_COMMUNITY_Community 114|Community 114]]

## God Nodes (most connected - your core abstractions)
1. `FootballClient` - 64 edges
2. `Matchup` - 60 edges
3. `ClientConfig` - 56 edges
4. `Pipeline` - 56 edges
5. `FeaturesConfig` - 45 edges
6. `LiveTournamentAdapter` - 40 edges
7. `bracket_config()` - 35 edges
8. `FittedDistribution` - 34 edges
9. `DataConfig` - 33 edges
10. `RoundResult` - 33 edges

## Surprising Connections (you probably didn't know these)
- `test_distribution_config_min_season_default()` --calls--> `DistributionConfig`  [INFERRED]
  tests/test_config.py → worldcup_playoff/config.py
- `test_matchup_fields()` --calls--> `Matchup`  [INFERRED]
  tests/test_config.py → worldcup_playoff/config.py
- `test_matchup_group_defaults_to_empty_string()` --calls--> `Matchup`  [INFERRED]
  tests/test_config.py → worldcup_playoff/config.py
- `test_load_config_round_trip()` --calls--> `load_config()`  [INFERRED]
  tests/test_config.py → worldcup_playoff/config.py
- `test_load_config_defaults_for_missing_sections()` --calls--> `load_config()`  [INFERRED]
  tests/test_config.py → worldcup_playoff/config.py

## Communities (116 total, 48 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.01
Nodes (171): 1. FC Heidenheim 1846, 1. FC K\u00f6ln, 1. FC Union Berlin, 1. FSV Mainz 05, AC Milan, AC Monza, AC Pisa 1909, AC Sparta Praha (+163 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (19): Tests for config loading and Pydantic validation., World Cup uses single-match ties — no series-length fields on SimulationConfig., World Cup uses single-match ties — no series-length fields on SimulationConfig., test_distribution_config_min_season_default(), test_epsilon_default(), test_extra_fields_ignored(), test_load_config_defaults_for_missing_sections(), test_load_config_round_trip() (+11 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (48): build_players_csv(), Build players.csv and write to disk.      Args:         output_path: Where to wr, FootballClient, Attempt the GET request up to ``max_retries + 1`` times.          Uses exponenti, Replace the requests.Session with a fresh instance., Create a new requests.Session with appropriate headers., Wraps football-data.org v4 REST calls with rate limiting and retries.      Imple, Issue a GET request against the v4 base URL and return parsed JSON.          Arg (+40 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (33): Rate-limited football-data.org API client with circuit-breaker retry logic., ClassifierFactory, ClassifierTrainer, create(), load_model(), Classifier creation, training, and persistence., Creates configured classifier instances from config., Handles data preparation and model fitting. (+25 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (30): app_config(), Default AppConfig with no modifications., _make_bracket_toml(), _make_config_toml(), _make_minimal_rounds(), CLI smoke tests using typer.testing.CliRunner.  All heavy computation (network,, TestParseSeasonRange, TestShouldRun (+22 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (43): fitter, scipy, DistributionFitter, FeatureSampler, FittedDistribution, load(), Statistical distribution fitting and feature sampling for football match data., Return only rows at or after ``min_season``.          Falls back to the full Dat (+35 more)

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (33): BracketBuilder, default_competition(), default_season(), generate_bracket_toml(), _pair_teams(), Bracket builder: generates a World Cup knockout bracket TOML from live standings, Serialize a BracketConfig to TOML format.      Args:         bracket: The bracke, Build a World Cup bracket from qualified teams and write to a TOML file.      Ar (+25 more)

### Community 7 - "Community 7"
Cohesion: 0.15
Nodes (12): _make_cleaner(), _minimal_details_df(), _minimal_matches_df(), Tests for DataCleaner., Return (matches_df, details_df) for full pipeline testing., Return (matches_df, details_df) for full pipeline testing., Minimal matches.csv DataFrame with no feature columns (to be merged with details, Minimal matches.csv DataFrame with no feature columns (to be merged with details (+4 more)

### Community 8 - "Community 8"
Cohesion: 0.04
Nodes (44): base_commit, cache_prewarmed, cache_ttl_probe, client_alive_at, codex_model, commit_enabled, current_cycle, cycle5_decisions (+36 more)

### Community 9 - "Community 9"
Cohesion: 0.05
Nodes (57): build_ranking_csv(), _extract_detail_row(), _extract_match_row(), _extract_ranking_row(), _heuristic_shots(), _heuristic_shots_on_target(), _load_partial(), MatchDetailsBuilder (+49 more)

### Community 10 - "Community 10"
Cohesion: 0.06
Nodes (56): build_match_details_csv(), build_matches_csv(), build_teams_csv(), Build teams.csv and write to disk.      Args:         output_path: Where to writ, Build matches.csv and write to disk.      Args:         output_path: Where to wr, Build match_details.csv and write to disk.      Args:         output_path: Where, train(), ModelEvaluator (+48 more)

### Community 11 - "Community 11"
Cohesion: 0.1
Nodes (30): matplotlib, bracket_config(), BracketConfig wrapping the 4-matchup simple_bracket., TestSerializeBracketToml, TestValidateBracket, _make_round_results(), Tests for ResultPlotter — bracket and probability visualizations.  Uses the Agg, Empty matchup list should be handled gracefully (early return). (+22 more)

### Community 12 - "Community 12"
Cohesion: 0.13
Nodes (10): _AlwaysAwayClassifier, _AlwaysHomeClassifier, Tests for GamePredictor — single-match knockout tie prediction., The classifier must receive a (1, 10) feature matrix., Always predicts home win (1)., Always predicts away win (0)., Always-home classifier must produce the same result every call., Unknown team name must raise KeyError from the distributions lookup. (+2 more)

### Community 13 - "Community 13"
Cohesion: 0.15
Nodes (13): BracketSlot, build_bracket_tree(), extract_bracket_slots(), Monte Carlo single-elimination tournament simulation.  Mirrors ``nba_playoff.sim, Return ``{round_number: [slots]}`` by BFS from root.      Round 0 = leaves (firs, A node in the bracket tree.      Leaves hold the two teams from a first-round ti, Build a bracket tree from the sequential matchup list.      Mirrors the pairing, _make_4_matchup_bracket() (+5 more)

### Community 14 - "Community 14"
Cohesion: 0.09
Nodes (20): GamePredictor, Knockout tie prediction using sampled features and a trained classifier.  Mirror, Predicts the winner of a single knockout tie between two national teams.      Us, Runs Monte Carlo simulations of a single-elimination knockout bracket.      Uses, Run *n_simulations* full single-elimination brackets.          Args:, Play one full bracket and mutate *rounds* to accumulate counts.          Args:, Decide a single knockout tie via the ``GamePredictor``.          This replaces t, TournamentSimulator (+12 more)

### Community 15 - "Community 15"
Cohesion: 0.4
Nodes (3): Simulate a single knockout tie and return the advancing team.          Samples o, Resolve a team's fitted distributions, falling back to the pool.          Sparse, Resolve a team's fitted distributions, falling back to the pool.          Sparse

### Community 16 - "Community 16"
Cohesion: 0.05
Nodes (5): cache_dir(), Source-blind example tests for issue #2: no-key martj42 historical loader.  Test, Filter invariant: wc2026_schedule never admits non-FIFA-World-Cup rows., Populate tmp_path with martj42 fixture CSVs; loader reads from here, no HTTP nee, test_when_wc2026_schedule_filters_any_results_df_then_all_rows_are_world_cup()

### Community 17 - "Community 17"
Cohesion: 0.08
Nodes (27): is_known(), normalize_series(), normalize_team(), Team-name normalization: maps alias spellings to canonical country names., Return the canonical country name for *name*, or the cleaned input if unknown., Vectorized version of *normalize_team*; preserves index and length., Return True when *name* maps to a canonical country name., Source-blind example tests for the team-name crosswalk module (issue #1).  Tests (+19 more)

### Community 18 - "Community 18"
Cohesion: 0.15
Nodes (12): Tracks per-team advancement counts at a given round.      Each team's probabilit, RoundResult, test_n_simulations_negative_raises_validation_error(), test_n_simulations_zero_raises_validation_error(), test_simulation_config_default_classifier(), test_valid_n_simulations_is_accepted(), Each team's probability is count / n_simulations (not count / total_wins)., RoundResult must expose round_num as documented. (+4 more)

### Community 19 - "Community 19"
Cohesion: 0.14
Nodes (13): mock_classifier(), Shared test fixtures for the worldcup_playoff test suite., Pre-built distributions for four national teams.      Five features each (goals,, Minimal 4-matchup bracket (power of 2 — 8 teams)., Single matchup bracket (minimum valid bracket)., A fake classifier that always predicts home win (1)., A temporary directory tree that mirrors the project layout., Minimal train_data.csv DataFrame with the correct schema.      Contains enough r (+5 more)

### Community 20 - "Community 20"
Cohesion: 0.06
Nodes (19): Tests for Issue #4 — WC2026 Round-of-32 bracket-slotting rules.  Source-blind: a, Criterion: R32_SLOTS encodes all 24 group-position slots (1A–1L, 2A–2L).     Eac, R32_SLOTS must include exactly 8 third-place placeholder slots., R32_SLOTS must be immutable (frozen tuple, frozenset, or similar)., Non-third-place slots must be position digit (1 or 2) + valid group letter., Each combination value maps exactly 8 third-place placeholder slots., The union of assigned groups in each value must equal the qualifying set in its, The data package must be importable with no side-effects. (+11 more)

### Community 21 - "Community 21"
Cohesion: 0.18
Nodes (11): vg_config, bon_n, dars_branching, gvr_enabled, judge_enabled, mav_aspects, orps_scoring, rex_scheduler (+3 more)

### Community 22 - "Community 22"
Cohesion: 0.29
Nodes (4): _make_valid_train_df(), HOME_TEAM + AWAY_TEAM + 10 features + HOME_WIN = 13 columns., Build a valid train_data.csv DataFrame with the full schema., TestTrainDataSchema

### Community 23 - "Community 23"
Cohesion: 0.07
Nodes (28): prompt_hashes, prompt-analyze-coverage.md, prompt-clarify-requirements.md, prompt-cycle-specializer.md, prompt-judge.md, prompt-ldb-debug.md, prompt-mav-acceptance-criteria.md, prompt-mav-runtime.md (+20 more)

### Community 24 - "Community 24"
Cohesion: 0.2
Nodes (5): Data loading and API client utilities for the World Cup prediction pipeline., Public API for the worldcup_playoff simulation subpackage.  Exports all types an, Test suite for worldcup_playoff., Visualization subpackage for World Cup 2026 knockout bracket prediction., FIFA World Cup 2026 knockout prediction via Monte Carlo simulation.

### Community 25 - "Community 25"
Cohesion: 0.4
Nodes (5): repo, full, host, name, owner

### Community 26 - "Community 26"
Cohesion: 0.13
Nodes (14): DataCleaner, Clean and preprocess raw match data into analysis-ready training data.  Output s, Remove drawn matches where HOME_GOALS == AWAY_GOALS.          A knockout tie alw, Merge match_details statistics onto the main DataFrame.          If ``details_df, Replace zero values in percentage columns with epsilon.          Covers ``POSSES, Drop rows with NaN in any feature column., Keep only matches before the training cutoff date., Add binary HOME_WIN column: 1 if HOME_GOALS > AWAY_GOALS else 0. (+6 more)

### Community 62 - "Community 62"
Cohesion: 0.11
Nodes (20): load_martj42_goalscorers(), load_martj42_results(), load_martj42_shootouts(), Martj42Loader, Cache-first loader for the martj42/international_results CC0 datasets., Return the results DataFrame coerced to the internal schema., Return the shootouts DataFrame coerced to the internal schema., Return the goalscorers DataFrame coerced to the internal schema. (+12 more)

### Community 63 - "Community 63"
Cohesion: 0.17
Nodes (7): LiveTournamentAdapter, Fetches live WC2026 data and assembles a TournamentState., _client_returning_matches(), _client_returning_standings(), Parsing a LAST_32 match with both team names null must never crash., TestLiveTournamentAdapterInterface, TestNullTeamParsing

### Community 64 - "Community 64"
Cohesion: 0.09
Nodes (22): Additional Notes, Bookmaker odds (backtest baseline only — NOT a feature), Caveats, code:block1 (worldcup_playoff/), Data Contracts (probed live — exact schemas, build to these), Data stack (no-key training; key only for live), Description, Elo — COMPUTE from martj42 (no clean eloratings.net API) (+14 more)

### Community 65 - "Community 65"
Cohesion: 0.13
Nodes (12): _extract_fields(), GroupStanding, LiveMatch, Live WC2026 tournament state adapter over football-data.org v4., A single match from the football-data.org matches endpoint., One team's row in a group standings table., Standings for a single WC group., Snapshot of the WC2026 tournament as of today. (+4 more)

### Community 66 - "Community 66"
Cohesion: 0.12
Nodes (16): code:block17 (football-data.org v4                Raw CSVs                ), code:block18 (worldcup_playoff/), code:bash (pytest                            # all tests), code:bash (# Run the full pipeline: clean → train → fit → simulate → vi), Features, FIFA World Cup 2026 Knockout Prediction, Key Design Decisions, License (+8 more)

### Community 67 - "Community 67"
Cohesion: 0.18
Nodes (8): _client_for_tournament_state(), Mock whose .get() yields matches JSON then standings JSON (two calls)., FINISHED group-stage match (id=101) must land in TournamentState.played., SCHEDULED group-stage match (id=102) must land in remaining_group_fixtures., LAST_32 match (id=103) is not a group-stage match → must not appear in played., LAST_32 match (id=103) must not appear in remaining_group_fixtures either., WC2026 has 12 groups — all must be present in TournamentState.standings., TestTournamentStatePartitioning

### Community 68 - "Community 68"
Cohesion: 0.17
Nodes (12): DataLoader, Load matches.csv into a validated DataFrame.          Returns:             DataF, Load teams.csv into a validated DataFrame.          Returns:             DataFra, Reads raw CSV files into DataFrames with column validation., Tests for DataLoader column validation., test_load_matches_raises_on_schema_mismatch(), test_load_matches_raises_when_path_is_none(), test_load_matches_reads_csv() (+4 more)

### Community 69 - "Community 69"
Cohesion: 0.24
Nodes (7): Load raw CSV datasets into validated DataFrames., Check column presence and dtypes, raising ``ValueError`` on mismatch.      Args:, _validate_dataframe(), validate_matches_df(), _make_valid_matches_df(), Tests for DataFrame schema validation — train_data.csv column contract., TestMatchesDfSchema

### Community 70 - "Community 70"
Cohesion: 0.2
Nodes (15): Validate that a results DataFrame conforms to the martj42 results schema., validate_results_df(), _make_valid_results_df(), Criterion explicitly states the validator must accept Int64., Criterion explicitly states the validator must accept bool., Missing-column invariant for results validator., Minimal DataFrame matching the internal results schema from the criteria., test_when_any_required_results_column_is_absent_then_value_error_is_raised() (+7 more)

### Community 71 - "Community 71"
Cohesion: 0.19
Nodes (15): rank_third_places(), Return the 8 qualifying third-place group letters ranked by FIFA tiebreak., Points are the primary FIFA tiebreak criterion., Goal difference is the secondary tiebreak criterion., Goals for is the tertiary tiebreak criterion., rank_third_places must be deterministic: identical input → identical output., All 12 groups with identical third-place statistics (ties broken by group letter, test_when_same_standings_given_twice_then_rank_third_places_returns_identical_result() (+7 more)

### Community 72 - "Community 72"
Cohesion: 0.18
Nodes (13): _augment(), _bipartite_match(), _compute_third_place_combinations(), _get_row_at(), WC2026 Round-of-32 bracket-slotting rules.  Pure data + pure functions. No netwo, Enumerate all valid 8-of-12 qualifying-third combinations via bipartite matching, Return the row whose `position` field equals *position*, or fall back by index., FIFA tiebreak key for third-placed teams (lower = better rank). (+5 more)

### Community 74 - "Community 74"
Cohesion: 0.19
Nodes (12): No-key CC0 loader for martj42/international_results datasets., Raise ValueError for missing columns or incompatible dtypes.      Extends loader, Validate that a goalscorers DataFrame conforms to the martj42 goalscorers schema, Return rows where TOURNAMENT == 'FIFA World Cup', including unplayed fixtures., validate_goalscorers_df(), _validate_martj42(), wc2026_schedule(), _make_valid_goalscorers_df() (+4 more)

### Community 75 - "Community 75"
Cohesion: 0.23
Nodes (13): Resolve 16 R32 ties from final group standings to concrete (home, away) pairs., resolve_r32(), No team appears more than once across the 16 ties., The 8 qualifying thirds (not the 4 eliminated) must fill the third-place slots., Build standings whose top-8 thirds exactly match the first entry in     THIRD_PL, _standings_for_first_known_combo(), test_when_standings_given_to_resolve_r32_then_all_12_group_winners_appear(), test_when_standings_given_to_resolve_r32_then_all_12_runners_up_appear() (+5 more)

### Community 76 - "Community 76"
Cohesion: 0.15
Nodes (13): `bracket`, `clean`, code:bash (worldcup-playoff clean), code:bash (worldcup-playoff train --classifier all          # svm, rand), code:bash (worldcup-playoff fit), code:bash (worldcup-playoff simulate --bracket config/playoff_2026.toml), code:bash (worldcup-playoff bracket --bracket config/playoff_2026.toml ), code:bash (worldcup-playoff run --bracket config/playoff_2026.toml) (+5 more)

### Community 77 - "Community 77"
Cohesion: 0.29
Nodes (6): Validate that a matches DataFrame has the required schema., Validate that a teams DataFrame has the required schema., validate_teams_df(), _minimal_valid_teams_df(), TestValidateTeamsDf, TestTeamsDfSchema

### Community 78 - "Community 78"
Cohesion: 0.24
Nodes (6): _minimal_valid_matches_df(), When a column name is close to 'HOME_TEAM', the error should suggest it., When a column name is close to 'HOME_TEAM', the error should suggest it., int64 and float64 are compatible numeric types — should not raise., int64 and float64 are compatible numeric types — should not raise., TestValidateMatchesDf

### Community 79 - "Community 79"
Cohesion: 0.17
Nodes (12): `build-match-details`, `build-matches`, `build-teams`, CLI Reference, code:bash (worldcup-playoff generate-bracket --season 2026), code:bash (worldcup-playoff download --seasons 2006-2026 --output-dir d), code:bash (worldcup-playoff build-teams --output-dir dataset/csv), code:bash (worldcup-playoff build-matches --start-year 2006 --end-year ) (+4 more)

### Community 80 - "Community 80"
Cohesion: 0.22
Nodes (6): BaseModel, LiveConfig, NaiveBayesConfig, Configuration loading and validation via Pydantic models., Configuration for the live WC2026 football-data.org adapter., SVMConfig

### Community 81 - "Community 81"
Cohesion: 0.18
Nodes (9): Architecture, code:bash (# Install (editable, with dev tools)), code:block2 (worldcup_playoff/), Commands, Configuration, Key design choices, Package layout, Project Overview (+1 more)

### Community 82 - "Community 82"
Cohesion: 0.18
Nodes (11): cache_read, cache_write, cache_write_1h, cache_write_5m, clear_events, compaction_events, input, num_turns (+3 more)

### Community 83 - "Community 83"
Cohesion: 0.25
Nodes (11): assign_thirds(), Map each 3X placeholder to a concrete group letter via THIRD_PLACE_COMBINATIONS., _first_combo(), Return the first key from THIRD_PLACE_COMBINATIONS as a frozenset., Every group in the qualifying frozenset must appear as an assigned value., Each placeholder returned must reference a slot that exists in R32_SLOTS., test_when_valid_combination_given_to_assign_thirds_then_all_qualifying_groups_are_assigned(), test_when_valid_combination_given_to_assign_thirds_then_dict_with_8_entries_is_returned() (+3 more)

### Community 84 - "Community 84"
Cohesion: 0.24
Nodes (5): fetch_tournament_state(), Build a TournamentState, creating a default FootballClient when none is passed., The default value for the `competition` parameter must be 'WC'., The default value for the `client` parameter must be None., TestFetchTournamentStateFunction

### Community 85 - "Community 85"
Cohesion: 0.2
Nodes (9): 0. Where we are now (and the problem), 1. Prior work (who has done this) — reference, 2. Football modeling logic — what to internalize, 3. How to build it — practical pipeline, 4. Data & providers — NO API key, 5. Recommended upgrade (the actual plan), 6. Caveats (don't oversell), 7. Sources (+1 more)

### Community 86 - "Community 86"
Cohesion: 0.22
Nodes (8): Acceptance criteria, Cycle 1 — Data Foundations, Following cycles, In scope, Objective, Out of scope, Preceding cycles, Project vision

### Community 87 - "Community 87"
Cohesion: 0.22
Nodes (8): Acceptance criteria, Cycle 2 — Statistical Abilities, Following cycles, In scope, Objective, Out of scope, Preceding cycles, Project vision

### Community 88 - "Community 88"
Cohesion: 0.22
Nodes (8): Acceptance criteria, Cycle 3 — Feature Assembly, Following cycles, In scope, Objective, Out of scope, Preceding cycles, Project vision

### Community 89 - "Community 89"
Cohesion: 0.22
Nodes (8): Acceptance criteria, Cycle 4 — Model Training & Comparison, Following cycles, In scope, Objective, Out of scope, Preceding cycles, Project vision

### Community 90 - "Community 90"
Cohesion: 0.22
Nodes (8): Acceptance criteria, Cycle 5 — Evaluation, Simulation & Integration, Following cycles, In scope, Objective, Out of scope, Preceding cycles, Project vision

### Community 91 - "Community 91"
Cohesion: 0.22
Nodes (9): _group_table(), Invariant: rank_third_places returns exactly 8 distinct valid groups for any inp, Build one row of a group table matching the football-data.org v4 schema., 4-row group table; third-placed team ('Team{letter}3') has the given stats., First *n* groups (A … nth) have clearly superior third-placed teams;     the rem, _standings_first_n_superior(), _table_row(), test_when_any_valid_standings_given_then_rank_third_places_always_returns_8_distinct_valid_groups() (+1 more)

### Community 92 - "Community 92"
Cohesion: 0.32
Nodes (7): Validate that a shootouts DataFrame conforms to the martj42 shootouts schema., validate_shootouts_df(), _make_valid_shootouts_df(), Missing-column invariant for shootouts validator., test_when_any_required_shootouts_column_is_absent_then_value_error_is_raised(), test_when_shootouts_df_is_missing_winner_then_value_error_is_raised(), test_when_valid_shootouts_df_is_validated_then_no_error_is_raised()

### Community 93 - "Community 93"
Cohesion: 0.33
Nodes (6): _make_group_match(), _make_standings_json(), Source-blind example tests for issue #3:   feat: live WC2026 adapter over Footba, Invariant from AC2: played ∪ remaining_group_fixtures must cover every group-sta, Return a standings payload with *n_groups* groups (default 12 for WC2026)., test_when_group_stage_matches_have_mixed_statuses_then_partition_covers_all_of_them()

### Community 94 - "Community 94"
Cohesion: 0.33
Nodes (6): API key, code:bash (git clone https://github.com/SilvioBaratto/worldcup-2026-pre), code:bash (pip install -e ".[dev]"), code:bash (pip install -r requirements.txt), code:bash (export FOOTBALL_DATA_API_KEY="your-token-here"), Installation

### Community 95 - "Community 95"
Cohesion: 0.4
Nodes (4): 1. [info] coverage-gap, Analyze Gate Report, Findings, Fix Hint

### Community 97 - "Community 97"
Cohesion: 0.5
Nodes (4): code:toml (name = "2026 FIFA World Cup — Knockout (Round of 32)"), `config/default.toml`, `config/playoff_*.toml`, Configuration

### Community 98 - "Community 98"
Cohesion: 0.67
Nodes (3): Elo Engine, Groll RF Hybrid Model, Dixon-Coles Poisson Model

## Knowledge Gaps
- **721 isolated node(s):** `Configuration loading and validation via Pydantic models.`, `Configuration for the no-key martj42 CC0 historical results loader.`, `Per-team match statistics fed to the classifier.      Five features per team (ho`, `Knockout simulation settings.      Unlike the NBA best-of-7 series, World Cup kn`, `Configuration for the live WC2026 football-data.org adapter.` (+716 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **48 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config Models` connect `Community 3` to `Community 1`, `Community 2`, `Community 68`, `Community 5`, `Community 6`, `Community 69`, `Community 4`, `Community 9`, `Community 10`, `Community 11`, `Community 74`, `Community 13`, `Community 14`, `Community 12`, `Community 16`, `Community 19`, `Community 7`, `Community 26`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Why does `Pipeline` connect `Community 10` to `Community 2`, `Community 3`, `Community 68`, `Community 5`, `Community 4`, `Community 9`, `Community 11`, `Community 14`, `Community 18`, `Community 26`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Why does `FootballClient` connect `Community 2` to `Community 65`, `Community 3`, `Community 6`, `Community 9`, `Community 10`, `Community 84`, `Community 63`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Are the 54 inferred relationships involving `FootballClient` (e.g. with `Pipeline` and `TeamsBuilder`) actually correct?**
  _`FootballClient` has 54 INFERRED edges - model-reasoned connections that need verification._
- **Are the 58 inferred relationships involving `Matchup` (e.g. with `ResultPlotter` and `RoundResult`) actually correct?**
  _`Matchup` has 58 INFERRED edges - model-reasoned connections that need verification._
- **Are the 54 inferred relationships involving `ClientConfig` (e.g. with `TeamsBuilder` and `MatchesBuilder`) actually correct?**
  _`ClientConfig` has 54 INFERRED edges - model-reasoned connections that need verification._
- **Are the 44 inferred relationships involving `Pipeline` (e.g. with `AppConfig` and `BracketConfig`) actually correct?**
  _`Pipeline` has 44 INFERRED edges - model-reasoned connections that need verification._