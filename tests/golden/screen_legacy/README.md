# screen_legacy golden files

Reference outputs of `psdl1.screen_relations` (CLI `main`) for the `case` fixture of `tests/test_screen_relations.py`.
Generated at commit `7794f37` (`feat/png-input`) by `tests/test_screen_legacy_golden.py`.

Scenarios: `a_train` (train, defaults), `b_test_fixed_floor` (test, `--floor` from a), `c_incremental` (delta rows, `--floor-json`),
`d_groups` (`--groups`, cross-group cap), `e_incremental_group_floor` (delta rows with a per-group floor).

The test compares `relations.csv` byte for byte. It compares the 18 legacy columns of `scores.csv` and the legacy keys of `summary.json` only,
so new columns and keys do not break it. Absolute paths are replaced by `<BASE_RELATIONS>`.

Regenerate only when the similarity or tier rule changes on purpose:

    PSDL1_REGEN_SCREEN_GOLDEN=1 pytest tests/test_screen_legacy_golden.py
