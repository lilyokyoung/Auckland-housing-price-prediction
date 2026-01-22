Project overview for AI coding agents

This repo contains data-preparation scripts for an Auckland housing price project.
Most work happens in `src/data_engineering/` where small, script-style Python files
read Excel inputs and write cleaned CSVs into a local `Raw_data` directory.

Quick facts (what to know immediately)
- Main code area: `src/data_engineering/` (examples: `average_weeklyrent.py`, `population.py`, `building_consents.py`, `household_income.py`).
- These scripts use pandas and expect Excel/CSV files in a local Raw_data folder outside the repository root: e.g. `C:\Users\Lily0708\Final_project\Raw_data\...`.
- Many scripts use absolute Windows `Path(...)` strings and write cleaned outputs (CSV) back to the same Raw_data folder. Examples of produced files:
  - `Weeklyrent_merged.csv`
  - `population_mergedcl.csv`
  - `AllDistricts_consents.csv`
- `net_immigration.py` is present but empty — no behavior to assume.

Architecture / data flow
- Per-district Excel -> cleaned district CSV -> concatenation into long/merged CSV.
- Common pattern: load Excel with `pd.read_excel(...)`, transform columns (rename Date/Year, parse month), aggregate local boards into district-level series, then `to_csv(...)`.
- District ordering is explicit and repeated in multiple files (look for `district_order` lists in `average_weeklyrent.py`, `population.py`, `household_income.py`).

Project-specific conventions
- Files are script-like (not packaged): running a single file executes its ETL and prints progress.
- Naming: cleaned outputs typically use `*_cl.csv` or `*_merged.csv` suffixes.
- Date handling: scripts transform source Date strings like `YYYYMmm` or `YYYY` into `YYYY-MM` or integer `Year` values — look for regex-based masks and `pd.to_datetime(...).dt.strftime("%Y-%m")`.
- Aggregation: local board columns are summed or population-weighted (see `population_weighted_income` in `household_income.py`).

Dependencies & environment
- Only dependency discovered: `pandas`. There is no `requirements.txt` or virtualenv config in the repo.
- Platform: Windows paths are used in-source. Running on a different OS requires editing those `Path(...)` literals.

Developer workflows (how these scripts are intended to be run)
- Run a single data-prep script directly with Python (PowerShell example):

  ```powershell
  python src\data_engineering\average_weeklyrent.py
  python src\data_engineering\population.py
  ```

- Confirm input paths exist and contain the expected Excel files (example dir: `C:\Users\Lily0708\Final_project\Raw_data\population`).
- There are no unit tests or CI config discovered — manual verification is expected after running scripts.

Integration points / external data
- Scripts assume local raw data files exported from external sources (council datasets, StatsNZ). These files are not stored in the repo — they live under the `Raw_data` directory referenced in code.
- Output CSVs are the integration artifacts used downstream (notebooks or models) — check `outputs/` and `notebooks/` for consumer code (notebooks folder is currently empty).

When editing or extending
- If adding reusable modules or doing cross-platform work, convert absolute `Path(...)` literals to relative paths or environment-driven configuration (env var or a top-level `config.py`).
- Preserve district ordering lists when changing aggregation logic — many downstream steps rely on their ordering.

Files to inspect first when changing ETL
- `src/data_engineering/average_weeklyrent.py` — rent aggregation and long-format output.
- `src/data_engineering/population.py` — population cleaning and merged output.
- `src/data_engineering/building_consents.py` — month parsing and district-level consent aggregation.
- `src/data_engineering/household_income.py` — population-weighted income calculation pattern.

Questions for the repo owner
- Should raw input files be checked into a data/ or moved under the repo for reproducibility?
- Do you want scripts to accept a configurable data-root (env var / CLI) rather than hard-coded absolute paths?

If anything here is incomplete or unclear, tell me which area to expand (run commands, path conversion examples, or automated test scaffolding). 
