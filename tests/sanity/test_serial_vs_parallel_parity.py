"""Sanity test: serial and parallel workflows produce identical parquet output.

This test runs the full model2obs workflow twice — once in serial mode using
``config_tutorial_1.yaml`` and once in parallel mode using
``config_tutorial_1_parallel.yaml`` — and asserts that every row in the
resulting parquet tables is identical (order-independent).

**This test is intentionally excluded from the default pytest run** because it
requires tutorial data, a compiled DART installation, and significant wall-clock
time.  Run it explicitly as an ad-hoc sanity check:

    pytest tests/sanity/ -s -v

Prerequisites:
- ``$TUTORIAL_DATA_PATH`` environment variable must point to the directory that
  contains the downloaded tutorial datasets.
- ``$DART_ROOT_PATH`` environment variable must point to the DART installation
  root (the ``models/MOM6/work`` subdirectory must contain the compiled
  ``perfect_model_obs`` executable).
- Tutorial data must already be downloaded.  If not, run::

      download_tutorials_data --destination <TUTORIAL_DATA_PATH>

When the parquet comparison fails, three automatic diagnostic steps run and
print to stdout (visible with ``-s``):

1. **obs_seq.out comparison** – Each ``obs_seq_NNNN.out`` file is loaded with
   ``pydartdiags`` and compared pair-by-pair.  Files are reported as
   ``MATCH``, ``MISMATCH``, ``EXTRA`` (only in one workflow), or ``MISSING``.

2. **Model input equivalence** – The serial single-file ``in_mom6/`` dataset
   is compared with the concatenated ``in_mom6_par/`` multi-file dataset using
   xarray.  Any differing variables are listed.

3. **Failure statistics** – Rows that differ between the two parquet tables are
   summarised by observation type, day, and QC code.  Note: the parquet does
   not store a thread-number column, so per-thread attribution is not available.
"""

import os
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest
import xarray as xr

import pydartdiags.obs_sequence.obs_sequence as obsq

from model2obs.utils.config import read_config
from model2obs.workflows import WorkflowModelObs

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TUTORIALS_DIR = Path(__file__).parents[2] / "tutorials"
_SERIAL_CONFIG = _TUTORIALS_DIR / "config_tutorial_1.yaml"
_PARALLEL_CONFIG = _TUTORIALS_DIR / "config_tutorial_1_parallel.yaml"

# Subdirectories that must exist inside $TUTORIAL_DATA_PATH/tutorial_1/
_REQUIRED_INPUT_SUBDIRS = [
    "tutorial_1/in_mom6",
    "tutorial_1/in_mom6_par",
    "tutorial_1/in_mom6_ref",
    "tutorial_1/in_CrocoLake",
]

# Columns used to sort rows before comparison so order does not matter
_SORT_COLS = ["time", "latitude", "longitude", "vertical", "type"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_tutorial_data() -> None:
    """Check all prerequisites; call pytest.fail() with instructions if any are missing."""
    tutorial_data_path = os.environ.get("TUTORIAL_DATA_PATH")
    dart_root_path = os.environ.get("DART_ROOT_PATH")

    missing = []
    if not tutorial_data_path:
        missing.append("  - TUTORIAL_DATA_PATH is not set")
    if not dart_root_path:
        missing.append("  - DART_ROOT_PATH is not set")

    if missing:
        pytest.fail(
            "Required environment variables are not set:\n"
            + "\n".join(missing)
            + "\n\nSet them before running this test, for example:\n"
            "  export TUTORIAL_DATA_PATH=/path/to/tutorial/data\n"
            "  export DART_ROOT_PATH=/path/to/DART"
        )

    base = Path(tutorial_data_path)
    missing_dirs = [
        d for d in _REQUIRED_INPUT_SUBDIRS
        if not (base / d).is_dir()
    ]
    if missing_dirs:
        pytest.fail(
            "The following tutorial data directories are missing under "
            f"$TUTORIAL_DATA_PATH ({tutorial_data_path}):\n"
            + "\n".join(f"  - {d}" for d in missing_dirs)
            + "\n\nDownload the tutorial data first:\n"
            f"  download_tutorials_data --destination {tutorial_data_path}"
        )

    dart_pmo = Path(dart_root_path) / "models" / "MOM6" / "work" / "perfect_model_obs"
    if not dart_pmo.exists():
        pytest.fail(
            f"DART perfect_model_obs executable not found at:\n  {dart_pmo}\n\n"
            "Ensure DART is compiled for MOM6 at $DART_ROOT_PATH."
        )


def _load_sorted_parquet(folder: str) -> pd.DataFrame:
    """Read all parquet files from *folder*, sort rows deterministically, reset index.

    Args:
        folder: Path to the directory containing ``*.parquet`` files.

    Returns:
        A pandas DataFrame with rows sorted by ``_SORT_COLS`` and a clean
        integer index.
    """
    parquet_files = list(Path(folder).glob("*.parquet"))
    if not parquet_files:
        pytest.fail(f"No parquet files found in: {folder}")

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    sort_cols = [c for c in _SORT_COLS if c in df.columns]
    return df.sort_values(by=sort_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Failure diagnostics
# ---------------------------------------------------------------------------

def _compare_obs_seq_files(serial_output_folder: str, parallel_output_folder: str) -> None:
    """Compare ``obs_seq_NNNN.out`` files pair-by-pair between the two workflows.

    Each file is loaded with ``pydartdiags.ObsSequence``; the resulting
    DataFrames are compared.  Results are printed to stdout as
    MATCH / MISMATCH / EXTRA / MISSING per counter.

    Args:
        serial_output_folder: Path to the serial workflow ``output_folder``.
        parallel_output_folder: Path to the parallel workflow ``output_folder``.
    """
    print("\n=== DIAGNOSTIC 1: obs_seq.out file comparison ===")
    serial_dir = Path(serial_output_folder)
    parallel_dir = Path(parallel_output_folder)

    def _counter(p: Path) -> str:
        """Extract the 4-digit counter from a filename like obs_seq_0003.out."""
        return p.stem.split("_")[-1]

    serial_files: Dict[str, Path] = {
        _counter(f): f for f in sorted(serial_dir.glob("obs_seq_*.out"))
    }
    parallel_files: Dict[str, Path] = {
        _counter(f): f for f in sorted(parallel_dir.glob("obs_seq_*.out"))
    }

    all_counters = sorted(set(serial_files) | set(parallel_files))
    if not all_counters:
        print("  No obs_seq_*.out files found in either workflow output folder.")
        return

    mismatch_count = 0
    for ctr in all_counters:
        if ctr not in serial_files:
            print(f"  obs_seq_{ctr}.out  EXTRA  (only in parallel)")
            mismatch_count += 1
            continue
        if ctr not in parallel_files:
            print(f"  obs_seq_{ctr}.out  MISSING from parallel")
            mismatch_count += 1
            continue

        try:
            s_seq = obsq.ObsSequence(str(serial_files[ctr]))
            p_seq = obsq.ObsSequence(str(parallel_files[ctr]))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  obs_seq_{ctr}.out  ERROR loading: {exc}")
            mismatch_count += 1
            continue

        s_df = s_seq.df.reset_index(drop=True)
        p_df = p_seq.df.reset_index(drop=True)

        n_serial = len(s_df)
        n_parallel = len(p_df)
        if s_df.equals(p_df):
            print(f"  obs_seq_{ctr}.out  MATCH  ({n_serial} rows)")
        else:
            if s_df.shape == p_df.shape:
                n_diff = int((s_df != p_df).any(axis=1).sum())
                diff_label = f"~{n_diff} differing rows"
            else:
                n_diff = abs(n_serial - n_parallel)
                diff_label = (
                    f"row count differs by {n_diff} "
                    f"— full content comparison skipped; see Diagnostic 3"
                )
            print(
                f"  obs_seq_{ctr}.out  MISMATCH  "
                f"(serial={n_serial} rows, parallel={n_parallel} rows, "
                f"{diff_label})"
            )
            mismatch_count += 1

    if mismatch_count == 0:
        print("  All obs_seq.out files match. The difference must be in the merge step.")
    else:
        print(f"  {mismatch_count}/{len(all_counters)} file(s) differ.")


def _compare_model_inputs(serial_config: dict, parallel_config: dict) -> None:
    """Compare serial and parallel model input datasets using xarray.

    Serial workflow uses a single file from ``in_mom6/``; parallel workflow
    uses multiple files from ``in_mom6_par/`` (same data, split differently).
    Both are sorted by time before comparison.

    Args:
        serial_config: Resolved config dict for the serial workflow.
        parallel_config: Resolved config dict for the parallel workflow.
    """
    print("\n=== DIAGNOSTIC 2: Model input data equivalence ===")
    serial_folder = serial_config.get("model_files_folder", "")
    parallel_folder = parallel_config.get("model_files_folder", "")

    serial_files = sorted(Path(serial_folder).glob("*.nc"))
    parallel_files = sorted(Path(parallel_folder).glob("*.nc"))

    if not serial_files:
        print(f"  No .nc files found in serial model folder: {serial_folder}")
        return
    if not parallel_files:
        print(f"  No .nc files found in parallel model folder: {parallel_folder}")
        return

    print(f"  Serial:   {len(serial_files)} file(s) in {serial_folder}")
    for f in serial_files:
        print(f"    - {f.name}")
    print(f"  Parallel: {len(parallel_files)} file(s) in {parallel_folder}")
    for f in parallel_files:
        print(f"    - {f.name}")

    try:
        ds_serial = xr.open_mfdataset(
            [str(f) for f in serial_files], combine="by_coords"
        ).sortby("time")
        ds_parallel = xr.open_mfdataset(
            [str(f) for f in parallel_files], combine="by_coords"
        ).sortby("time")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  ERROR opening datasets: {exc}")
        return

    # Check that the two have the same variables
    serial_vars = set(ds_serial.data_vars)
    parallel_vars = set(ds_parallel.data_vars)
    if serial_vars != parallel_vars:
        print(f"  Variable mismatch: serial={serial_vars - parallel_vars} extra, "
              f"parallel={parallel_vars - serial_vars} extra")
        return

    overall_equal = ds_serial.equals(ds_parallel)
    if overall_equal:
        print("  Model input data are IDENTICAL across serial and parallel inputs.")
    else:
        print("  Model input data DIFFER. Checking variable by variable:")
        for var in sorted(serial_vars):
            try:
                if not ds_serial[var].equals(ds_parallel[var]):
                    print(f"    {var}: DIFFERS")
                else:
                    print(f"    {var}: ok")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"    {var}: ERROR comparing ({exc})")

    ds_serial.close()
    ds_parallel.close()


def _report_diff_statistics(serial_df: pd.DataFrame, parallel_df: pd.DataFrame) -> None:
    """Print statistics about rows that differ between the two parquet tables.

    Performs an outer-join on ``(time, latitude, longitude, vertical, type)``
    and reports mismatching rows grouped by observation type, day, and QC code,
    broken down by source (serial-only / parallel-only / value-mismatch).

    Note: the parquet output does not include a thread-number column, so
    per-thread attribution is not possible from the parquet alone.

    Args:
        serial_df: Sorted parquet DataFrame from the serial workflow.
        parallel_df: Sorted parquet DataFrame from the parallel workflow.
    """
    print("\n=== DIAGNOSTIC 3: Failure statistics ===")
    print(
        "  Note: Diagnostic 1 reports the row-count difference between matching\n"
        "  obs_seq files (e.g. |2527-2534| = 7).  Diagnostic 3 outer-joins on\n"
        "  (time, latitude, longitude, vertical, type); it may show much larger\n"
        "  numbers when the model time stored in the parquet differs between\n"
        "  workflows for the same observations (e.g. if obs_seq_0003 was matched\n"
        "  to a different model snapshot in serial vs. parallel)."
    )

    join_keys = [c for c in _SORT_COLS if c in serial_df.columns and c in parallel_df.columns]

    merged = pd.merge(
        serial_df,
        parallel_df,
        on=join_keys,
        how="outer",
        suffixes=("_serial", "_parallel"),
        indicator=True,
    )

    only_serial = merged[merged["_merge"] == "left_only"]
    only_parallel = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"]

    # Always print both counts so the reader can see 0 explicitly
    print(f"\n  Rows ONLY in serial output:   {len(only_serial)}")
    print(f"  Rows ONLY in parallel output: {len(only_parallel)}")

    # For rows present in both, find value differences in key numeric columns
    counterparts = {
        "interpolated_model_serial": "interpolated_model_parallel",
        "obs_serial": "obs_parallel",
    }
    diff_mask = pd.Series(False, index=both.index)
    for col, other in counterparts.items():
        if col in both.columns and other in both.columns:
            diff_mask |= (both[col] - both[other]).abs() > 1e-12

    value_diffs = both[diff_mask]
    print(f"  Rows present in both but with different values: {len(value_diffs)}")

    if only_serial.empty and only_parallel.empty and value_diffs.empty:
        print("  (No differing rows found via outer-join — difference may be in row count or dtypes.)")
        return

    # Helper: count a column per group for each of the three row sources
    def _breakdown(col: str, label: str) -> None:
        sources = [
            ("serial-only",   only_serial,  col if col in only_serial.columns   else col + "_serial"),
            ("parallel-only", only_parallel, col if col in only_parallel.columns else col + "_parallel"),
            ("value-mismatch", value_diffs,  col if col in value_diffs.columns   else col + "_serial"),
        ]
        # Collect all unique values across sources
        all_vals: set = set()
        for _, df_src, resolved_col in sources:
            if not df_src.empty and resolved_col in df_src.columns:
                all_vals.update(df_src[resolved_col].dropna().unique())
        if not all_vals:
            return
        all_vals_sorted = sorted(all_vals, key=str)

        print(f"\n  Differing rows by {label}:")
        col_w = max(len(str(v)) for v in all_vals_sorted)
        header = f"    {'':>{col_w}}  {'serial-only':>12}  {'parallel-only':>13}  {'value-mismatch':>14}"
        print(header)
        for val in all_vals_sorted:
            counts = []
            for _, df_src, resolved_col in sources:
                if not df_src.empty and resolved_col in df_src.columns:
                    counts.append(int((df_src[resolved_col] == val).sum()))
                else:
                    counts.append(0)
            print(f"    {str(val):>{col_w}}  {counts[0]:>12}  {counts[1]:>13}  {counts[2]:>14}")

    # Observation type breakdown
    if "type" in merged.columns:
        _breakdown("type", "observation type")

    # Day-level breakdown
    if "time" in merged.columns:
        for df_part in [only_serial, only_parallel, value_diffs]:
            if not df_part.empty and "time" in df_part.columns:
                df_part = df_part.copy()
                df_part["_day"] = pd.to_datetime(df_part["time"]).dt.date
        # Rebuild sources with _day column
        def _add_day(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty or "time" not in df.columns:
                return df
            out = df.copy()
            out["_day"] = pd.to_datetime(out["time"]).dt.date
            return out

        only_serial_day   = _add_day(only_serial)
        only_parallel_day = _add_day(only_parallel)
        value_diffs_day   = _add_day(value_diffs)

        all_days: set = set()
        for df_src in [only_serial_day, only_parallel_day, value_diffs_day]:
            if not df_src.empty and "_day" in df_src.columns:
                all_days.update(df_src["_day"].dropna().unique())

        if all_days:
            all_days_sorted = sorted(all_days)
            print("\n  Differing rows by day:")
            print(f"    {'':>12}  {'serial-only':>12}  {'parallel-only':>13}  {'value-mismatch':>14}")
            for day in all_days_sorted:
                counts = []
                for df_src in [only_serial_day, only_parallel_day, value_diffs_day]:
                    if not df_src.empty and "_day" in df_src.columns:
                        counts.append(int((df_src["_day"] == day).sum()))
                    else:
                        counts.append(0)
                print(f"    {str(day):>12}  {counts[0]:>12}  {counts[1]:>13}  {counts[2]:>14}")

    # QC breakdown — separate table per source
    def _qc_table(df_src: pd.DataFrame, source_label: str, preferred_cols: list) -> None:
        if df_src.empty:
            return
        qc_col = next((c for c in preferred_cols if c in df_src.columns), None)
        if qc_col is None:
            return
        vc = df_src[qc_col].dropna().value_counts().sort_index()
        if vc.empty:
            return
        print(f"\n  QC code distribution — {source_label}:")
        for code, cnt in vc.items():
            print(f"    {code}: {cnt} rows")

    _qc_table(only_serial,  "serial-only rows",
              ["interpolated_model_QC_serial",   "interpolated_model_QC"])
    _qc_table(only_parallel, "parallel-only rows",
              ["interpolated_model_QC_parallel", "interpolated_model_QC"])
    _qc_table(value_diffs,   "value-mismatch rows (serial QC)",
              ["interpolated_model_QC_serial",   "interpolated_model_QC"])

    print(
        "\n  NOTE: The parquet output does not include a thread-number column,\n"
        "  so per-thread attribution of differing rows is not available here.\n"
        "  Check the per-file logs in the parallel output_folder for thread details."
    )


def _diagnose_failure(
    serial_config: dict,
    parallel_config: dict,
    serial_df: pd.DataFrame,
    parallel_df: pd.DataFrame,
) -> None:
    """Run all three diagnostic steps and print results to stdout.

    Call this inside an ``except AssertionError`` block; it does not re-raise.

    Args:
        serial_config: Resolved config dict for the serial workflow.
        parallel_config: Resolved config dict for the parallel workflow.
        serial_df: Sorted parquet DataFrame from the serial workflow.
        parallel_df: Sorted parquet DataFrame from the parallel workflow.
    """
    print("\n" + "=" * 70)
    print("PARITY TEST FAILED — running diagnostics …")
    print("=" * 70)

    try:
        _compare_obs_seq_files(
            serial_config["output_folder"],
            parallel_config["output_folder"],
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  Diagnostic 1 raised an unexpected error: {exc}")

    try:
        _compare_model_inputs(serial_config, parallel_config)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  Diagnostic 2 raised an unexpected error: {exc}")

    try:
        _report_diff_statistics(serial_df, parallel_df)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  Diagnostic 3 raised an unexpected error: {exc}")

    print("=" * 70 + "\n")

@pytest.mark.sanity
def test_serial_parallel_parquet_parity() -> None:
    """Serial and parallel workflows produce identical parquet output.

    Given: Tutorial data and DART are available
    When:  The serial workflow (config_tutorial_1.yaml) and the parallel
           workflow (config_tutorial_1_parallel.yaml) are both run
    Then:  Their parquet outputs contain the same rows in the same columns,
           regardless of row order

    If the assertion fails, three diagnostic steps run automatically and print
    structured information to stdout (pass ``-s`` to pytest to see them).
    """
    _require_tutorial_data()

    # Run serial workflow
    serial_workflow = WorkflowModelObs.from_config_file(str(_SERIAL_CONFIG))
    serial_workflow.run(clear_output=True)
    serial_config = serial_workflow.config
    serial_parquet_folder = serial_config["parquet_folder"]

    # Run parallel workflow
    parallel_workflow = WorkflowModelObs.from_config_file(str(_PARALLEL_CONFIG))
    parallel_workflow.run(clear_output=True, parallel=True)
    parallel_config = parallel_workflow.config
    parallel_parquet_folder = parallel_config["parquet_folder"]

    serial_df = _load_sorted_parquet(serial_parquet_folder)
    parallel_df = _load_sorted_parquet(parallel_parquet_folder)

    try:
        pd.testing.assert_frame_equal(
            serial_df.reset_index(drop=True),
            parallel_df.reset_index(drop=True),
            check_like=True,  # column order does not matter
            obj="Serial vs parallel parquet output",
        )
    except AssertionError:
        _diagnose_failure(serial_config, parallel_config, serial_df, parallel_df)
        raise
