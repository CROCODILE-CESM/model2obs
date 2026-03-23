"""Sanity test: serial and parallel workflows produce identical parquet output.

This test runs the full model2obs workflow twice — once in serial mode using
``config_tutorial_1.yaml`` and once in parallel mode using
``config_tutorial_1_parallel.yaml`` — and asserts that every row in the
resulting parquet tables is identical (order-independent).

**This test is intentionally excluded from the default pytest run** because it
requires tutorial data, a compiled DART installation, and significant wall-clock
time.  Run it explicitly as an ad-hoc sanity check:

    pytest tests/sanity/

Prerequisites:
- ``$TUTORIAL_DATA_PATH`` environment variable must point to the directory that
  contains the downloaded tutorial datasets.
- ``$DART_ROOT_PATH`` environment variable must point to the DART installation
  root (the ``models/MOM6/work`` subdirectory must contain the compiled
  ``perfect_model_obs`` executable).
- Tutorial data must already be downloaded.  If not, run::

      download_tutorials_data --destination <TUTORIAL_DATA_PATH>
"""

import os
from pathlib import Path

import pandas as pd
import pytest

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
# Sanity test
# ---------------------------------------------------------------------------

@pytest.mark.sanity
def test_serial_parallel_parquet_parity() -> None:
    """Serial and parallel workflows produce identical parquet output.

    Given: Tutorial data and DART are available
    When:  The serial workflow (config_tutorial_1.yaml) and the parallel
           workflow (config_tutorial_1_parallel.yaml) are both run
    Then:  Their parquet outputs contain the same rows in the same columns,
           regardless of row order
    """
    _require_tutorial_data()

    # Run serial workflow
    serial_workflow = WorkflowModelObs.from_config_file(str(_SERIAL_CONFIG))
    serial_workflow.run(clear_output=True)
    serial_parquet_folder = serial_workflow.get_config("parquet_folder")

    # Run parallel workflow
    parallel_workflow = WorkflowModelObs.from_config_file(str(_PARALLEL_CONFIG))
    parallel_workflow.run(clear_output=True, parallel=True)
    parallel_parquet_folder = parallel_workflow.get_config("parquet_folder")

    serial_df = _load_sorted_parquet(serial_parquet_folder)
    parallel_df = _load_sorted_parquet(parallel_parquet_folder)

    pd.testing.assert_frame_equal(
        serial_df.reset_index(drop=True),
        parallel_df.reset_index(drop=True),
        check_like=True,  # column order does not matter
        obj="Serial vs parallel parquet output",
    )
