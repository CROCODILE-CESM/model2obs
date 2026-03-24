"""Model-observation comparison workflow for model2obs."""

import concurrent.futures
from datetime import datetime, timedelta
import glob
from importlib.resources import files
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pydartdiags.obs_sequence.obs_sequence as obsq
import warnings
import xarray as xr

from . import workflow
from ..io import file_utils
from ..io import model_tools  
from ..io import obs_seq_tools
from ..utils import config as config_utils
from ..utils import namelist

from dataclasses import dataclass
@dataclass(frozen=True)
class RunOptions:
    """Immutable record of the run-time flags for a workflow execution.

    Attributes:
        trim_obs: Trim obs_seq.in files to model grid boundaries before
            passing them to ``perfect_model_obs``.
        no_matching: Skip time-based obs/model file matching; assume a 1-to-1
            correspondence between model output files and obs files.
        force_obs_time: Override model snapshot time with the observation
            reference time when writing ``input.nml``.
    """

    # defaults chosen for the widest compatibility with tutorial data
    trim_obs: bool = False
    no_matching: bool = False
    force_obs_time: bool = False

class WorkflowModelObs(workflow.Workflow):
    """Model-observation comparison workflow.
    
    Orchestrates the comparison between ocean model outputs and observation datasets,
    including trimming observations to model grid boundaries, running perfect_model_obs,
    and converting results to parquet format for analysis.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize model-observation workflow with configuration.

        Args:
            config: Configuration dictionary containing workflow parameters
        """

        super().__init__(config)
        self.input_nml_template = files('model2obs.utils').joinpath('input_template.nml')
        self.model_obs_df = None
        self.perfect_model_obs_log_file = "perfect_model_obs.log"
        if os.path.isfile(self.perfect_model_obs_log_file):
            os.remove(self.perfect_model_obs_log_file)

    def run(self, trim_obs: bool = True, no_matching: bool = False,
            force_obs_time: bool = False, parquet_only: bool = False,
            clear_output: bool = False, parallel: bool = False) -> int:
        """Execute the complete model-observation workflow.
        
        Args:
            trim_obs: Whether to trim obs_seq.in files to model grid boundaries
            no_matching: Whether to skip time-matching and assume 1:1 correspondence
            force_obs_time: Whether to assign observations reference time to model files
            parquet_only: Whether to skip building perfect obs and directly convert to parquet
            clear_output: Whether to clear output folder before running the workflow (default: False)
            parallel: Whether to process model files in parallel using
                :class:`concurrent.futures.ThreadPoolExecutor`.  When ``True``,
                one worker thread is dispatched per model output file and pairs
                within each file are processed sequentially inside that worker.
                Defaults to ``False`` (serial execution).
            
        Returns:
            Number of files processed
        """

        self.run_opts = RunOptions(
            trim_obs = trim_obs,
            no_matching = no_matching,
            force_obs_time = force_obs_time
        )

        self.model_adapter.validate_run_options(self.run_opts)

        if clear_output:
            print("Clearing all output folders...")
            output_folders = [
                self.config['parquet_folder'],
                self.config['input_nml_bck'],
                self.config['trimmed_obs_folder'],
                self.config['output_folder'],
            ]
            for folder in output_folders:
                print("  Clearing folder:", folder)
                config_utils.clear_folder(folder)
            tmp_folder = self.config['tmp_folder']
            print("  Clearing folder:", tmp_folder)
            shutil.rmtree(tmp_folder, ignore_errors=True)
            os.makedirs(tmp_folder, exist_ok=True)
            print("  All output folders cleared.")
            print()
        
        if not parquet_only:
            print("Starting files processing.")
            files_processed = self.process_files(
                trim_obs=trim_obs,
                no_matching=no_matching,
                force_obs_time=force_obs_time,
                parallel=parallel,
            )
            print()
        
        # Convert to parquet
        print("Converting obs_seq format to parquet and adding some diagnostics data...")
        self.merge_model_obs_to_parquet(trim_obs)
        print(f"  Parquet data saved to: {self.config['parquet_folder']}")
        print()

        print("Done!")


    def process_files(self, trim_obs: bool = False, no_matching: bool = False,
                     force_obs_time: bool = False, parallel: bool = False) -> int:
        """Process model and observation files.
        
        Args:
            trim_obs: Whether to trim obs_seq.in files to model grid boundaries
            no_matching: Whether to skip time-matching and assume 1:1 correspondence
            force_obs_time: Whether to assign observations reference time to model files
            parallel: Whether to process model files in parallel.  When ``True``,
                one :class:`~concurrent.futures.ThreadPoolExecutor` worker is
                dispatched per model output file.  The number of threads is
                chosen automatically by Python (``min(32, os.cpu_count() + 4)``).
                Defaults to ``False``.
        """

        # Check that perfect_model_obs_dir is set
        if self.config.get('perfect_model_obs_dir') is None:
            raise ValueError("Configuration parameter 'perfect_model_obs_dir' missing: did you specify the path to the perfect_model_obs executable?")

        # Initialize base input.nml
        print("  Initializing input.nml for process_model_obs subprocess...")
        self._initialize_model_namelist()
        
        # Print configuration
        self._print_workflow_config(trim_obs)

        # Validate configuration parameters
        if not hasattr(self, 'run_opts'):
            self.run_opts = RunOptions(
                trim_obs = trim_obs,
                no_matching = no_matching,
                force_obs_time = force_obs_time
            )
        self.model_adapter.validate_paths(self.config, self.run_opts)

        # Get and validate file lists
        model_in_files = file_utils.get_sorted_files(self.config['model_files_folder'], "*.nc")
        obs_in_files = file_utils.get_sorted_files(self.config['obs_seq_in_folder'], "*")

        print(f"  Found {len(model_in_files)} files to process")

        # Get model boundaries if trimming observations
        hull_polygon, hull_points = None, None
        if trim_obs:
            print("  Getting model boundaries...")
            hull_polygon, hull_points = model_tools.get_model_boundaries(self.config['ocean_geometry'])

        # Validate that model files have non-overlapping timestamps (cheap metadata read)
        self._validate_model_file_timestamps(model_in_files)

        try:
            # Process files
            if no_matching:
                if parallel:
                    # Pre-read model times in main thread to avoid xarray thread-safety issues.
                    # xarray's CachingFileManager is not safe for concurrent access.
                    precomputed_times: Dict[str, Optional[Tuple[int, int]]] = {}
                    if not force_obs_time:
                        for model_in_f in model_in_files:
                            precomputed_times[model_in_f] = self.model_adapter.get_model_time_in_days_seconds(
                                model_in_f
                            )

                    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
                        futures = [
                            executor.submit(
                                self._process_model_obs_pair,
                                model_in_file, obs_in_file, trim_obs, counter,
                                hull_polygon, hull_points, force_obs_time,
                                precomputed_times.get(model_in_file),
                            )
                            for counter, (model_in_file, obs_in_file)
                            in enumerate(zip(model_in_files, obs_in_files))
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            future.result()  # re-raise any worker exception
                else:
                    for counter, (model_in_file, obs_in_file) in enumerate(zip(model_in_files, obs_in_files)):
                        self._process_model_obs_pair(
                            model_in_file, obs_in_file, trim_obs, counter,
                            hull_polygon, hull_points, force_obs_time,
                        )
            else:
                if parallel:
                    # Pre-match obs files to model snapshots in the main thread so
                    # each worker receives a disjoint, pre-determined set of pairs.
                    # This prevents multiple workers from independently matching the
                    # same obs file (the bug that occurs when every worker iterates
                    # the full obs list with a fresh used_obs_in_files=[]).
                    prematched = self._precompute_time_matching(model_in_files, obs_in_files)

                    # Base counters are derived from actual matched-pair counts so
                    # output filenames remain contiguous and ordered by model file.
                    cumulative_total = 0
                    base_counters: Dict[str, int] = {}
                    for model_in_f in model_in_files:
                        base_counters[model_in_f] = cumulative_total
                        cumulative_total += len(prematched[model_in_f])

                    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
                        futures = [
                            executor.submit(
                                self._process_model_file_worker,
                                model_in_f, obs_in_files, base_counters[model_in_f],
                                trim_obs, hull_polygon, hull_points, force_obs_time,
                                prematched_pairs=prematched[model_in_f],
                            )
                            for model_in_f in model_in_files
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            future.result()  # re-raise any worker exception
                else:
                    self._process_with_time_matching(
                        model_in_files, obs_in_files, trim_obs,
                        hull_polygon, hull_points, force_obs_time,
                    )
        finally:
            # In the parallel path the symlink lives in each worker's tmpdir
            # (already removed), so this is a harmless no-op.
            self._namelist.cleanup_namelist_symlink()


    def merge_model_obs_to_parquet(self, trim_obs: bool) -> None:
        """Merge model and observation files to parquet format."""
        output_folder = self.config['output_folder']
        parquet_folder = self.config['parquet_folder']
        
        if trim_obs:
            obs_folder = self.config['trimmed_obs_folder']
        else:
            obs_folder = self.config['obs_seq_in_folder']
            print(self.config['obs_seq_in_folder'])

        perf_obs_files = sorted(glob.glob(os.path.join(output_folder, "obs_seq*.out")))
        orig_obs_files = sorted(glob.glob(os.path.join(obs_folder, "*")))

        print(f"  perf_obs_files: {perf_obs_files}")
        print(f"  orig_obs_files: {orig_obs_files}")
        
        tmp_parquet_folder = os.path.join(parquet_folder, "tmp")
        os.makedirs(tmp_parquet_folder, exist_ok=True)

        for perf_obs_f, orig_obs_f in zip(perf_obs_files, orig_obs_files):
            self._merge_pair_to_parquet(perf_obs_f, orig_obs_f, tmp_parquet_folder)

        ddf = dd.read_parquet(tmp_parquet_folder)
        ddf = ddf.repartition(partition_size="300MB")
        name_function = lambda x: f"model-obs-{x}.parquet"
        ddf.to_parquet(
            parquet_folder,
            append=False,
            name_function=name_function
        )

        shutil.rmtree(tmp_parquet_folder)
        self._set_model_obs_df()
        print(f"  Total number of observations in output dataset: {len(self.get_all_model_obs_df())}")
        print(f"  Succesfull interpolations in output dataset   : {len(self.get_good_model_obs_df())}")
        print(f"  Failed interpolations in output dataset       : {len(self.get_failed_model_obs_df())}")

    def _print_workflow_config(self, trim_obs: bool) -> None:
        """Print workflow configuration."""
        print("  Configuration:")
        print(f"    perfect_model_obs_dir: {self.config['perfect_model_obs_dir']}")
        print(f"    input_nml: {self._namelist.namelist_path}")
        print(f"    model_files_folder: {self.config['model_files_folder']}")
        print(f"    obs_seq_in_folder: {self.config['obs_seq_in_folder']}")
        print(f"    output_folder: {self.config['output_folder']}")
        print(f"    input_nml_bck: {self.config.get('input_nml_bck', 'input.nml.backup')}")
        print(f"    tmp_folder: {self.config['tmp_folder']}")
        if trim_obs:
            print(f"    trimmed_obs_folder: {self.config.get('trimmed_obs_folder', 'trimmed_obs_seq')}")
    
    def _initialize_model_namelist(self) -> None:
        """Initialize model namelist parameters."""
        self._namelist = namelist.Namelist(self.input_nml_template)

        self._namelist.update_namelist_param(
            "model_nml", "assimilation_period_days", self.config['time_window']['days'], string=False
        )
        self._namelist.update_namelist_param(
            "model_nml", "assimilation_period_seconds", self.config['time_window']['seconds'], string=False
        )

        print(f'ocean model: {self.model_adapter.ocean_model}')
        common_model_keys = self.model_adapter.get_common_model_keys()
        for key in self.config.keys():
            if key=='debug':
                self._namelist.update_namelist_param(
                    "model_nml", key, self.config[key],string=False
                )
            elif key == 'variables':
                # ROMS 5-field format: 'var', 'QTY', 'NA', 'NA', 'UPDATE'
                self._namelist.update_namelist_param(
                    "model_nml", key, self.config[key], dict_format='quintuplet'
                )
            elif key == 'model_state_variables':
                # MOM6 3-field format: 'var', 'QTY', 'UPDATE' (no clamp placeholders)
                self._namelist.update_namelist_param(
                    "model_nml", key, self.config[key], dict_format='triplet'
                )
            elif key in common_model_keys:
                self._namelist.update_namelist_param(
                    "model_nml", key, self.config[key]
                )

        # Update observation types if specified in config
        if 'use_these_obs' in self.config:
            print("  Processing observation types from config...")
            rst_file_path = os.path.join(
                self.config['perfect_model_obs_dir'],
                '../../../observations/forward_operators/obs_def_ocean_mod.rst'
            )
            try:
                obs_types_tuple = self.model_adapter.parse_dart_obs_type(rst_file_path)
                expanded_obs_types = config_utils.validate_and_expand_obs_types(
                    self.config['use_these_obs'], obs_types_tuple
                )
                print(f"    Target observation types: {expanded_obs_types}")
                if not expanded_obs_types:
                    raise ValueError("Expanded observation types list cannot be empty")
                self._namelist.update_namelist_param('obs_kind_nml', 'assimilate_these_obs_types', expanded_obs_types)
                print("    Updated obs_kind_nml section with observation types")
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not process observation types: {e}")
                print("Continuing with existing obs_kind_nml configuration")

        # Snapshot of the fully configured base content shared across worker threads.
        # Workers copy from this string rather than re-applying all common parameters.
        self._base_nml_content: str = self._namelist.content

    def _validate_model_file_timestamps(self, model_in_files: List[str]) -> None:
        """Raise ValueError if any timestamp appears in more than one model file.

        Opens every model file via the model adapter (metadata-only read) and
        collects all ``time`` coordinate values.  This enforces the assumption
        that model files cover non-overlapping time ranges, which is required
        for correct obs-file assignment in both serial and parallel execution.

        Args:
            model_in_files: Sorted list of model NetCDF file paths.

        Raises:
            ValueError: If a duplicate timestamp is found across the file set.
        """
        seen: Dict[Any, str] = {}  # timestamp -> first file that contained it
        for model_in_f in model_in_files:
            with self.model_adapter.open_dataset_ctx(model_in_f) as ds:
                for t in ds["time"].values:
                    if t in seen:
                        raise ValueError(
                            f"Duplicate timestamp {t} found in both "
                            f"'{seen[t]}' and '{model_in_f}'. "
                            "Model input files must have unique timestamps."
                        )
                    seen[t] = model_in_f

    def _process_model_file_worker(
        self,
        model_in_f: str,
        obs_in_files: List[str],
        base_counter: int,
        trim_obs: bool,
        hull_polygon: Optional[Any],
        hull_points: Optional[np.ndarray],
        force_obs_time: bool,
        used_obs_in_files: Optional[List[str]] = None,
        prematched_pairs: Optional[List[Tuple[int, str]]] = None,
    ) -> int:
        """Process all matched pairs for a single model output file.

        Supports two operating modes:

        **Pre-matched mode** (parallel path): when *prematched_pairs* is given,
        the (snapshot index, obs file) assignments have already been determined
        by :meth:`_precompute_time_matching` in the main thread.  The worker
        iterates these pairs directly; no time-matching is performed and no
        shared state is accessed.

        **Time-matching mode** (serial path): when *prematched_pairs* is
        ``None``, the worker performs the time-window scan itself.  A shared
        *used_obs_in_files* list prevents the same obs file from being matched
        by more than one model file.

        Each matched pair is processed via :meth:`_process_model_obs_pair` with
        a counter of ``base_counter + local_match_index``.

        Args:
            model_in_f: Path to the model output file handled by this worker.
            obs_in_files: Full list of observation input files (read-only;
                only used when *prematched_pairs* is ``None``).
            base_counter: Starting counter value for pairs found in this file.
            trim_obs: Whether to trim observations to model grid boundaries.
            hull_polygon: Model grid boundary polygon (used when *trim_obs*).
            hull_points: Model grid boundary points array (used when *trim_obs*).
            force_obs_time: Whether to use obs time instead of model time.
            used_obs_in_files: Shared list of already-matched obs file paths
                (serial path only).  Ignored when *prematched_pairs* is given.
            prematched_pairs: Pre-assigned ``(snapshot_index, obs_file_path)``
                tuples from :meth:`_precompute_time_matching` (parallel path).
                When provided, *used_obs_in_files* is ignored.

        Returns:
            Number of model-obs pairs processed for this model file.
        """
        print(f"    Processing model file {model_in_f}...")

        with self.model_adapter.open_dataset_ctx(model_in_f) as ds:
            time_var = "time"  # open_dataset_ctx() renames model time varname to 'time'
            snapshots_nb = ds.sizes[time_var]
            print(f"      model has {snapshots_nb} snapshots.")

            if prematched_pairs is not None:
                # ---- Parallel path: pairs pre-assigned; no time-matching needed ----
                for local_match_index, (t_id, obs_in_file) in enumerate(prematched_pairs):
                    print(f"      processing pre-matched snapshot {t_id} "
                          f"with obs_seq {os.path.basename(obs_in_file)}...")
                    counter = base_counter + local_match_index
                    tmp_model_in_file = os.path.join(
                        self.config['tmp_folder'],
                        os.path.basename(model_in_f) + "_tmp_" + str(t_id),
                    )
                    if snapshots_nb > 1:
                        model_time_varname = self.model_adapter.time_varname
                        ncks = [
                            "ncks", "-d", f"{model_time_varname},{t_id}",
                            model_in_f, tmp_model_in_file,
                        ]
                        print(f"        Calling {' '.join(ncks)}")
                        subprocess.run(ncks, check=True)
                    else:
                        tmp_model_in_file = model_in_f

                    self._process_model_obs_pair(
                        tmp_model_in_file, obs_in_file, trim_obs, counter,
                        hull_polygon, hull_points, force_obs_time,
                        original_model_file=model_in_f,
                    )

                    if snapshots_nb > 1:
                        os.remove(tmp_model_in_file)

                return len(prematched_pairs)

            # ---- Serial path: perform time-matching on the fly ----
            if used_obs_in_files is None:
                used_obs_in_files = []

            local_match_index = 0
            for t_id, time in enumerate(ds[time_var].values):
                print(f"      processing snapshot {t_id+1} of {snapshots_nb}...")
                for obs_in_file in obs_in_files:
                    if obs_in_file in used_obs_in_files:
                        continue

                    print(f"        checking obs_seq file {obs_in_file}")
                    obs_in_df = obsq.ObsSequence(obs_in_file)
                    t1 = obs_in_df.df.time.min()
                    t2 = obs_in_df.df.time.max()
                    print(f"        obs_seq min time: {t1}")
                    print(f"        obs_seq max time: {t2}")
                    print(f"        snapshot time: {pd.Timestamp(time)}")

                    tw = timedelta(
                        days=self.config["time_window"]["days"],
                        seconds=self.config["time_window"]["seconds"],
                    )
                    half_tw = tw / 2
                    ts = pd.Timestamp(time)
                    ts1 = ts - half_tw
                    ts2 = ts + half_tw
                    print(
                        f"        Validating obs_seq if obs are with in window {tw} "
                        f"centered on {ts}, i.e. between {ts1} and {ts2}."
                    )

                    if (ts1 <= t1 <= ts2) and (ts1 <= t2 <= ts2):
                        used_obs_in_files.append(obs_in_file)
                        counter = base_counter + local_match_index
                        tmp_model_in_file = os.path.basename(model_in_f) + "_tmp_" + str(t_id)
                        tmp_model_in_file = os.path.join(self.config['tmp_folder'], tmp_model_in_file)

                        if snapshots_nb > 1:
                            model_time_varname = self.model_adapter.time_varname
                            ncks = [
                                "ncks", "-d", f"{model_time_varname},{t_id}",
                                model_in_f, tmp_model_in_file,
                            ]
                            print(f"        Calling {' '.join(ncks)}")
                            subprocess.run(ncks, check=True)
                        else:
                            tmp_model_in_file = model_in_f

                        self._process_model_obs_pair(
                            tmp_model_in_file, obs_in_file, trim_obs, counter,
                            hull_polygon, hull_points, force_obs_time,
                            original_model_file=model_in_f,
                        )

                        if snapshots_nb > 1:
                            os.remove(tmp_model_in_file)

                        local_match_index += 1
                        break

        return local_match_index

    def _precompute_time_matching(
        self,
        model_in_files: List[str],
        obs_in_files: List[str],
    ) -> Dict[str, List[Tuple[int, str]]]:
        """Assign obs files to model snapshots before parallel workers are launched.

        Runs the same serial time-window logic as :meth:`_process_model_file_worker`
        but without processing any pairs.  Each obs file is assigned to at most one
        (model file, snapshot) pair; once assigned it is excluded from subsequent
        model files.

        This must be called in the main thread before spawning workers so that
        every worker receives a disjoint, pre-determined set of (snapshot, obs)
        assignments and no two workers can claim the same obs file.

        Args:
            model_in_files: Ordered list of model output file paths.
            obs_in_files: Full list of obs sequence input file paths.

        Returns:
            A dict mapping each model file path to a list of
            ``(snapshot_index, obs_file_path)`` tuples in match order.
        """
        print("  Pre-matching obs files to model snapshots...")
        used_obs: List[str] = []
        result: Dict[str, List[Tuple[int, str]]] = {f: [] for f in model_in_files}

        for model_in_f in model_in_files:
            with self.model_adapter.open_dataset_ctx(model_in_f) as ds:
                for t_id, time in enumerate(ds["time"].values):
                    tw = timedelta(
                        days=self.config["time_window"]["days"],
                        seconds=self.config["time_window"]["seconds"],
                    )
                    half_tw = tw / 2
                    ts = pd.Timestamp(time)
                    ts1 = ts - half_tw
                    ts2 = ts + half_tw

                    for obs_file in obs_in_files:
                        if obs_file in used_obs:
                            continue
                        obs_df = obsq.ObsSequence(obs_file)
                        t1 = obs_df.df.time.min()
                        t2 = obs_df.df.time.max()
                        if (ts1 <= t1 <= ts2) and (ts1 <= t2 <= ts2):
                            used_obs.append(obs_file)
                            result[model_in_f].append((t_id, obs_file))
                            break

        total = sum(len(v) for v in result.values())
        print(f"  Pre-matching complete: {total} pair(s) assigned across "
              f"{len(model_in_files)} model file(s).")
        return result

    def _process_with_time_matching(self, model_in_files: List[str], obs_in_files: List[str],
                                  trim_obs: bool, hull_polygon: Optional[Any],
                                  hull_points: Optional[np.ndarray],
                                  force_obs_time: bool) -> int:
        """Process files with time-matching logic (serial path).

        Delegates per-file work to :meth:`_process_model_file_worker`, passing a
        shared ``used_obs_in_files`` list so that an obs file matched by one
        model file is not re-matched by a later one.
        """
        used_obs_in_files: List[str] = []
        counter = 0
        for model_in_f in model_in_files:
            n = self._process_model_file_worker(
                model_in_f, obs_in_files, counter,
                trim_obs, hull_polygon, hull_points, force_obs_time,
                used_obs_in_files=used_obs_in_files,
            )
            counter += n
        return counter
    
    def _process_model_obs_pair(self, model_in_file: str, obs_in_file: str,
                               trim_obs: bool, counter: int, hull_polygon: Optional[Any],
                               hull_points: Optional[np.ndarray], force_obs_time: bool,
                               precomputed_model_time: Optional[Tuple[int, int]] = None,
                               original_model_file: Optional[str] = None) -> None:
        """Process a single model-observation file pair.

        This method is self-contained and thread-safe: it creates its own
        temporary working directory and a local
        :class:`~model2obs.utils.namelist.Namelist` instance from the
        pre-configured base content, so it can be called from multiple threads
        simultaneously without shared-state conflicts.

        The per-pair DART log is written to
        ``<output_folder>/perfect_model_obs_<NNNN>.log``.  A human-readable
        pair summary (files used, observation counts, interpolation success
        rate) is written to ``<output_folder>/pair_summary_<NNNN>.log``
        regardless of whether DART exits successfully.

        Args:
            model_in_file: Path to the (possibly sliced) model NetCDF file
                submitted to DART.
            obs_in_file: Path to the obs_seq input file.
            trim_obs: Whether to trim observations to model grid boundaries.
            counter: Zero-based pair index used to generate unique filenames.
            hull_polygon: Model grid boundary polygon (used when *trim_obs*).
            hull_points: Model grid boundary points array (used when *trim_obs*).
            force_obs_time: Whether to use obs time instead of model time.
            precomputed_model_time: Optional pre-read ``(days, seconds)`` tuple
                from the model file.  When provided, the xarray read inside this
                method is skipped, which avoids thread-safety issues with
                xarray's ``CachingFileManager`` during parallel execution.
            original_model_file: Path to the original (un-sliced) model file,
                when *model_in_file* is a temporary ncks slice.  Used only for
                the pair summary log.  Defaults to ``None`` (no separate
                "original" entry is written).
        """
        obs_in_filename = os.path.basename(obs_in_file)
        file_number = f"{counter:04d}"

        obs_in_file_nml = obs_in_file
        if trim_obs:
            print(f"        Trimming obs_seq file {obs_in_filename} to model grid boundaries...")
            trimmed_obs_file = os.path.join(
                self.config['trimmed_obs_folder'],
                f"trimmed_obs_seq_{file_number}.in"
            )
            obs_seq_tools.trim_obs_seq_in(obs_in_file, hull_polygon, hull_points, trimmed_obs_file)
            obs_in_file_nml = trimmed_obs_file

        perfect_output_filename = f"perfect_output_{file_number}.nc"
        perfect_output_path = os.path.join(self.config['output_folder'], perfect_output_filename)

        obs_output_filename = f"obs_seq_{file_number}.out"
        obs_output_path = os.path.join(self.config['output_folder'], obs_output_filename)

        # Collect obs_seq.in statistics for the pair summary log (before DART runs)
        _obs_seq_in = obsq.ObsSequence(obs_in_file_nml)
        _obs_submitted_count = len(_obs_seq_in.df)
        _obs_min_time = _obs_seq_in.df["time"].min()
        _obs_max_time = _obs_seq_in.df["time"].max()
        _obs_original_count: Optional[int] = None
        if obs_in_file != obs_in_file_nml:
            _obs_original_count = len(obsq.ObsSequence(obs_in_file).df)

        print(f"        Processing file #{counter + 1}:")
        print(f"          Model input file: {model_in_file}")
        print(f"          Obs input file: {obs_in_file_nml}")
        print(f"          Perfect output file: {perfect_output_filename}")
        print(f"          Obs output file: {obs_output_filename}")

        worker_tmpdir = tempfile.mkdtemp(dir=self.config['tmp_folder'])
        try:
            # Create a thread-local namelist from the pre-configured base content
            local_nml = namelist.Namelist.from_content(
                self._base_nml_content, working_dir=worker_tmpdir
            )

            # Apply pair-specific namelist parameters
            local_nml.update_namelist_param(
                "perfect_model_obs_nml", "input_state_files", model_in_file
            )
            local_nml.update_namelist_param(
                "perfect_model_obs_nml", "output_state_files", perfect_output_path
            )
            local_nml.update_namelist_param(
                "perfect_model_obs_nml", "obs_seq_in_file_name", obs_in_file_nml
            )
            local_nml.update_namelist_param(
                "perfect_model_obs_nml", "obs_seq_out_file_name", obs_output_path
            )

            if not force_obs_time:
                if precomputed_model_time is not None:
                    model_time_days, model_time_seconds = precomputed_model_time
                else:
                    print("          Retrieving model time from model input file and updating namelist...")
                    model_time_days, model_time_seconds = self.model_adapter.get_model_time_in_days_seconds(
                        model_in_file
                    )
                local_nml.update_namelist_param(
                    "perfect_model_obs_nml", "init_time_days", model_time_days, string=False
                )
                local_nml.update_namelist_param(
                    "perfect_model_obs_nml", "init_time_seconds", model_time_seconds, string=False
                )
                _log_time_days, _log_time_seconds, _log_time_source = (
                    model_time_days, model_time_seconds, "model file"
                )
            else:
                print("          Retrieving obs time from obs_seq and updating namelist...")
                obs_time_days, obs_time_seconds = file_utils.get_obs_time_in_days_seconds(obs_in_file)
                local_nml.update_namelist_param(
                    "perfect_model_obs_nml", "init_time_days", obs_time_days, string=False
                )
                local_nml.update_namelist_param(
                    "perfect_model_obs_nml", "init_time_seconds", obs_time_seconds, string=False
                )
                _log_time_days, _log_time_seconds, _log_time_source = (
                    obs_time_days, obs_time_seconds, "obs midpoint"
                )

            # Write the pair-specific namelist to the persistent backup folder
            input_nml_bck_path = os.path.join(
                self.config['input_nml_bck'],
                f"input.nml_{file_number}.backup"
            )
            local_nml.write_namelist(input_nml_bck_path)
            # Symlink lives inside worker_tmpdir so the subprocess finds it
            local_nml.symlink_to_namelist(input_nml_bck_path)
            print(f"          {input_nml_bck_path} created.")
            print()

            # Call perfect_model_obs from the isolated worker directory
            print("          Calling perfect_model_obs...")
            perfect_model_obs_exe = os.path.join(
                self.config['perfect_model_obs_dir'], "perfect_model_obs"
            )
            log_file_path = os.path.join(
                self.config['output_folder'], f"perfect_model_obs_{file_number}.log"
            )
            with open(log_file_path, "w") as logfile:
                process = subprocess.Popen(
                    [perfect_model_obs_exe],
                    cwd=worker_tmpdir,
                    stdout=logfile,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                process.wait()

            _dart_exit_code = process.returncode

            # Collect interpolation counts from obs_seq.out for the pair summary log
            _n_success: Optional[int] = None
            _n_fail: Optional[int] = None
            if os.path.exists(obs_output_path):
                try:
                    _perf_out = obsq.ObsSequence(obs_output_path)
                    _qc_cols = [c for c in _perf_out.df.columns if c.endswith("_QC")]
                    if _qc_cols:
                        _qc_vals = _perf_out.df[_qc_cols[0]]
                        _n_success = int((_qc_vals <= 2).sum())
                        _n_fail = int((_qc_vals > 2).sum())
                except Exception:  # pylint: disable=broad-except
                    pass

            self._write_pair_summary_log(
                file_number=file_number,
                model_in_file=model_in_file,
                original_model_file=original_model_file,
                log_time_days=_log_time_days,
                log_time_seconds=_log_time_seconds,
                time_source=_log_time_source,
                obs_in_file_nml=obs_in_file_nml,
                obs_in_file_orig=obs_in_file,
                obs_min_time=_obs_min_time,
                obs_max_time=_obs_max_time,
                obs_submitted_count=_obs_submitted_count,
                obs_original_count=_obs_original_count,
                dart_exit_code=_dart_exit_code,
                obs_output_path=obs_output_path,
                n_success=_n_success,
                n_fail=_n_fail,
            )

            if _dart_exit_code != 0:
                raise RuntimeError(
                    f"perfect_model_obs failed for file #{counter + 1} "
                    f"(exit code {_dart_exit_code}). "
                    f"See log: {log_file_path}"
                )

            print(f"          Perfect model output saved to: {perfect_output_path}")
            print(f"          obs_seq.out output saved to: {obs_output_path}")
            print(f"          perfect_model_obs log saved to: {log_file_path}")
        finally:
            shutil.rmtree(worker_tmpdir, ignore_errors=True)

    def _write_pair_summary_log(
        self,
        file_number: str,
        model_in_file: str,
        original_model_file: Optional[str],
        log_time_days: int,
        log_time_seconds: int,
        time_source: str,
        obs_in_file_nml: str,
        obs_in_file_orig: str,
        obs_min_time: pd.Timestamp,
        obs_max_time: pd.Timestamp,
        obs_submitted_count: int,
        obs_original_count: Optional[int],
        dart_exit_code: int,
        obs_output_path: str,
        n_success: Optional[int],
        n_fail: Optional[int],
    ) -> None:
        """Write a plain-text key-value summary log for one model-observation pair.

        The log is written to ``<output_folder>/pair_summary_<file_number>.log``.
        If writing fails for any reason (e.g. disk full, permission error), a
        warning is printed and the workflow continues unaffected.

        The time window used to match this obs_seq file to the model snapshot is
        derived from ``self.config["time_window"]`` and the model snapshot time.

        Args:
            file_number: Zero-padded 4-digit pair counter string, e.g. ``'0003'``.
            model_in_file: Path to the model file submitted to DART (may be a
                temporary ncks slice).
            original_model_file: Path to the original (un-sliced) model file, or
                ``None`` when *model_in_file* is already the original.
            log_time_days: DART init_time days component (relative to 1601-01-01).
            log_time_seconds: DART init_time seconds component.
            time_source: Human-readable label for the time origin, e.g.
                ``'model file'`` or ``'obs midpoint'``.
            obs_in_file_nml: Path to the obs_seq file passed to DART (may be a
                trimmed copy).
            obs_in_file_orig: Path to the original obs_seq file before trimming.
            obs_min_time: Earliest observation time in *obs_in_file_nml*.
            obs_max_time: Latest observation time in *obs_in_file_nml*.
            obs_submitted_count: Number of observations submitted to DART.
            obs_original_count: Number of observations in *obs_in_file_orig*
                before trimming, or ``None`` if trimming was not applied.
            dart_exit_code: Return code from ``perfect_model_obs``.
            obs_output_path: Path to the ``obs_seq_NNNN.out`` file produced by DART.
            n_success: Number of successfully interpolated observations (QC ≤ 2),
                or ``None`` if the output could not be read.
            n_fail: Number of failed interpolations (QC > 2), or ``None`` if the
                output could not be read.
        """
        _DART_EPOCH = datetime(1601, 1, 1)
        _W = 32  # label column width for alignment

        def _line(label: str, value: str) -> str:
            return f"{label:<{_W}}: {value}"

        try:
            time_used = _DART_EPOCH + timedelta(days=log_time_days, seconds=log_time_seconds)
            time_used_str = time_used.isoformat(sep="T", timespec="seconds")

            tw = timedelta(
                days=self.config["time_window"]["days"],
                seconds=self.config["time_window"]["seconds"],
            )
            half_tw = tw / 2
            ts = pd.Timestamp(time_used)
            window_start = (ts - half_tw).isoformat(sep="T", timespec="seconds")
            window_end   = (ts + half_tw).isoformat(sep="T", timespec="seconds")

            lines = [
                "=== Model-Observation Pair Summary ===",
                _line("Pair number", file_number),
                "---------- Model input ----------",
                _line("NC file submitted to DART", model_in_file),
            ]
            if original_model_file and original_model_file != model_in_file:
                lines.append(_line("Original NC file", original_model_file))
            lines.append(_line("Time used", f"{time_used_str}  (from {time_source})"))
            lines.append(_line("Time window", str(tw)))
            lines.append("---------- Window for observations ----------")
            lines.append(_line("Window start", window_start))
            lines.append(_line("Window end",   window_end))

            lines.append("---------- Observation input ----------")
            lines.append(_line("obs_seq.in file", obs_in_file_nml))
            if obs_in_file_orig != obs_in_file_nml:
                lines.append(_line("Original obs_seq.in", obs_in_file_orig))
            lines.append(_line("obs_seq.in time min",
                                pd.Timestamp(obs_min_time).isoformat(sep="T", timespec="seconds")))
            lines.append(_line("obs_seq.in time max",
                                pd.Timestamp(obs_max_time).isoformat(sep="T", timespec="seconds")))
            submitted_str = str(obs_submitted_count)
            if obs_original_count is not None:
                submitted_str += f"  ({obs_original_count} in original obs_seq.in before trimming)"
            lines.append(_line("Obs submitted to DART", submitted_str))

            lines.append("---------- DART output ----------")
            lines.append(_line("obs_seq.out file", obs_output_path))
            dart_status = "success" if dart_exit_code == 0 else "FAILED"
            lines.append(_line("DART exit code", f"{dart_exit_code}  ({dart_status})"))
            if n_success is not None and n_fail is not None:
                total = n_success + n_fail
                pct_ok = 100.0 * n_success / total if total > 0 else 0.0
                pct_fail = 100.0 * n_fail / total if total > 0 else 0.0
                lines.append(_line("Successfully interpolated obs",
                                   f"{n_success} / {total}  ({pct_ok:.1f}%)"))
                lines.append(_line("Failed interpolations",
                                   f"{n_fail} / {total}  ({pct_fail:.1f}%)"))
            else:
                lines.append(_line("Interpolation counts",
                                   "obs_seq.out not found or QC column absent"))

            log_path = os.path.join(self.config["output_folder"],
                                    f"pair_summary_{file_number}.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            print(f"          Pair summary log saved to: {log_path}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"          WARNING: could not write pair summary log: {exc}")

    def _merge_pair_to_parquet(self, perf_obs_file: str, orig_obs_file: str, 
                              parquet_path: str) -> None:
        """Merge a pair of observation files into parquet format."""
        # Read obs_sequence files
        perf_obs_out = obsq.ObsSequence(perf_obs_file)
        perf_obs_out.update_attributes_from_df()
        trimmed = obsq.ObsSequence(orig_obs_file)
        trimmed.update_attributes_from_df()

        obs_col = [col for col in trimmed.df.columns.to_list() if col.endswith("_observation") or col=="observation"]
        if len(obs_col) > 1:
            raise ValueError("More than one observation columns found.")
        else:
            trimmed.df = trimmed.df.rename(columns={obs_col[0]:"obs"})

        qc_col = [col for col in perf_obs_out.df.columns.to_list() if col.endswith("_QC")]
        if len(qc_col) > 1:
            raise ValueError("More than one QC column found.")
        perf_model_col = 'interpolated_model'
        perf_model_col_QC = perf_model_col + "_QC"
        perf_obs_out.df = perf_obs_out.df.rename(
            columns={
                "truth":perf_model_col,
                qc_col[0]:perf_model_col_QC
            })

        def compute_hash(df: pd.DataFrame, cols: List[str], hash_col: str = "hash") -> pd.DataFrame:
            """Generate unique hash for merging"""
            concat = df[cols].astype(str).agg('-'.join, axis=1)
            df[hash_col] = pd.util.hash_pandas_object(concat, index=False).astype('int64')
            return df

        trimmed.df = compute_hash(trimmed.df, ['obs_num', 'seconds', 'days'])
        perf_obs_out.df = compute_hash(perf_obs_out.df, ['obs_num', 'seconds', 'days'])

        # Merge DataFrames
        merge_key = "hash"
        trimmed.df = trimmed.df.set_index(merge_key, drop=True)
        perf_obs_out.df = perf_obs_out.df.set_index(merge_key, drop=True)
        ref_cols = ['longitude', 'latitude', 'time', 'vertical', 'type', 'obs_err_var']
        merged = pd.merge(
            trimmed.df[ref_cols + ['obs']],
            perf_obs_out.df[ref_cols + [perf_model_col, perf_model_col_QC]],
            left_index=True,
            right_index=True,
            how='outer',
            suffixes=('_trim', '_perf')
        )

        # Check that reference columns are identical and deduplicate them
        for col in ref_cols:
            c_trim, c_perf = f"{col}_trim", f"{col}_perf"
            if c_trim in merged and c_perf in merged:
                if merged[c_trim].equals(merged[c_perf]) or np.all(np.isclose(merged[c_trim], merged[c_perf], atol=1e-13)):
                    merged = merged.drop(columns=[c_trim])
                    merged = merged.rename(columns={c_perf: col})
                else:
                    raise ValueError(f"{col}: {c_trim} and {c_perf} not identical. The two files probably do not refer to the same observation space.")
            else:
                raise ValueError(f"{col}: one of {c_trim}, {c_perf} not present in merged DataFrame. The two files probably do not refer to the same observation space.")

        # Sort dataframe by time -> position -> depth
        sort_order = ['time', 'longitude', 'latitude', 'vertical']
        merged = merged.sort_values(by=sort_order)

        # DART's pmo uses different units for MOM6 and ROMS_Rutgers
        merged = self.model_adapter.convert_units(merged)

        # Add diagnostic columns
        merged['difference'] = merged['obs'] - merged[perf_model_col]
        merged['abs_difference'] = np.abs(merged['difference'])
        merged['squared_difference'] = merged['difference'] ** 2
        with warnings.catch_warnings(): # catch warnings for sqrt and log on invalid numbers (e.g. NaNs or so for failed interpolations)
            warnings.simplefilter("ignore", RuntimeWarning)
            merged['normalized_difference'] = merged['difference'] / np.sqrt(merged['obs_err_var'])
            merged['log_likelihood'] = -0.5 * (
                merged['difference'] ** 2 / merged['obs_err_var'] +
                np.log(2 * np.pi * merged['obs_err_var'])
            )

        # Reorder columns
        column_order = [
            'time', 'longitude', 'latitude', 'vertical', 'type',
            perf_model_col, 'obs', 'obs_err_var',
        ]
        remaining_cols = [col for col in merged.columns if col not in column_order]
        merged = merged[column_order + remaining_cols]

        ddf = dd.from_pandas(merged)
        name_function = lambda x: f"tmp-model-obs-{x}.parquet"
        append = True
        if not os.listdir(parquet_path):
            append = False  # create new dataset if it's the first in the folder

        ddf.to_parquet(
            parquet_path,
            append=append,
            name_function=name_function,
            write_metadata_file=True,
            ignore_divisions=True,
            write_index=False
        )

    def _set_model_obs_df(self, path: Optional[str] = None) -> None:
        """Create model_obs_df dask dataframe linked to parquet with model-obs data"""
        if path is None:
            path = self.config['parquet_folder']
        if self.model_obs_df is not None:
            print("WARNING: model_obs_df not None, replacing it.")
        self.model_obs_df = dd.read_parquet(path)

    def _get_model_obs_df(self, filters: Optional[str] = None, compute: Optional[bool] = False, path: Optional[str] = None) -> Union[pd.DataFrame, dd.DataFrame]:
        """Get model_obs_df dataframe, computed or not, takes 'all','good','failed' filters and path"""
        if self.model_obs_df is None:
            self._set_model_obs_df(path=path)

        admissible_filters = ['all','good','failed']
        if filters is None or filters == "all":
            ddf = self.model_obs_df
        elif filters == "good":
            ddf = self.model_obs_df[
                self.model_obs_df['interpolated_model_QC']<=2
            ]
        elif filters == "failed":
            ddf = self.model_obs_df[
                self.model_obs_df['interpolated_model_QC']>2
            ]
        else:
            raise ValueError(f"filters value {filters} not supported, use one of {admissible_filters}.")

        if compute:
            return ddf.compute()
        else:
            return ddf

    def get_all_model_obs_df(self, compute: Optional[bool] = False, path: Optional[str] = None) -> Union[pd.DataFrame, dd.DataFrame]:
        """Get all rows in model_obs_df"""
        return self._get_model_obs_df(compute=compute,path=path)

    def get_good_model_obs_df(self, compute: Optional[bool] = False, path: Optional[str] = None) -> Union[pd.DataFrame, dd.DataFrame]:
        """Get only rows in model_obs_df corresponding to successful interpolations"""
        return self._get_model_obs_df(filters='good', compute=compute, path=path)

    def get_failed_model_obs_df(self, compute: Optional[bool] = False, path: Optional[str] = None) -> Union[pd.DataFrame, dd.DataFrame]:
        """Get only rows in model_obs_df corresponding to failed interpolations"""
        return self._get_model_obs_df(filters='failed', compute=compute, path=path)
