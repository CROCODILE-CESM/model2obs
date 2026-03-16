"""Model-observation comparison workflow for CrocoCamp."""

from datetime import timedelta
import glob
from importlib.resources import files
import os
import shutil
import subprocess
from typing import Any, List, Optional, Dict, Union

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
    # default run options for largest compatibility
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
        self.input_nml_template = files('crococamp.utils').joinpath('input_template.nml')
        self.model_obs_df = None
        self.perfect_model_obs_log_file = "perfect_model_obs.log"
        if os.path.isfile(self.perfect_model_obs_log_file):
            os.remove(self.perfect_model_obs_log_file)

    def run(self, trim_obs: bool = True, no_matching: bool = False,
            force_obs_time: bool = False, parquet_only: bool = False,
            clear_output: bool = False) -> int:
        """Execute the complete model-observation workflow.
        
        Args:
            trim_obs: Whether to trim obs_seq.in files to model grid boundaries
            no_matching: Whether to skip time-matching and assume 1:1 correspondence
            force_obs_time: Whether to assign observations reference time to model files
            parquet_only: Whether to skip building perfect obs and directly convert to parquet
            clear_output: Whether to clear output folder before running the workflow (default: False)
            
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
                self.config['output_folder']
            ]
            for folder in output_folders:
                print("  Clearing folder:", folder)
                config_utils.clear_folder(folder)
            print("  All output folders cleared.")
            print()
        
        if not parquet_only:
            print("Starting files processing.")
            files_processed = self.process_files(
                trim_obs=trim_obs,
                no_matching=no_matching,
                force_obs_time=force_obs_time
            )
            print()
        
        # Convert to parquet
        print("Converting obs_seq format to parquet and adding some diagnostics data...")
        self.merge_model_obs_to_parquet(trim_obs)
        print(f"  Parquet data saved to: {self.config['parquet_folder']}")
        print()

        print("Done!")


    def process_files(self, trim_obs: bool = False, no_matching: bool = False, 
                     force_obs_time: bool = False) -> int:
        """Process model and observation files.
        
        Args:
            trim_obs: Whether to trim obs_seq.in files to model grid boundaries
            no_matching: Whether to skip time-matching and assume 1:1 correspondence
            force_obs_time: Whether to assign observations reference time to model files
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

        # Process files
        if no_matching:
            for counter, (model_in_file, obs_in_file) in enumerate(zip(model_in_files, obs_in_files)):
                self._process_model_obs_pair(
                    model_in_file, obs_in_file, trim_obs, counter,
                    hull_polygon, hull_points, force_obs_time
                )
        else:
            counter = self._process_with_time_matching(
                model_in_files, obs_in_files, trim_obs,
                hull_polygon, hull_points, force_obs_time
            )

        # Cleanup
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
            elif key=='variables':
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
                expanded_obs_types = config_utils.validate_and_expand_obs_types(
                    self.config['use_these_obs'], rst_file_path
                )
                print(f"    Target observation types: {expanded_obs_types}")
                if not expanded_obs_types:
                    raise ValueError("Expanded observation types list cannot be empty")
                self._namelist.update_namelist_param('obs_kind_nml', 'assimilate_these_obs_types', expanded_obs_types)
                print("    Updated obs_kind_nml section with observation types")
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not process observation types: {e}")
                print("Continuing with existing obs_kind_nml configuration")

    def _process_with_time_matching(self, model_in_files: List[str], obs_in_files: List[str],
                                  trim_obs: bool, hull_polygon: Optional[Any], 
                                  hull_points: Optional[np.ndarray],
                                  force_obs_time: bool) -> int:
        """Process files with time-matching logic."""
        counter = 0
        used_obs_in_files = []
        
        for model_in_f in model_in_files:
            print(f"    Processing model file {model_in_f}...")

            with self.model_adapter.open_dataset_ctx(model_in_f) as ds:
                time_var =  "time" # open_dataset_ctx() renames model time varname to 'time'
                snapshots_nb = ds.sizes[time_var]
                print(f"      model has {snapshots_nb} snapshots.")
                
                for t_id, time in enumerate(ds[time_var].values):
                    print(f"      processing snapshot {t_id+1} of {snapshots_nb}...")
                    for obs_in_file in obs_in_files:
                        # Skip obs_seq.in files already used
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
                        half_tw = tw/2
                        ts = pd.Timestamp(time)
                        ts1 = ts-half_tw
                        ts2 = ts+half_tw
                        print(f"        Validating obs_seq if obs are with in window {tw} centered on {ts}, i.e. between {ts1} and {ts2}.")
                        
                        if (ts1 <= t1 <= ts2) and (ts1 <= t2 <= ts2):
#                        if t1 <= pd.Timestamp(time) <= t2:
                            used_obs_in_files.append(obs_in_file)
                            tmp_model_in_file = os.path.basename(model_in_f) + "_tmp_" + str(t_id)
                            tmp_model_in_file = os.path.join(self.config['tmp_folder'], tmp_model_in_file)

                            if snapshots_nb > 1:
                                # Slice out the snapshot into a temporary file
                                model_time_varname = self.model_adapter.time_varname
                                ncks = [
                                    "ncks", "-d", f"{model_time_varname},{t_id}",
                                    model_in_f, tmp_model_in_file
                                ]
                                print(f"        Calling {' '.join(ncks)}")
                                subprocess.run(ncks, check=True)
                            else:
                                # If only one snapshot, just use the file as is
                                tmp_model_in_file = model_in_f

                            # Call perfect_model_obs on the pair
                            self._process_model_obs_pair(
                                tmp_model_in_file, obs_in_file, trim_obs, counter,
                                hull_polygon, hull_points, force_obs_time
                            )

                            # Remove temporary file if it was created
                            if snapshots_nb > 1:
                                os.remove(tmp_model_in_file)

                            counter += 1
                            break
        
        return counter
    
    def _process_model_obs_pair(self, model_in_file: str, obs_in_file: str, 
                               trim_obs: bool, counter: int, hull_polygon: Optional[Any],
                               hull_points: Optional[np.ndarray], force_obs_time: bool) -> None:

        """Process a single model-observation file pair."""
        model_in_filename = os.path.basename(model_in_file)
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

        print(f"        Processing file #{counter + 1}:")
        print(f"          Model input file: {model_in_file}")
        print(f"          Obs input file: {obs_in_file_nml}")
        print(f"          Perfect output file: {perfect_output_filename}")
        print(f"          Obs output file: {obs_output_filename}")

        # Update namelist parameters
        self._namelist.update_namelist_param(
            "perfect_model_obs_nml", "input_state_files", model_in_file
        )
        self._namelist.update_namelist_param(
            "perfect_model_obs_nml", "output_state_files", perfect_output_path
        )
        self._namelist.update_namelist_param(
            "perfect_model_obs_nml", "obs_seq_in_file_name", obs_in_file_nml
        )
        self._namelist.update_namelist_param(
            "perfect_model_obs_nml", "obs_seq_out_file_name", obs_output_path
        )

        if not force_obs_time:
            # Assign time to model file
            print("          Retrieving model time from model input file and updating namelist...")
            model_time_days, model_time_seconds = file_utils.get_model_time_in_days_seconds(model_in_file,self.model_adapter.time_varname)
            self._namelist.update_namelist_param(
                "perfect_model_obs_nml", "init_time_days", model_time_days,
                string=False
            )
            self._namelist.update_namelist_param(
                "perfect_model_obs_nml", "init_time_seconds", model_time_seconds,
                string=False
            )
        else:
            # Assign time to model file
            print("          Retrieving obs time from obs_seq and updating namelist...")
            obs_time_days, obs_time_seconds = file_utils.get_obs_time_in_days_seconds(obs_in_file)
            self._namelist.update_namelist_param(
                "perfect_model_obs_nml", "init_time_days", obs_time_days,
                string=False
            )
            self._namelist.update_namelist_param(
                "perfect_model_obs_nml", "init_time_seconds", obs_time_seconds,
                string=False
            )

        # Write updated namelist
        input_nml_bck_path = os.path.join(
            self.config['input_nml_bck'], 
            f"input.nml_{file_number}.backup"
        )
        self._namelist.write_namelist(input_nml_bck_path)
        self._namelist.symlink_to_namelist(input_nml_bck_path)
        print(f"          {input_nml_bck_path} created.")
        print()

        # Call perfect_model_obs
        print("          Calling perfect_model_obs...")
        perfect_model_obs = os.path.join(self.config['perfect_model_obs_dir'], "perfect_model_obs")
        perfect_model_obs_log_file = "perfect_model_obs.log"
        with open(perfect_model_obs_log_file, "a") as logfile: # redirect pmo output to logfile
            process = subprocess.Popen(
                [perfect_model_obs],
                cwd=os.getcwd(),
                stdout=logfile,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True
            )
            process.wait()

        # Wait for the process to finish
        process.wait()

        # Check the return code for errors
        if process.returncode != 0:
            raise RuntimeError(f"Error: {process.stderr.read()}")

        print(f"          Perfect model output saved to: {perfect_output_path}")
        print(f"          obs_seq.out output saved to: {obs_output_path}")
        print(f"          perfect_model_obs log saved to: {perfect_model_obs_log_file}")
    
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
