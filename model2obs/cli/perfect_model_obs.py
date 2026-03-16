"""Command-line interface for perfect model observation workflows."""

import argparse
import sys

from ..workflows.workflow_model_obs import WorkflowModelObs


def main() -> None:
    """Main CLI entry point for perfect model observation processing."""
    
    parser = argparse.ArgumentParser(
        description='Script to call perfect_model_obs on multiple model and obs_seq.in files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        help="Path to configuration file (default: config.yaml)",
        required=False,
        default='./config.yaml'
    )

    parser.add_argument(
        '-t', '--trim',
        action='store_true',
        help="Trim obs_seq.in files to model grid boundaries (default: False)",
        required=False,
        default=False
    )

    parser.add_argument(
        '--no-matching',
        action='store_true',
        help="If the obs and model files match 1:1 when alphabetically sorted, skip pair-building through time-matching (faster; default: False)",
        required=False,
        default=False
    )

    parser.add_argument(
        '--force-obs-time',
        action='store_true',
        help="Assign observations reference time to model file in model-obs files pair. Generally discouraged, but relevant when the real model time is not significant (default: False)",
        required=False,
        default=False
    )

    parser.add_argument(
        '--parquet-only',
        action='store_true',
        help="Skip building perfect obs and directly convert existing ones to parquet (default: False)",
        required=False,
        default=False
    )

    parser.add_argument(
        '--clear-output',
        action='store_true',
        help="Clear all output folders defined in config file before processing files (default: False)",
        required=False,
        default=False
    )

    args = parser.parse_args()
    if args.parquet_only and args.trim:
        print("Warning: -t/--trim has no effect when --parquet-only is used.")

    config_file = args.config
    trim_obs = args.trim
    no_matching = args.no_matching
    force_obs_time = args.force_obs_time
    clear_output = args.clear_output

    print(f"Reading configuration from: {config_file}")

    # Create workflow instance
    workflow = WorkflowModelObs.from_config_file(config_file)

    # Validate that perfect_model_obs_dir is specified
    if workflow.get_config('perfect_model_obs_dir') is None:
        raise ValueError("perfect_model_obs_dir must be specified in the config file")

    # Run the workflow
    files_processed = workflow.run(
        trim_obs=trim_obs,
        no_matching=no_matching,
        force_obs_time=force_obs_time,
        parquet_only=args.parquet_only,
        clear_output=clear_output
    )

    if not args.parquet_only:
        print(f"Total files processed: {files_processed}")
        print("Backup saved as: input.nml.backup")

    print("Script executed successfully.")


if __name__ == "__main__":
    main()
