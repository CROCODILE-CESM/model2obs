# Feature Report: Parallel Processing for `model2obs` Workflow

## Summary

Adds optional file-level parallelism to the `WorkflowModelObs` pipeline via a new
`parallel` parameter on `run()` and `process_files()`.  When enabled, each
model–observation pair (or each model file in time-matching mode) is processed
concurrently using `concurrent.futures.ThreadPoolExecutor`, with thread safety
guaranteed by making every worker fully self-contained.

---

## Files Modified

| File | Change type | Description |
|------|-------------|-------------|
| `model2obs/utils/namelist.py` | Modified | Added `working_dir` param to `__init__`; added `from_content()` classmethod |
| `model2obs/workflows/workflow_model_obs.py` | Modified | Core parallelism implementation (see below) |
| `tests/unit/test_workflow_model_obs_methods.py` | Modified | Updated and extended unit tests |
| `tests/unit/test_config_namelist_edge_cases.py` | Bugfix | Fixed `test_symlink_to_namelist_same_source_dest` |
| `tests/integration/test_workflow_model_obs_integration.py` | Modified | Fixed error message match; added parallel integration tests |

---

## Design Decisions

### Thread Safety Strategy

The key challenge is that `WorkflowModelObs` holds shared mutable state
(`self._namelist`, working directory, subprocess log files).  The chosen
approach is to make `_process_model_obs_pair` **fully self-contained**:

1. Each worker creates its own `tempfile.mkdtemp(dir=config['tmp_folder'])`.
2. A thread-local `Namelist` is instantiated via `Namelist.from_content(
   self._base_nml_content, working_dir=worker_tmpdir)`, copying the
   pre-configured base namelist state without touching shared objects.
3. The `perfect_model_obs` subprocess runs with `cwd=worker_tmpdir`, so its
   output files land in the worker's private directory.
4. The worker's `finally` block removes `worker_tmpdir`.

This avoids locks entirely for the hot path (subprocess execution).

### `_base_nml_content` Snapshot

`_initialize_model_namelist()` now saves `self._base_nml_content` — the fully
configured namelist content (time window, variables, model keys, obs types) —
at the end of initialisation.  Workers copy from this snapshot, which is
**read-only after initialisation**, so no synchronisation is needed.

### Pre-reading Model Times (xarray thread safety)

`xr.open_dataset` (backed by netCDF4) is not safe for concurrent access from
multiple threads because xarray's `CachingFileManager` uses a non-thread-safe
LRU cache.  To avoid this, the parallel `no_matching` path pre-reads all model
timestamps in the **main thread** before dispatching workers:

```python
precomputed_times[model_in_f] = file_utils.get_model_time_in_days_seconds(...)
```

Workers receive the pre-read `(days, seconds)` tuple via the new optional
`precomputed_model_time` parameter of `_process_model_obs_pair` and skip the
xarray read entirely.  The serial path is unchanged.

### Counter Pre-assignment (time-matching mode)

In time-matching mode each model file can match multiple obs files, so the
output counter must be globally consistent regardless of which worker finishes
first.  A serial pre-scan opens each model file, counts `ds.sizes['time']`, and
assigns `base_counters[model_in_f] = cumulative_total`.  Workers then use
`base_counter + local_match_index` for their output filenames.

### Per-pair Log Files

Instead of appending to a shared `perfect_model_obs.log` in the working
directory, each pair writes its own log to
`<output_folder>/perfect_model_obs_<NNNN>.log`.  This applies to both serial
and parallel execution.

### Timestamp Validation

`_validate_model_file_timestamps()` is called unconditionally (serial and
parallel) before processing begins.  It opens every model file (single-threaded
metadata read) and raises `ValueError` if any timestamp appears in more than one
file.  This is cheap and catches a class of mis-configured inputs early.

---

## Public API Changes

```python
# WorkflowModelObs.run()
workflow.run(trim_obs=True, no_matching=False, parallel=False)

# WorkflowModelObs.process_files()
workflow.process_files(trim_obs=False, no_matching=False, parallel=False)
```

`parallel=False` is the default, preserving full backward compatibility.

---

## Known Limitations

- **GIL**: Python's GIL means true CPU parallelism is not achieved.  The speedup
  comes from overlapping I/O and subprocess wait time (the dominant cost for
  `perfect_model_obs` runs).
- **`force_obs_time=True` path**: `get_obs_time_in_days_seconds` (pydartdiags)
  is still called per worker.  It reads text files, not NetCDF, so no xarray
  thread-safety concern applies, but it has not been profiled under high
  concurrency.
- **`max_workers=None`**: defaults to the number of CPUs.  No user-facing knob
  is exposed yet.  This can be added later without breaking the API.

---

## Test Coverage Added

- `TestValidateModelFileTimestamps` — unit tests for duplicate timestamp detection
- `TestProcessModelFileWorker` — unit tests for the time-matching worker
- `TestParallelDispatch` — unit tests for ThreadPoolExecutor dispatch logic
- `TestWorkflowModelObsParallelProcessing` — integration tests:
  - Same call count as serial path
  - Worker exceptions propagate to main thread
  - Per-pair log files are created
