# Plan: Parallel Support for model2obs Core Functionality

## Scope

Implement **file-level parallelism** (Solution 2) in `WorkflowModelObs`.
One worker is dispatched per model output file.  Within a worker, pairs are
processed sequentially.  Parallelism is controlled by a single
`parallel: bool` flag on `run()` and `process_files()`.  Both the
`no_matching` and time-matching paths are parallelized.

Parallelism backend: `concurrent.futures.ThreadPoolExecutor` (standard library).
The number of threads defaults to `None`, which lets Python choose
`min(32, os.cpu_count() + 4)`.

---

## Critical Shared-State Hazards

| Resource | Current behaviour | Risk in parallel | Mitigation |
|---|---|---|---|
| `self._namelist` | Single mutable `Namelist` instance, mutated per pair | Race condition on reads/writes | `_process_model_obs_pair` creates its own local `Namelist` from a pre-computed base content string; `self._namelist` is no longer mutated per pair |
| `input.nml` symlink in cwd | Single symlink, overwritten by each pair | Two workers point it at the wrong file | Each worker runs `perfect_model_obs` from its own temporary subdirectory with its own `input.nml` symlink |
| `perfect_model_obs.log` | Single append-mode log | Interleaved output in parallel | Per-worker log named `perfect_model_obs_{counter:04d}.log` in `output_folder`; the shared log is dropped |
| Output file counters | Assigned sequentially inside loop | Non-deterministic naming → wrong order in `merge_model_obs_to_parquet` | Counters pre-assigned in a serial scan before any workers are dispatched (see below) |
| `used_obs_in_files` list | Mutated inside `_process_with_time_matching` | Shared mutation between workers | Each worker operates on its own slice of obs files; no cross-worker sharing needed (see counter design) |

---

## Counter Pre-Assignment Strategy

`merge_model_obs_to_parquet` zips sorted output files with sorted original obs
files. To preserve correct ordering, counters must be assigned such that
sorting them reproduces the input file order.

**Serial pre-scan** (cheap metadata-only reads):
```
for each model_in_f in model_in_files:
    open_dataset_ctx(model_in_f) → read ds.sizes['time'] → n_snapshots
    base_counter[model_in_f] = cumulative_total
    cumulative_total += n_snapshots
```

Within each worker, each matched pair is numbered
`base_counter + local_match_index` (0-based within the file).
Counters may have gaps (unmatched snapshots), but files from earlier model
files always have lower counters than files from later model files, so sorted
order is preserved.

For `no_matching`, counters are simply `0, 1, 2, …` (trivial – same as now).

---

## New / Modified Components

### `Namelist` – `model2obs/utils/namelist.py`

Add a `working_dir: Optional[str]` parameter to `__init__` (default `None`
→ `os.getcwd()`).

- `shutil.copy2(…)` writes to `os.path.join(working_dir, "input.nml.backup")`
- `self.symlink_dest = os.path.join(working_dir, "input.nml")`

Add a `from_content()` classmethod:
```python
@classmethod
def from_content(cls, content: str, working_dir: str) -> 'Namelist':
    """Create a Namelist instance from a pre-built content string."""
    # Write content to a temporary file inside working_dir, then call __init__
```
This lets workers start from the already-configured base state without
re-reading the template or re-applying all common parameters.

### `WorkflowModelObs` – `model2obs/workflows/workflow_model_obs.py`

#### `_initialize_model_namelist()` – minor addition
After building and configuring `self._namelist`, save:
```python
self._base_nml_content: str = self._namelist.content
```
This is the content workers copy from. `self._namelist` is still used for the
initial backup and the final `cleanup_namelist_symlink()` call (serial path).

#### `_process_model_obs_pair()` – self-contained refactor
Currently uses `self._namelist` directly.  Refactor so it:
1. Creates `worker_tmpdir = tempfile.mkdtemp(dir=self.config['tmp_folder'])`.
2. Instantiates `local_nml = Namelist.from_content(self._base_nml_content, working_dir=worker_tmpdir)`.
3. Applies per-pair namelist parameters to `local_nml` (input/output files, time).
4. Calls `local_nml.write_namelist(input_nml_bck_path)` and
   `local_nml.symlink_to_namelist(input_nml_bck_path)`.
5. Runs `subprocess.Popen([perfect_model_obs], cwd=worker_tmpdir, …)`.
6. Cleans up `worker_tmpdir` with `shutil.rmtree` in a `finally` block.

This makes `_process_model_obs_pair` safe to call from multiple threads
simultaneously and removes the dependency on `self._namelist` for pair
execution.  The serial path uses the same code path with no changes to
external behaviour.

The per-pair log argument changes from `"perfect_model_obs.log"` (shared, in
cwd) to `os.path.join(self.config['output_folder'], f"perfect_model_obs_{file_number}.log")`.

#### `_process_model_file_worker()` – NEW private method
Handles one model output file: performs time-matching for that file,
calls `_process_model_obs_pair` for each matched pair, and manages ncks
temporary slice files.

```python
def _process_model_file_worker(
    self,
    model_in_f: str,
    obs_in_files: List[str],      # read-only; worker tracks own used list
    base_counter: int,
    trim_obs: bool,
    hull_polygon: Optional[Any],
    hull_points: Optional[np.ndarray],
    force_obs_time: bool,
) -> int:
    """Process all matched pairs for a single model output file."""
```

Internals mirror the current inner loop of `_process_with_time_matching`
(open model file → iterate snapshots → find matching obs → ncks slice →
call `_process_model_obs_pair` → delete slice).

`obs_in_files` is passed as a read-only list; the worker tracks its own
`used_obs_in_files` locally.

#### `_validate_model_file_timestamps()` – NEW private method
Called once at the start of `process_files()`, before any dispatch, for both
the serial and parallel paths (the check is cheap and always desirable).

Opens every model file via `open_dataset_ctx`, collects all timestamp values
from the `time` variable, and raises a `ValueError` if any timestamp appears
more than once across the full set of files.

```python
def _validate_model_file_timestamps(self, model_in_files: List[str]) -> None:
    """Raise ValueError if any timestamp appears in more than one model file."""
    seen: Dict[Any, str] = {}   # timestamp → first file that contained it
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
```

This enforces the assumption that model files cover non-overlapping time
ranges, which is required for correct obs-file assignment in both serial and
parallel execution.

**Placement in `process_files()`:**
```
file lists resolved
hull_polygon computed (if trim_obs)
→ _validate_model_file_timestamps(model_in_files)   ← NEW, always runs
→ dispatch (serial or parallel)
```

#### `process_files()` – signature + dispatch

Add `parallel: bool = False` parameter.

**Serial path** (`parallel=False`): identical to current behaviour; calls
`_process_with_time_matching` and the `no_matching` loop as-is.

**Parallel path** (`parallel=True`):

_Time-matching:_
```
serial: base_counter pre-scan (metadata read)
ThreadPoolExecutor(max_workers=None):
    map(_process_model_file_worker, model_in_files,
        obs_in_files=[obs_in_files]*N,
        base_counters=[base_counter[f] for f in model_in_files],
        …)
```

_no_matching:_
```
ThreadPoolExecutor(max_workers=None):
    submit _process_model_obs_pair for each
           (counter, model_in_file, obs_in_file)
           in enumerate(zip(model_in_files, obs_in_files))
```

Exceptions from workers are re-raised in the main thread via
`future.result()`.

#### `run()` – signature addition
Add `parallel: bool = False`; passes it through to `process_files()`.

---

## Call Graph (parallel time-matching path)

```
run(parallel=True)
  └─ process_files(parallel=True)
       ├─ _initialize_model_namelist()                serial
       ├─ _validate_model_file_timestamps()           serial, always runs
       ├─ base_counter pre-scan                       serial
       └─ ThreadPoolExecutor
            ├─ worker[0]: _process_model_file_worker(model_files[0], …)
            │     time-match snapshot 0 → _process_model_obs_pair(…)  [own tmpdir]
            │     time-match snapshot 1 → _process_model_obs_pair(…)  [own tmpdir]
            │     …
            ├─ worker[1]: _process_model_file_worker(model_files[1], …)
            │     …
            └─ …
```

---

## Files to Change

| File | Change |
|---|---|
| `model2obs/workflows/workflow_model_obs.py` | Add `parallel` param to `run()` and `process_files()`; add `_validate_model_file_timestamps()`; add `_process_model_file_worker()`; refactor `_process_model_obs_pair()` to be self-contained (own tmpdir + Namelist); add base_counter pre-scan; add parallel dispatch for `no_matching` path; save `self._base_nml_content` in `_initialize_model_namelist()` |
| `model2obs/utils/namelist.py` | Add `working_dir` param to `__init__`; add `from_content()` classmethod |
| `tests/unit/test_workflow_model_obs_methods.py` | Add tests for `_process_model_file_worker`, base_counter pre-scan, parallel dispatch; update existing `_process_model_obs_pair` tests for the self-contained (tmpdir) design |
| `tests/integration/test_workflow_model_obs_integration.py` | Add integration test for `parallel=True` producing same output as `parallel=False` |
| `pm/planning/PLANNING_REPORT.md` | Update with final decisions |
| `pm/planning/PLANNING_STATUS.md` | Update task list |

---

## Edge Cases and Risks

| Item | Notes |
|---|---|
| Worker exception handling | All `Future.result()` calls must be made in the main thread so exceptions propagate and the run fails loudly. Use `as_completed` or collect results explicitly. |
| ncks on same file from multiple workers | Concurrent reads of the same `.nc` file are safe (read-only). Verified for local and typical NFS filesystems. |
| Obs file assigned to two workers | The `obs_in_files` list is passed as read-only to every worker. Each worker maintains its own `used_obs_in_files` list. Since model files are non-overlapping time ranges, the same obs file should not be claimed by two workers. `_validate_model_file_timestamps()` enforces this assumption at startup and raises `ValueError` if any timestamp appears in more than one file. |
| Output folder write concurrency | Each worker writes unique filenames (`obs_seq_{counter:04d}.out`, `perfect_output_{counter:04d}.nc`); no collision possible given the counter pre-assignment design. |
| `trimmed_obs_folder` write concurrency | Each `trim_obs` call writes `trimmed_obs_seq_{counter:04d}.in` with a unique counter; no collision. |
| `input_nml_bck` write concurrency | Each worker writes `input.nml_{counter:04d}.backup` with a unique counter; no collision. |
| `process_files` cleanup call | `self._namelist.cleanup_namelist_symlink()` at end of `process_files` is kept; in the parallel path the symlink is in a tmpdir (already removed), so this call is a harmless no-op. |
