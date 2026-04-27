"""
core/runner.py
==============
Parallel job runner with resume, atomic checkpointing, and Ctrl+C drain.

This is a generalisation of the parallel/resume machinery that
experiment3.py and experiment_ablation.py duplicated. Both phases
(conversation generation and bulk scoring) run jobs that share these
needs:

  - Each job is independent and CPU-light (the bottleneck is LLM API calls).
  - Jobs can fail individually without crashing the run.
  - The user may Ctrl+C; we want to drain in-flight completions and save.
  - On resume, we want to skip jobs whose output already exists on disk.

`run_jobs(...)` is the single entry point. The caller supplies:

  - jobs : list[Job]                 — opaque payloads to be passed to worker
  - is_done : Callable[[Job], bool]  — should we skip this job?
  - worker : Callable[[Job], Result] — runs one job, returns a result dict

The worker is responsible for writing its own outputs (atomic_write_*
from storage.py). The runner only manages the pool, prints progress,
and writes a checkpoint of completed/failed job IDs.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from .storage import RunPaths, atomic_write_json

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Types
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Job:
    """One unit of work for the runner.

    `job_id` must be unique within a run. `payload` is opaque to the
    runner and is passed verbatim to the worker.
    """
    job_id: str
    payload: Any


@dataclass
class JobResult:
    """Worker return value.

    `status` is "ok" | "error". On "ok", `info` may carry summary fields
    for progress display (e.g. mean correction). On "error", `info`
    should include "error" and "traceback".
    """
    job_id: str
    status: str
    info: dict = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════════════════════

def run_jobs(
    *,
    jobs: list[Job],
    worker: Callable[[Any], JobResult],
    paths: RunPaths,
    n_workers: int = 8,
    is_done: Callable[[Job], bool] | None = None,
    progress_label: str = "job",
    checkpoint_name: str = "checkpoint.json",
) -> tuple[list[JobResult], list[JobResult]]:
    """Run jobs in parallel, with resume and Ctrl+C drain.

    Parameters
    ----------
    jobs : list[Job]
        Full job set. Resume filtering happens internally via `is_done`.
    worker : Callable[[Any], JobResult]
        Runs one job. Must be thread-safe — no shared mutable state. The
        worker receives `job.payload` (not the Job itself) for ergonomics.
        IO is the worker's responsibility; the runner does not write
        outputs other than the checkpoint.
    paths : RunPaths
        Used only for the checkpoint location.
    n_workers : int
        Pool size.
    is_done : Callable[[Job], bool] | None
        Returns True if a job's output already exists and can be skipped.
        Default: never skip.
    progress_label : str
        Label used in progress output (e.g. "session", "scoring").
    checkpoint_name : str
        Filename for the checkpoint within paths.root. Multiple phases
        can coexist by passing different names (e.g. "checkpoint.json"
        for conversations, "scoring_checkpoint.json" for scoring).

    Returns
    -------
    (completed, failed) : (list[JobResult], list[JobResult])
        Just-completed jobs from this invocation. Skipped jobs are not
        returned. Run a second time with the same jobs list to re-attempt
        only the failed ones.
    """
    is_done = is_done or (lambda j: False)
    checkpoint_path = paths.root / checkpoint_name

    # ── Filter to remaining jobs ─────────────────────────────────────────
    remaining: list[Job] = []
    n_skipped = 0
    for job in jobs:
        if is_done(job):
            n_skipped += 1
            continue
        remaining.append(job)

    total = len(jobs)
    print(
        f"\n  {len(remaining)} {progress_label}(s) to run "
        f"({n_skipped} already complete out of {total}). "
        f"Workers: {n_workers}.\n"
    )

    if not remaining:
        return [], []

    # ── Parallel execution ───────────────────────────────────────────────
    completed: list[JobResult] = []
    failed: list[JobResult] = []
    save_lock = threading.Lock()

    def persist_checkpoint() -> None:
        atomic_write_json(checkpoint_path, {
            "phase": progress_label,
            "n_total": total,
            "n_skipped_at_start": n_skipped,
            "n_completed": len(completed),
            "n_failed": len(failed),
            "completed_ids": [r.job_id for r in completed],
            "failed": [
                {"job_id": r.job_id, "info": r.info}
                for r in failed
            ],
            "last_updated": datetime.now().isoformat(),
        })

    t_start = time.time()
    n_consumed = 0
    pool = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="job")
    future_to_job: dict = {}
    interrupted = False

    try:
        future_to_job = {
            pool.submit(_safe_call, worker, job): job
            for job in remaining
        }

        for fut in as_completed(future_to_job):
            n_consumed += 1
            job = future_to_job[fut]
            try:
                result = fut.result()
            except Exception as e:
                # _safe_call should catch — defensive.
                result = JobResult(
                    job_id=job.job_id,
                    status="error",
                    info={
                        "error": f"runner-level exception: {e}",
                        "traceback": traceback.format_exc(),
                    },
                )

            elapsed = time.time() - t_start
            rate = elapsed / n_consumed if n_consumed > 0 else 0
            eta = rate * (len(remaining) - n_consumed)
            n_total_done = n_skipped + n_consumed

            with save_lock:
                if result.status == "ok":
                    completed.append(result)
                    info_str = " ".join(
                        f"{k}={v}" for k, v in result.info.items()
                    )
                    print(
                        f"  [{n_total_done}/{total}] OK   {result.job_id}  "
                        f"{info_str}  | "
                        f"elapsed {elapsed/60:.1f}min  "
                        f"ETA {eta/60:.1f}min"
                    )
                else:
                    failed.append(result)
                    err = result.info.get("error", "")
                    print(
                        f"  [{n_total_done}/{total}] FAIL {result.job_id}  "
                        f"{str(err)[:80]}  | "
                        f"elapsed {elapsed/60:.1f}min"
                    )
                    if result.info.get("traceback"):
                        print(result.info["traceback"])

                persist_checkpoint()

    except KeyboardInterrupt:
        interrupted = True
        print(
            "\n*** KeyboardInterrupt — cancelling queued jobs, draining "
            "in-flight completions, and saving checkpoint. "
            "Press Ctrl+C again to force-exit. ***\n"
        )
        pool.shutdown(wait=False, cancel_futures=True)

        try:
            drained = 0
            for fut, job in future_to_job.items():
                if not fut.done() or fut.cancelled():
                    continue
                try:
                    result = fut.result(timeout=0)
                except Exception:
                    continue
                if result.job_id in {r.job_id for r in completed}:
                    continue
                if result.job_id in {r.job_id for r in failed}:
                    continue
                with save_lock:
                    if result.status == "ok":
                        completed.append(result)
                    else:
                        failed.append(result)
                    drained += 1
            if drained:
                print(f"  Drained {drained} completed job(s) before shutdown.")
        except KeyboardInterrupt:
            print("  Second Ctrl+C — skipping drain.")

        with save_lock:
            persist_checkpoint()
        print(
            f"\n  Saved checkpoint: {len(completed)} completed, "
            f"{len(failed)} failed. Resume by re-running.\n"
        )
        raise
    finally:
        pool.shutdown(wait=not interrupted)

    return completed, failed


def _safe_call(worker: Callable, job: Job) -> JobResult:
    """Wrap the worker so unhandled exceptions become structured errors."""
    try:
        return worker(job.payload)
    except Exception as e:
        return JobResult(
            job_id=job.job_id,
            status="error",
            info={
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )