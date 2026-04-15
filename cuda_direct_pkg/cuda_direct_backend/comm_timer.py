"""
Communication timing for cuda_direct collectives.

Records every collective call as a raw CommEvent. Consumers (training
profilers, benchmarks, etc.) drain the event list and aggregate however
they need.

Usage
-----
Attach a CommTimer to a ProcessGroup:

    pg.comm_timer = CommTimer(rank=rank)

Tag the current phase from outside (e.g. training profiler):

    pg.comm_timer.set_context("fwd")
    # ... forward pass ...
    pg.comm_timer.set_context("bwd")

    # or with context manager:
    with pg.comm_timer.phase("fwd"):
        ...

Collect and aggregate at step end:

    events = pg.comm_timer.drain()
    stats  = CommTimer.summary(events, group_by=("op",))
    stats  = CommTimer.summary(events, group_by=("context", "op"))
    print(CommTimer.format_summary(stats))
"""

import threading
import time
from collections import defaultdict, namedtuple
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# Raw event
# ---------------------------------------------------------------------------

CommEvent = namedtuple("CommEvent", [
    "op",            # str:   allgather | reduce_scatter | allreduce
                     #        alltoall  | broadcast
    "algo",          # str:   direct | ring | shm
    "rank",          # int:   local rank that recorded this event
    "call_seq",      # int:   monotonic counter, unique per CommTimer instance
    "timestamp_ms",  # float: perf_counter (seconds * 1000) at call start —
                     #        enables timeline reconstruction and gap analysis
    "numel",         # int:   number of elements in the *input* tensor
                     #        (shard size for allgather/reduce_scatter,
                     #         full tensor for allreduce/broadcast/alltoall)
    "dtype",         # str:   tensor dtype, e.g. "bf16", "fp32"
    "bytes",         # int:   numel * element_size — bytes of the input tensor.
                     #        Combined with duration_ms gives bandwidth.
    "duration_ms",   # float: wall time from first CPU instruction of the
                     #        collective to the final synchronize() inside
                     #        the worker thread
    "context",       # str:   free-form tag set by external code.
                     #        Training profiler sets "fwd"/"bwd"/"comm".
                     #        Leave empty if not needed.
])


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class CommTimer:
    """Thread-safe recorder for collective call events.

    One instance per ProcessGroup. The worker thread calls record(); the
    main training thread calls drain() and summary().
    """

    def __init__(self, rank: int):
        self.rank = rank
        self._lock = threading.Lock()
        self._events: list[CommEvent] = []
        self._call_seq: int = 0
        self._context: str = ""

    # ------------------------------------------------------------------
    # Context control  (called from the main training thread)
    # ------------------------------------------------------------------

    def set_context(self, ctx: str) -> None:
        """Set the current phase tag. Thread-safe.

        The tag is stamped on every event recorded until the next
        set_context() call.  Example values: "fwd", "bwd", "comm", "opt".
        Pass "" to clear.
        """
        with self._lock:
            self._context = ctx

    @contextmanager
    def phase(self, ctx: str):
        """Context manager that sets a phase tag and restores the previous one.

        Example::

            with pg.comm_timer.phase("fwd"):
                model(batch)
        """
        with self._lock:
            prev = self._context
            self._context = ctx
        try:
            yield
        finally:
            with self._lock:
                self._context = prev

    # ------------------------------------------------------------------
    # Recording  (called from the worker / collective thread)
    # ------------------------------------------------------------------

    def record(
        self,
        op: str,
        algo: str,
        duration_ms: float,
        timestamp_ms: float,
        numel: int,
        dtype: str,
        bytes_transferred: int,
    ) -> None:
        """Record one collective call. Called from the async worker thread.

        Parameters
        ----------
        op:
            Collective name: "allgather", "reduce_scatter", "allreduce",
            "alltoall", "broadcast".
        algo:
            Algorithm variant: "direct", "ring", or "shm".
        duration_ms:
            Wall time in milliseconds from start to end of the collective,
            including all internal barriers and synchronize() calls.
        timestamp_ms:
            ``time.perf_counter() * 1000`` captured at the start of the
            collective, before any GPU or sync work.
        numel:
            Element count of the input tensor handed to the collective.
        dtype:
            Human-readable dtype string, e.g. "bf16", "fp32".
        bytes_transferred:
            ``numel * element_size`` — bytes of the input tensor.
        """
        with self._lock:
            seq = self._call_seq
            self._call_seq += 1
            ctx = self._context
            self._events.append(CommEvent(
                op=op,
                algo=algo,
                rank=self.rank,
                call_seq=seq,
                timestamp_ms=timestamp_ms,
                numel=numel,
                dtype=dtype,
                bytes=bytes_transferred,
                duration_ms=duration_ms,
                context=ctx,
            ))

    # ------------------------------------------------------------------
    # Retrieval  (called from the main training thread)
    # ------------------------------------------------------------------

    def drain(self) -> list[CommEvent]:
        """Return all recorded events and reset the internal list.

        Thread-safe. Call this once per profiling window (e.g. per step).
        The returned list is a snapshot — further records go into a fresh list.
        """
        with self._lock:
            events = self._events
            self._events = []
            return events

    def peek(self) -> list[CommEvent]:
        """Return a copy of recorded events without resetting.

        Useful for debugging mid-step; prefer drain() at step boundaries.
        """
        with self._lock:
            return list(self._events)

    # ------------------------------------------------------------------
    # Aggregation  (static — works on any list of CommEvents)
    # ------------------------------------------------------------------

    @staticmethod
    def summary(
        events: list[CommEvent],
        group_by: tuple[str, ...] = ("op",),
    ) -> dict[tuple, dict]:
        """Aggregate a list of CommEvents into summary statistics.

        Parameters
        ----------
        events:
            Output of drain() or peek().
        group_by:
            Tuple of CommEvent field names to group by.
            Common groupings:

            ``("op",)``            — totals per collective type
            ``("context", "op")``  — totals per training phase × op
            ``("algo", "op")``     — compare algorithms
            ``("op", "dtype")``    — see if dtype affects latency

        Returns
        -------
        dict keyed by a tuple of the grouped field values, e.g.
        ``("fwd", "allgather")`` for ``group_by=("context", "op")``.

        Each value is a dict::

            {
                "total_ms":       float,  # sum of duration_ms
                "count":          int,    # number of calls
                "total_mb":       float,  # total input bytes in MB
                "avg_ms":         float,  # mean duration per call
                "avg_mb":         float,  # mean input MB per call
                "bandwidth_gbps": float,  # total_bytes / total_seconds
            }
        """
        groups: dict = defaultdict(lambda: {
            "total_ms": 0.0, "count": 0, "total_bytes": 0
        })

        for e in events:
            key = tuple(getattr(e, f) for f in group_by)
            g = groups[key]
            g["total_ms"]    += e.duration_ms
            g["count"]       += 1
            g["total_bytes"] += e.bytes

        result = {}
        for key, g in sorted(groups.items(), key=lambda x: -x[1]["total_ms"]):
            total_ms    = g["total_ms"]
            count       = g["count"]
            total_bytes = g["total_bytes"]
            total_mb    = total_bytes / 1024 ** 2
            avg_ms      = total_ms    / count if count else 0.0
            avg_mb      = total_mb    / count if count else 0.0
            # bandwidth: total input bytes moved, divided by total seconds
            bw_gbps = (total_bytes / 1e9) / (total_ms / 1e3) if total_ms > 0 else 0.0
            result[key] = {
                "total_ms":       round(total_ms,  2),
                "count":          count,
                "total_mb":       round(total_mb,  2),
                "avg_ms":         round(avg_ms,    2),
                "avg_mb":         round(avg_mb,    2),
                "bandwidth_gbps": round(bw_gbps,   3),
            }

        return result

    @staticmethod
    def format_summary(
        stats: dict[tuple, dict],
        group_by: tuple[str, ...] = ("op",),
    ) -> str:
        """Format a summary dict as a human-readable string.

        Designed to slot cleanly into log output. Example::

            allgather        count=12  total=1840ms  avg=153ms  98MB  0.636 GB/s
            reduce_scatter   count=12  total=1920ms  avg=160ms  98MB  0.611 GB/s
            allreduce        count= 1  total= 185ms  avg=185ms  48MB  0.259 GB/s
        """
        if not stats:
            return "  (no comm events)"

        lines = []
        # figure out the widest key string for alignment
        key_strs = [
            "  " + " / ".join(str(v) for v in key)
            for key in stats
        ]
        width = max(len(s) for s in key_strs)

        for key_str, (key, s) in zip(key_strs, stats.items()):
            lines.append(
                f"{key_str:<{width}}  "
                f"count={s['count']:>3}  "
                f"total={s['total_ms']:>7.1f}ms  "
                f"avg={s['avg_ms']:>6.1f}ms  "
                f"{s['total_mb']:>6.1f}MB  "
                f"{s['bandwidth_gbps']:.3f} GB/s"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TODO: barrier_ms breakdown
# ---------------------------------------------------------------------------
# Each CommEvent currently records total duration_ms which includes both
# actual DMA/transfer time AND time spent spin-waiting in barriers between
# ranks. Separating these would tell you whether a slow collective is caused
# by rank imbalance (one rank arrives at the barrier late) vs actual data
# volume (the transfer itself is slow).
#
# Plan:
#   1. Add `barrier_ms: float = 0.0` field to CommEvent
#   2. Add optional accumulator arg to SharedMemorySync.barrier() in sync.py:
#        def barrier(self, _acc: list | None = None):
#            t0 = time.perf_counter()
#            self._barrier.wait()
#            if _acc is not None:
#                _acc[0] += (time.perf_counter() - t0) * 1000
#   3. In each collective in collectives.py, pass a [0.0] accumulator through
#      all barrier() calls, then return the total to process_group.py
#   4. process_group.py passes barrier_ms to comm_timer.record()
#
# Caveat: SharedBarrier.wait() uses time.sleep(0.0001) so granularity is
# ~0.1ms. Fine for barriers >1ms, rough for sub-ms barriers.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers used by process_group.py when calling record()
# ---------------------------------------------------------------------------

def dtype_str(tensor) -> str:
    """Return a short dtype string from a torch tensor, e.g. 'bf16', 'fp32'."""
    _MAP = {
        "torch.bfloat16":  "bf16",
        "torch.float16":   "fp16",
        "torch.float32":   "fp32",
        "torch.float64":   "fp64",
        "torch.int32":     "int32",
        "torch.int64":     "int64",
        "torch.int8":      "int8",
        "torch.uint8":     "uint8",
        "torch.float8_e4m3fn":   "fp8_e4m3",
        "torch.float8_e5m2":     "fp8_e5m2",
    }
    return _MAP.get(str(tensor.dtype), str(tensor.dtype).replace("torch.", ""))
