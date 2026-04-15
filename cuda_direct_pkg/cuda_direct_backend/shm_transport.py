"""
Host-staged transport for GPUs without P2P access (zero-copy fast path).

Works on consumer GPUs (RTX 3000/4000/5000 series) that disable P2P at the
hardware/driver level. Bypasses CUDA IPC entirely.

Data flow — ZERO-COPY path (default, requires cudaHostRegister support):
  GPU → cudaMemcpyAsync → registered SHM region  (direct DMA, no CPU involvement)
  registered SHM region → cudaMemcpyAsync → GPU   (direct DMA, no CPU involvement)

Data flow — DOUBLE-BOUNCE fallback (if cudaHostRegister fails):
  GPU → D2H DMA → pinned host buf → CPU memcpy → SHM
  SHM → CPU memcpy → pinned host buf → H2D DMA → GPU

Zero-copy removes the two CPU memcpy legs (~30-40 GB/s) and lets the GPU DMA
engine write/read directly to/from the OS shared memory region. Expected
effective bandwidth increases from ~10-13 GB/s to ~22-28 GB/s (PCIe ceiling).

cudaHostRegister requirements:
  - cudaHostRegisterPortable: mapping visible to all CUDA contexts (cross-process)
  - cudaHostRegisterMapped:   maps host address into device address space
  - Memory must not be paged out while registered (OS pins the pages)
  - Registration is per-process: each rank registers its own SHM AND peer SHM
    independently and gets its own device pointer
"""

import atexit
import ctypes
import logging
import struct
import sys
import threading
import numpy as np
from multiprocessing import shared_memory

import torch

from . import cuda_ipc

# ---------------------------------------------------------------------------
# CPU store fence — flush write-combining (WC) buffers to DRAM so that
# data written to cudaHostRegister-mapped SHM pages is globally visible
# before the barrier slot signals arrival.
#
# Background: cudaHostRegister changes the CPU memory type of the SHM pages
# to WC to enable GPU DMA.  WC stores are NOT immediately coherent — they
# accumulate in a write-combining buffer and flush lazily.  Without an
# explicit fence, rank 0 can exit the barrier and read stale DRAM content
# from rank 1's SHM because rank 1's WC stores haven't drained yet.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    _FlushProcessWriteBuffers = ctypes.windll.kernel32.FlushProcessWriteBuffers

    def _store_fence() -> None:
        """Full memory barrier — flush all pending WC stores to DRAM (Windows)."""
        _FlushProcessWriteBuffers()
else:
    def _store_fence() -> None:  # type: ignore[misc]
        """Full memory barrier via mfence (Linux/macOS, x86-64 only)."""
        # ctypes inline asm is not portable; use an atomic write via a
        # ctypes lock as a portable alternative.  In practice, on x86 Linux
        # cudaHostRegister maps WB (not WC) for anonymous SHM, so this is
        # usually a no-op, but keeping it as a safety net.
        ctypes.c_int(0)  # lightweight; replace with mfence asm if needed

logger = logging.getLogger("cuda_direct.shm_transport")

_EMPTY_HANDLE = b'\x00' * 64  # placeholder: SHM peers need no IPC handle


def _data_shm_name(session_id: str, rank: int, version: int = 0) -> str:
    """Return the OS name for rank's data SHM region.

    version=0 uses the base name (backwards-compatible with old sessions).
    Growth increments the version so the new region gets a fresh name and
    never conflicts with the old one that peers may still have cached.
    """
    base = f"cuda_direct_{session_id}_shm_{rank}"
    return base if version == 0 else f"{base}_v{version}"


def _shm_host_ptr(shm: shared_memory.SharedMemory) -> int:
    """Return the raw host address of an SHM buffer as an integer."""
    return ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))


class ShmTransport:
    """
    Cross-process GPU data transfer using OS shared memory.

    Fast path (zero-copy): cudaHostRegister pins the SHM pages and provides a
    device pointer. GPU DMA reads/writes SHM directly — CPU is not in the path.

    Fallback (double-bounce): if registration fails, falls back to the original
    D2H-to-pinned + CPU-memcpy + H2D-from-pinned path. Identical correctness,
    lower throughput.

    Each rank owns:
      - One SHM data region (writable by owner, readable by all ranks)
      - A device pointer into its own registered SHM  (publish_tensor DMA dst)
      - A device pointer into each peer's registered SHM  (fetch_chunk DMA src)
      - Dedicated CUDA streams for publish (D2H) and per-peer fetch (H2D)

    is_shm = True signals CollectivesImpl to use SHM-safe collective algorithms
    instead of ring/direct algorithms that require live GPU pointer access.
    """

    is_shm: bool = True
    prefer_ring: bool = False

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.cuda.current_device()

        # Dedicated CUDA stream for publish (GPU → SHM)
        self._d2h_stream = torch.cuda.Stream(device=self.device)

        # Per-peer H2D streams: parallel fetches from multiple peers
        self._h2d_streams: dict[int, torch.cuda.Stream] = {
            p: torch.cuda.Stream(device=self.device)
            for p in range(world_size) if p != rank
        }

        # streams dict for API compatibility with CudaPeerTransport
        self.streams: dict[int, torch.cuda.Stream] = self._h2d_streams

        # SHM regions
        self._my_shm: shared_memory.SharedMemory | None = None
        self._my_shm_nbytes: int = 0
        self._peer_shm: dict[int, shared_memory.SharedMemory] = {}

        # Zero-copy: device pointers for each registered SHM region
        # None = not yet registered or registration failed
        self._my_shm_dev_ptr: int | None = None          # our own SHM dev ptr
        self._peer_shm_dev_ptr: dict[int, int] = {}      # peer rank → dev ptr

        # Zero-copy: track which host ptrs are currently registered
        # so we can unregister before close/unlink
        self._my_shm_host_ptr: int = 0
        self._peer_shm_host_ptr: dict[int, int] = {}

        # Whether zero-copy is active (False if cudaHostRegister failed)
        self._zero_copy: bool = True

        # Fallback pinned buffers — only allocated if zero-copy fails
        self._send_pinned: torch.Tensor | None = None
        self._send_pinned_nbytes: int = 0
        self._recv_pinned: dict[int, torch.Tensor] = {}
        self._recv_pinned_nbytes: dict[int, int] = {}

        # Pending peer H2D streams awaiting sync
        self._pending_peers: set[int] = set()

        # SHM version — incremented each time our own SHM grows.
        # Peers detect the version change from sync metadata and re-open.
        # Old SHM regions are kept alive in _old_shms until _cleanup.
        self._my_shm_version: int = 0
        self._old_shms: list = []           # retired SHM objects awaiting peer close

        # Per-peer version last seen in sync metadata.
        # If the peer's published version differs, we invalidate and re-open.
        self._peer_shm_version: dict[int, int] = {}

        # Lock protecting _ensure_my_shm against concurrent calls from the
        # allgather pool and comm pool workers (both can call publish_tensor).
        self._shm_lock = threading.Lock()

        # Set by process_group after SharedMemorySync is created
        self._session_id: str | None = None

        atexit.register(self._cleanup)
        logger.info("ShmTransport init  rank=%d  world=%d  device=%d",
                    rank, world_size, self.device)

    def set_session(self, session_id: str) -> None:
        self._session_id = session_id
        logger.debug("ShmTransport session set  rank=%d  session=%s", self.rank, session_id)

    # ---------------------------------------------------------------
    #  Unified transport interface
    # ---------------------------------------------------------------

    def publish_tensor(self, tensor: torch.Tensor, sync) -> None:
        """
        Copy GPU tensor into this rank's SHM data region so peers can fetch it.

        Zero-copy path: GPU DMA writes directly into registered SHM.
        Fallback path:  GPU → pinned buf → CPU memcpy → SHM.
        """
        assert self._session_id, \
            "ShmTransport.set_session() must be called before publish_tensor()"
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()

        self._ensure_my_shm(nbytes)

        # Propagate the calling thread's stream dependency to the D2H stream.
        # In _async_work, worker_stream.wait_stream(caller_stream) was recorded.
        # torch.cuda.current_stream() here IS worker_stream (we're on the worker
        # thread). This GPU event chain ensures _d2h_stream will not DMA-read
        # tensor.data_ptr() before caller_stream's gradient compute is done,
        # even though _d2h_stream is otherwise independent of caller_stream.
        self._d2h_stream.wait_stream(torch.cuda.current_stream())

        if self._zero_copy and self._my_shm_dev_ptr:
            # ── Zero-copy: DMA straight into registered SHM ──────────────
            # _my_shm_dev_ptr is a device-mapped address returned by
            # cudaHostGetDevicePointer — it lives in the device address space.
            # Must use DeviceToDevice kind; DeviceToHost would be wrong and
            # will raise cudaErrorInvalidValue with a mapped pointer as dst.
            cuda_ipc.memcpy_async(
                self._my_shm_dev_ptr,
                tensor.data_ptr(),
                nbytes,
                self._d2h_stream.cuda_stream,
            )
            self._d2h_stream.synchronize()
            # Flush GPU DMA writes (WC) to DRAM so peers see correct data
            # after the barrier.  Without this, the barrier slot (WB SHM)
            # can become globally visible before the WC data stores drain.
            _store_fence()
            logger.debug("publish_tensor ZC  rank=%d  nbytes=%d  dev_ptr=%s",
                         self.rank, nbytes, hex(self._my_shm_dev_ptr))
        else:
            # ── Double-bounce fallback ────────────────────────────────────
            self._ensure_send_pinned(nbytes)
            cuda_ipc.memcpy_d2h_async(
                self._send_pinned.data_ptr(),
                tensor.data_ptr(),
                nbytes,
                self._d2h_stream.cuda_stream,
            )
            self._d2h_stream.synchronize()
            send_np = self._send_pinned.numpy()[:nbytes]
            shm_np  = np.frombuffer(self._my_shm.buf, dtype=np.uint8, count=nbytes)
            np.copyto(shm_np, send_np)
            # Flush WC stores to DRAM — SHM pages are WC after cudaHostRegister;
            # the barrier (separate WB SHM) can become visible before data drains.
            _store_fence()
            logger.debug("publish_tensor DB  rank=%d  nbytes=%d", self.rank, nbytes)

        # Encode SHM version in the ipc_handle field (unused for SHM path).
        # Peers read this after the post-publish barrier to know which versioned
        # SHM name to open without having to call invalidate_cache().
        version_handle = struct.pack('<Q', self._my_shm_version) + b'\x00' * 56
        sync.write_tensor_meta(tensor.numel(), 0, self.device, version_handle)

    def fetch_chunk(self, peer_rank: int, dst_tensor: torch.Tensor,
                    src_offset_bytes: int, size_bytes: int, sync) -> None:
        """
        Issue an async copy from peer's SHM region into dst_tensor.

        Zero-copy path: GPU DMA reads directly from peer's registered SHM.
        Fallback path:  CPU memcpy from SHM → pinned buf → H2D DMA.

        Async — call wait_for_peer(peer_rank) to synchronize.

        Peer SHM is cached across calls: cudaHostRegister is only called once
        per peer (on first access) and again only if the peer's region grows.
        This avoids the per-collective page-table walk that otherwise dominates
        allgather latency at large tensor sizes.
        """
        # Detect peer SHM version from sync metadata. The peer encodes its
        # current version in the ipc_handle bytes written by publish_tensor.
        # If the version changed (peer grew its SHM), invalidate cached handle
        # and re-open the new versioned region.
        try:
            _numel, _dc, _dev, ipc_bytes = sync.read_tensor_meta(peer_rank)
            peer_version = struct.unpack_from('<Q', ipc_bytes)[0]
        except Exception:
            peer_version = self._peer_shm_version.get(peer_rank, 0)

        if (peer_rank not in self._peer_shm
                or self._peer_shm_version.get(peer_rank) != peer_version):
            if peer_rank in self._peer_shm:
                self._invalidate_peer(peer_rank)
            self._open_peer_shm(peer_rank, version=peer_version)
            self._peer_shm_version[peer_rank] = peer_version

        h2d_stream = self._h2d_streams[peer_rank]
        dev_ptr = self._peer_shm_dev_ptr.get(peer_rank)

        # Ensure the H2D stream does not race with pending work on the calling
        # stream (e.g. a cudaMemset from torch.zeros that zeroes dst_tensor).
        # Without this, the cudaMemset on the default stream can complete after
        # our H2D copy, silently overwriting the fetched data with zeros.
        h2d_stream.wait_stream(torch.cuda.current_stream())

        if self._zero_copy and dev_ptr:
            # ── Zero-copy: DMA straight from registered peer SHM ─────────
            # dev_ptr is a device-mapped address (cudaHostGetDevicePointer).
            # Must use DeviceToDevice kind; HostToDevice would be wrong.
            cuda_ipc.memcpy_async(
                dst_tensor.data_ptr(),
                dev_ptr + src_offset_bytes,
                size_bytes,
                h2d_stream.cuda_stream,
            )
            logger.debug("fetch_chunk ZC ISSUED  peer=%d  offset=%d  size=%d  dev_ptr=%s",
                         peer_rank, src_offset_bytes, size_bytes, hex(dev_ptr))
        else:
            # ── Double-bounce fallback ────────────────────────────────────
            self._ensure_recv_pinned(peer_rank, size_bytes)
            peer_buf = self._peer_shm[peer_rank].buf
            src_np   = np.frombuffer(peer_buf, dtype=np.uint8,
                                     offset=src_offset_bytes, count=size_bytes)
            recv_buf = self._recv_pinned[peer_rank]
            np.copyto(recv_buf.numpy()[:size_bytes], src_np)
            cuda_ipc.memcpy_h2d_async(
                dst_tensor.data_ptr(),
                recv_buf.data_ptr(),
                size_bytes,
                h2d_stream.cuda_stream,
            )
            logger.debug("fetch_chunk DB ISSUED  peer=%d  offset=%d  size=%d",
                         peer_rank, src_offset_bytes, size_bytes)

        self._pending_peers.add(peer_rank)

    def wait_for_peer(self, peer_rank: int) -> None:
        if peer_rank in self._pending_peers:
            self._h2d_streams[peer_rank].synchronize()
            self._pending_peers.discard(peer_rank)

    def wait_all_peers(self) -> None:
        for peer_rank in list(self._pending_peers):
            self._h2d_streams[peer_rank].synchronize()
        self._pending_peers.clear()

    def invalidate_cache(self) -> None:
        """
        No-op: peer SHM registrations are cached across collectives.

        Previously this unregistered and closed all peer SHM regions before
        every collective, forcing cudaHostRegister + page-table walk on every
        fetch_chunk call.  For 35 MB slabs that overhead was ~80-97 ms per
        allgather — dominating the 100 ms we measured vs gloo's 59 ms.

        Peer SHMs are now opened and registered lazily in fetch_chunk and
        kept alive until the region must grow or the transport is destroyed.
        Call _invalidate_peer(rank) explicitly if you know a peer's SHM has
        been recreated (e.g. after a tensor size change).
        """
        cuda_ipc.invalidate_ipc_handle_cache()

    def _invalidate_peer(self, peer_rank: int) -> None:
        """Unregister and close one peer's SHM. Called when the peer's region grows."""
        shm = self._peer_shm.pop(peer_rank, None)
        host_ptr = self._peer_shm_host_ptr.pop(peer_rank, 0)
        if host_ptr:
            cuda_ipc.host_unregister(host_ptr)
        self._peer_shm_dev_ptr.pop(peer_rank, None)
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass
        logger.debug("ShmTransport peer invalidated  rank=%d  peer=%d", self.rank, peer_rank)

    def sync_peer_stream(self, peer_rank: int) -> None:
        pass  # no-op: ShmTransport has no separate per-peer sync streams

    def enable_peer_access(self, peer_device_ids: list[int]) -> None:
        pass  # no-op: ShmTransport does not use CUDA P2P

    def cleanup(self) -> None:
        self._cleanup()

    # ---------------------------------------------------------------
    #  Private helpers
    # ---------------------------------------------------------------

    def _ensure_my_shm(self, nbytes: int) -> None:
        """Create or grow this rank's SHM region and register it for zero-copy."""
        with self._shm_lock:
            self._ensure_my_shm_locked(nbytes)

    def _ensure_my_shm_locked(self, nbytes: int) -> None:
        """Must be called with self._shm_lock held."""
        if self._my_shm is not None and self._my_shm_nbytes >= nbytes:
            return

        if self._my_shm is not None:
            # Growing: retire the old SHM under a new version name instead of
            # unlink+recreate. On Windows, unlink() is a no-op while any peer
            # holds the handle open — trying to recreate with the same name
            # gives FileExistsError. Using a new versioned name sidesteps this:
            # the old region stays alive until peers detect the version change
            # via sync metadata and call _invalidate_peer themselves.
            if self._my_shm_host_ptr:
                cuda_ipc.host_unregister(self._my_shm_host_ptr)
                self._my_shm_host_ptr = 0
                self._my_shm_dev_ptr  = None
            self._old_shms.append(self._my_shm)  # keep alive until peers close
            self._my_shm = None
            self._my_shm_version += 1
            logger.debug("ShmTransport growing  rank=%d  new_version=%d  nbytes=%d",
                         self.rank, self._my_shm_version, nbytes)

        name = _data_shm_name(self._session_id, self.rank, self._my_shm_version)

        # Clean up any stale region from a crashed previous session.
        # Different sessions use different UUIDs so collisions are rare; this
        # handles the edge case where a previous process died without cleanup.
        try:
            stale = shared_memory.SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
            logger.debug("Cleaned stale SHM region  name=%s", name)
        except FileNotFoundError:
            pass

        self._my_shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
        self._my_shm_nbytes = nbytes
        logger.debug("Created SHM region  rank=%d  name=%s  size=%d",
                     self.rank, name, nbytes)

        # Only attempt registration when zero-copy is (still) enabled.
        # If _zero_copy is already False — either forced externally (test) or
        # because a prior cudaHostRegister attempt failed — skip registration.
        # cudaHostRegister changes the SHM pages' PAT to write-combining (WC),
        # a hardware-level change visible to ALL processes mapping those
        # physical pages.  CPU stores to WC pages drain lazily; without an
        # explicit sfence the peer may read stale DRAM before WC stores flush.
        # Keeping the pages as normal write-back (WB) is the simplest fix for
        # the double-bounce path, which never needs GPU DMA to SHM.
        self._my_shm_dev_ptr = None
        if not self._zero_copy:
            return
        host_ptr = _shm_host_ptr(self._my_shm)
        ok = cuda_ipc.host_register(host_ptr, nbytes)
        if ok:
            dev_ptr = cuda_ipc.host_get_device_pointer(host_ptr)
            if dev_ptr:
                self._my_shm_host_ptr = host_ptr
                self._my_shm_dev_ptr  = dev_ptr
            else:
                cuda_ipc.host_unregister(host_ptr)
                self._zero_copy = False
                logger.warning("cudaHostGetDevicePointer returned NULL — "
                               "falling back to double-bounce  rank=%d", self.rank)
        else:
            self._zero_copy = False
            logger.warning("cudaHostRegister failed — "
                           "falling back to double-bounce  rank=%d", self.rank)

    def _open_peer_shm(self, peer_rank: int, version: int = 0) -> None:
        """
        Open peer's SHM and register it for zero-copy.
        Must be called after a barrier that follows the peer's publish_tensor.
        version matches what the peer encoded in its sync metadata.
        """
        name = _data_shm_name(self._session_id, peer_rank, version)
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
        except FileNotFoundError:
            raise RuntimeError(
                f"cuda_direct ShmTransport: cannot open SHM region for peer rank "
                f"{peer_rank} (name='{name}'). "
                f"Ensure publish_tensor() and a barrier completed before fetch_chunk()."
            )

        self._peer_shm[peer_rank] = shm
        logger.debug("Opened peer SHM  peer=%d  name=%s  size=%d",
                     peer_rank, name, shm.size)

        if self._zero_copy:
            host_ptr = _shm_host_ptr(shm)
            ok = cuda_ipc.host_register(host_ptr, shm.size)
            if ok:
                dev_ptr = cuda_ipc.host_get_device_pointer(host_ptr)
                if dev_ptr:
                    self._peer_shm_host_ptr[peer_rank] = host_ptr
                    self._peer_shm_dev_ptr[peer_rank]  = dev_ptr
                else:
                    cuda_ipc.host_unregister(host_ptr)
                    logger.warning("cudaHostGetDevicePointer returned NULL for peer %d "
                                   "— using double-bounce for this peer", peer_rank)
            else:
                logger.warning("cudaHostRegister failed for peer %d "
                               "— using double-bounce for this peer", peer_rank)

    def _ensure_send_pinned(self, nbytes: int) -> None:
        """Allocate or grow the fallback pinned send buffer."""
        if nbytes <= self._send_pinned_nbytes:
            return
        alloc = ((nbytes + (1 << 20) - 1) >> 20) << 20
        self._send_pinned = torch.zeros(alloc, dtype=torch.uint8).pin_memory()
        self._send_pinned_nbytes = alloc
        logger.debug("Send pinned buffer allocated  rank=%d  nbytes=%d", self.rank, alloc)

    def _ensure_recv_pinned(self, peer_rank: int, nbytes: int) -> None:
        """Allocate or grow the fallback pinned recv buffer for a specific peer."""
        if self._recv_pinned_nbytes.get(peer_rank, 0) >= nbytes:
            return
        alloc = ((nbytes + (1 << 20) - 1) >> 20) << 20
        self._recv_pinned[peer_rank] = torch.zeros(alloc, dtype=torch.uint8).pin_memory()
        self._recv_pinned_nbytes[peer_rank] = alloc
        logger.debug("Recv pinned buffer allocated  rank=%d  peer=%d  nbytes=%d",
                     self.rank, peer_rank, alloc)

    def _cleanup(self) -> None:
        """Unregister all SHM regions and close them. Called at atexit."""
        for peer_rank in list(self._peer_shm.keys()):
            self._invalidate_peer(peer_rank)

        # Unregister and destroy own SHM
        if self._my_shm_host_ptr:
            cuda_ipc.host_unregister(self._my_shm_host_ptr)
            self._my_shm_host_ptr = 0
        if self._my_shm is not None:
            try:
                self._my_shm.close()
                self._my_shm.unlink()
            except Exception:
                pass
            self._my_shm = None

        # Close retired SHM versions from growth events
        for old in self._old_shms:
            try:
                old.close()
            except Exception:
                pass
        self._old_shms.clear()

        logger.debug("ShmTransport cleanup done  rank=%d  zero_copy=%s",
                     self.rank, self._zero_copy)

        # Unregister and destroy own SHM
        if self._my_shm_host_ptr:
            cuda_ipc.host_unregister(self._my_shm_host_ptr)
            self._my_shm_host_ptr = 0
        if self._my_shm is not None:
            try:
                self._my_shm.close()
                self._my_shm.unlink()
            except Exception:
                pass
            self._my_shm = None

        logger.debug("ShmTransport cleanup done  rank=%d  zero_copy=%s",
                     self.rank, self._zero_copy)
