# SPDX-License-Identifier: Apache-2.0
"""Content-addressed store of loaded steer vector payloads.

Separates *vector data* (tensors loaded from disk, deduplicated by path)
from *application config* (scale / triggers / layers, which are per request
and cost nothing). Loading through the store happens at preload time or at
request admission — never inside the forward pass.
"""

import os
from collections import OrderedDict
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import SteerVectorConfig
    from vllm.steer_vectors.models import LoadedSteerVector

logger = init_logger(__name__)


def file_version(path: str) -> tuple[int, int]:
    """(mtime_ns, size) of a vector file, (0, 0) if unreadable.

    Used to detect regenerated vectors at the same path. For checkpoint
    directories this tracks the directory inode (best effort: in-place
    rewrites of files inside a directory may not bump it).
    """
    try:
        st = os.stat(path)
    except OSError:
        return (0, 0)
    return (st.st_mtime_ns, st.st_size)


class VectorStore:
    """LRU cache of loaded steer vector payloads.

    Keyed by (path, algorithm, mtime_ns, size) so a vector regenerated at
    the same path loads fresh instead of silently serving the stale cached
    payload, while byte-identical reuse still dedups. Entries hold
    unscaled per-layer payload tensors on the target device; per-request
    scaling is applied when a config is distributed to layer slots, so any
    number of configs can share one resident entry.
    """

    def __init__(self, device: str, steer_vector_config: "SteerVectorConfig"):
        self.device = device
        self.steer_vector_config = steer_vector_config
        self.capacity = max(1, steer_vector_config.max_steer_vectors)
        self._entries: OrderedDict[tuple, LoadedSteerVector] = OrderedDict()
        self._warned_lazy: set[str] = set()

    def get(
        self,
        path: str,
        algorithm: str,
        *,
        target_layers: list[int] | None = None,
        lazy: bool = False,
    ) -> "LoadedSteerVector":
        """Return the loaded entry for the file's current version, loading
        if needed.

        `target_layers` is forwarded to the loader: single-layer formats
        (.pt vectors, linear .pkl, lm_steer projectors) need it to build
        their layer payloads and refuse to guess. When given it joins the
        cache key, so the same file requested with different layer sets
        loads once per set — payloads are small and correctness beats
        dedup here; layer-carrying formats (gguf, ReFT dirs) are normally
        requested without it and keep full dedup.

        With lazy=True (request admission), a one-time warning recommends
        preloading. Raises on load failure.
        """
        layers_key = tuple(sorted(target_layers)) if target_layers else None
        key = (path, algorithm, layers_key, *file_version(path))
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            return entry

        # Drop stale versions of the same vector so they don't hold
        # capacity (and so the reload is visible in the logs).
        stale = [k for k in self._entries if k[0] == path and k[1] == algorithm]
        for k in stale:
            del self._entries[k]
        if stale:
            logger.info("Steer vector file changed on disk, reloading: %s", path)
        elif lazy and path not in self._warned_lazy:
            self._warned_lazy.add(path)
            logger.warning(
                "Steer vector %s was not preloaded; loading it now blocks "
                "request admission once. Use "
                "LLM.preload_steer_vectors([...]) to avoid this.",
                path,
            )

        from vllm.steer_vectors.models import LoadedSteerVector

        entry = LoadedSteerVector.from_local_checkpoint(
            steer_vector_model_path=path,
            steer_vector_id=0,
            config=self.steer_vector_config,
            device=self.device,
            scale_factor=1.0,
            algorithm=algorithm,
            target_layers=list(target_layers) if target_layers else None,
        )
        self._entries[key] = entry
        logger.info("Loaded steer vector into store: %s (%s)", path, algorithm)
        while len(self._entries) > self.capacity:
            evicted_key, _ = self._entries.popitem(last=False)
            logger.info("Evicted steer vector from store: %s", evicted_key[0])
        return entry

    def preload(self, path: str, algorithm: str = "direct") -> None:
        self.get(path, algorithm, lazy=False)

    def reload(self, path: str, algorithm: str = "direct") -> None:
        """Force-refresh a vector from disk regardless of stat changes."""
        self.unload(path)
        self.preload(path, algorithm)

    def unload(self, path: str) -> bool:
        removed = False
        for key in [k for k in self._entries if k[0] == path]:
            del self._entries[key]
            removed = True
        return removed

    def resident_paths(self) -> list[str]:
        return [k[0] for k in self._entries]
