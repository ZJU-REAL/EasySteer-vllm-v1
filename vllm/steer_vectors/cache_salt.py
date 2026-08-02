# SPDX-License-Identifier: Apache-2.0
"""Engine-global prefix-cache salt for server-level steering.

When server-level steering is configured, every request's KV is steered
by the server config without carrying a per-request steering config, so
block hashes would not reflect it. The engine core sets this salt (the
server config's fingerprint, computed at startup) before any request is
hashed; `generate_block_hash_extra_keys` mixes it into every block hash.

The salt is static per engine boot. Runtime scale updates keep the
cache correct via `reset_prefix_cache` (the scale-update endpoint calls
it), matching the upstream pattern for RLHF-style weight updates; fresh
runtime installs of server steering on prefix-caching engines are
rejected because pre-install blocks were hashed without any salt.
"""

_server_steer_salt: str | None = None


def set_server_steer_salt(salt: str | None) -> None:
    global _server_steer_salt
    _server_steer_salt = salt


def get_server_steer_salt() -> str | None:
    return _server_steer_salt
