# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Pinned Hugging Face revisions of the models every reported result used.

Every script loads a model through `pinned`, so a fresh download fetches the
exact weights behind the paper's numbers, not whatever `main` points to later.
"""

from __future__ import annotations

REVISIONS = {
    "google/gemma-3n-E4B-it": "c1221e9c62e34a43ab7ffacd1be0ea71f126ef10",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
}


def pinned(model_id: str) -> str | None:
    """Revision used in the paper for `model_id`, or None (latest) for others."""
    return REVISIONS.get(model_id)
