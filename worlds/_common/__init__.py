"""Shared generation + verification *drivers* for foundry worlds (VLMH9).

Unlike ``worlds._scene`` / ``worlds._measure`` (which must stay disjoint),
these drivers are ordinary *consumers*: a world's ``generate.py`` / ``verify.py``
already imports both its renderer and its oracle, and these helpers just hoist the
identical boilerplate (argument parsing, the paired scene x family loop, oracle
certification, manifest writing, re-certification) out of every world. The
per-world renderer, oracle, and prompts remain fully bespoke.

A world opts in by exposing the standard interface (see ``make_generate_main``)
and writing a three-line ``generate.py`` / ``verify.py``. Worlds built before this
library keep their own copies; behaviour is byte-identical.
"""

from .generate import make_generate_main
from .verify import make_verify_main

__all__ = ["make_generate_main", "make_verify_main"]
