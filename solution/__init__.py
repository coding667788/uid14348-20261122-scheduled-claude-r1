"""Offline tenant-isolation control-plane model.

The package exposes three commands through ``python -m solution``:

* ``replay`` -- execute a supplied event timeline under a supplied plan and
  report every barrier witness, the earliest violation prefix, and the fate of
  each in-flight marker.
* ``plan``   -- search for a fixed release table that is safe for every
  permitted link-latency assignment and for the (at most one) single-fault
  domain scenario, while satisfying reachability.
* ``check``  -- structural validation of a case file (used by the samples and
  by tests).

Everything is deterministic: no wall clock, no randomness, no I/O beyond the
case file named on the command line and the JSON document written to stdout.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
