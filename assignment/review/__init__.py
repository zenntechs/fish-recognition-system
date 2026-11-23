"""Human-review workflow for the fish dataset.

All review state lives under ``assignment/data/`` — nothing is ever written
to ``SAMPLES FISH/``.
"""

from assignment.data.review import ReviewManifest, ReviewDecision  # noqa: F401
