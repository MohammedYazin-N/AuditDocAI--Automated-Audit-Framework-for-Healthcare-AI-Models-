"""Shared building blocks: audit statuses and the Finding record.

Statuses follow the project brief exactly:
  Passed        - the defined check ran and met its documented criterion.
  Warning       - a potential concern or limitation was identified.
  Failed        - the check detected an issue per its documented criterion.
  Not Evaluated - required information or functionality was unavailable.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field

PASSED, WARNING, FAILED, NOT_EVALUATED = "Passed", "Warning", "Failed", "Not Evaluated"
STATUSES = (PASSED, WARNING, FAILED, NOT_EVALUATED)
STATUS_COLOURS = {PASSED: "#1B873F", WARNING: "#C77700", FAILED: "#C0392B", NOT_EVALUATED: "#6B7785"}

# Where an implementation came from, so the report can state what was reused vs added.
FROM_NOTEBOOK = "Reused/generalised from notebook"
NEW = "New (added for the app)"


@dataclass
class Finding:
    """One audit check outcome."""
    area: str            # e.g. "Data quality", "Leakage"
    check: str           # short name
    status: str          # one of STATUSES
    detail: str          # what was found, with numbers
    criterion: str = ""  # the documented rule that produced the status
    action: str = ""     # recommended corrective action
    origin: str = NEW
    category: str = ""   # leakage only: Confirmed / Potential / Suspicious / Cannot be performed

    def as_dict(self) -> dict:
        return asdict(self)


def worst_status(findings) -> str:
    """Overall status of a group: Failed > Warning > Passed; Not Evaluated if nothing ran."""
    s = {f.status for f in findings}
    for st in (FAILED, WARNING, PASSED):
        if st in s:
            return st
    return NOT_EVALUATED


def counts(findings) -> dict:
    out = {s: 0 for s in STATUSES}
    for f in findings:
        out[f.status] += 1
    return out
