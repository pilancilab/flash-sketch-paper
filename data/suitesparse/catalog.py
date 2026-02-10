from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass

import globals as g


@dataclass(frozen=True)
class SuiteSparseEntry:
    """Catalog entry for a SuiteSparse matrix."""

    group: str
    name: str


def get_default_catalog():
    """Return a curated list of SuiteSparse matrices for regression testing."""
    return [
        SuiteSparseEntry(group="HB", name="1138_bus"),
        SuiteSparseEntry(group="HB", name="bcsstk01"),
        SuiteSparseEntry(group="HB", name="bcsstk04"),
        SuiteSparseEntry(group="HB", name="west0479"),
        SuiteSparseEntry(group="HB", name="steam1"),
    ]


def dataset_id(entry):
    """Return a stable dataset id string for a catalog entry."""
    return f"{g.DATASET_SUITESPARSE}:{entry.group}/{entry.name}"
