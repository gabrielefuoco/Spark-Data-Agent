"""Settings-derived lakehouse and sample storage paths.

Deriving all Bronze, Silver, Gold, and sample locations from ``Settings`` keeps
local paths and future ``s3a://`` URIs consistent without hardcoded literals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import Settings, get_settings


def join_storage_path(base: str | Path, *parts: str) -> str:
    """Join subpaths onto a URI (``scheme://...``) or local filesystem path."""
    raw = str(base).strip()
    cleaned_parts = [part.strip("/") for part in parts if part and part.strip("/")]

    if "://" in raw:
        cleaned = raw.rstrip("/")
        return f"{cleaned}/{'/'.join(cleaned_parts)}" if cleaned_parts else cleaned

    target = Path(raw)
    for part in cleaned_parts:
        target = target.joinpath(*part.split("/"))
    return str(target)


@dataclass(frozen=True)
class LakehousePaths:
    """Resolved storage paths for every Medallion layer and sample directory."""

    storage_root: str

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LakehousePaths:
        cfg = settings or get_settings()
        return cls(storage_root=str(cfg.lakehouse_root))

    @property
    def bronze_posts(self) -> str:
        return join_storage_path(self.storage_root, "lakehouse", "bronze", "posts")

    @property
    def bronze_subjects(self) -> str:
        return join_storage_path(self.storage_root, "lakehouse", "bronze", "subjects")

    @property
    def silver_posts(self) -> str:
        return join_storage_path(self.storage_root, "lakehouse", "silver", "posts")

    @property
    def silver_quarantine(self) -> str:
        return join_storage_path(self.storage_root, "lakehouse", "silver", "quarantine")

    @property
    def gold_user_longitudinal_features(self) -> str:
        return join_storage_path(
            self.storage_root, "lakehouse", "gold", "gold_user_longitudinal_features"
        )

    @property
    def gold_cohort_lexical_matrix(self) -> str:
        return join_storage_path(
            self.storage_root, "lakehouse", "gold", "gold_cohort_lexical_matrix"
        )

    @property
    def gold_cohort_topic_matrix(self) -> str:
        return join_storage_path(self.storage_root, "lakehouse", "gold", "gold_cohort_topic_matrix")

    @property
    def gold_dataset_manifest(self) -> str:
        return join_storage_path(
            self.storage_root, "lakehouse", "gold", "gold_dataset_manifest.json"
        )

    @property
    def sample_root(self) -> str:
        return join_storage_path(self.storage_root, "sample")

    @property
    def sample_chunks(self) -> str:
        return join_storage_path(self.sample_root, "chunk")

    @property
    def sample_truth(self) -> str:
        return join_storage_path(self.sample_root, "risk-golden-truth-test.txt")
