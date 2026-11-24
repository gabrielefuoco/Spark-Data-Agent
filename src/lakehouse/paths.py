"""Settings-derived lakehouse and sample storage paths.

Deriving all Bronze, Silver, Gold, and sample locations from ``Settings`` keeps
local paths and future ``s3a://`` URIs consistent without hardcoded literals.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings, get_settings


def _join_uri(base: str, *parts: str) -> str:
    cleaned = base.rstrip("/")
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return f"{cleaned}/{suffix}" if suffix else cleaned


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
        return _join_uri(self.storage_root, "lakehouse", "bronze", "posts")

    @property
    def bronze_subjects(self) -> str:
        return _join_uri(self.storage_root, "lakehouse", "bronze", "subjects")

    @property
    def silver_posts(self) -> str:
        return _join_uri(self.storage_root, "lakehouse", "silver", "posts")

    @property
    def silver_quarantine(self) -> str:
        return _join_uri(self.storage_root, "lakehouse", "silver", "quarantine")

    @property
    def gold_user_longitudinal_features(self) -> str:
        return _join_uri(
            self.storage_root, "lakehouse", "gold", "gold_user_longitudinal_features"
        )

    @property
    def gold_cohort_lexical_matrix(self) -> str:
        return _join_uri(
            self.storage_root, "lakehouse", "gold", "gold_cohort_lexical_matrix"
        )

    @property
    def gold_dataset_manifest(self) -> str:
        return _join_uri(
            self.storage_root, "lakehouse", "gold", "gold_dataset_manifest.json"
        )

    @property
    def sample_root(self) -> str:
        return _join_uri(self.storage_root, "sample")

    @property
    def sample_chunks(self) -> str:
        return _join_uri(self.sample_root, "chunk")

    @property
    def sample_truth(self) -> str:
        return _join_uri(self.sample_root, "risk-golden-truth-test.txt")
