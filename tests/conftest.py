"""Shared Pytest fixtures for Spark and synthetic eRISK lakehouse tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import BaseMessage
from pydantic import SecretStr
from pyspark.sql import DataFrame, SparkSession
from scripts.generate_dev_sample import generate_synthetic_sample
from src.config import Settings
from src.lakehouse.bronze import run_bronze_ingestion
from src.lakehouse.paths import LakehousePaths
from src.spark import build_spark_session


class RecordingFakeChatModel(FakeMessagesListChatModel):
    """Fake tool-calling chat model capturing invocations and bound tools."""

    recorded_invocations: list[list[BaseMessage]] = []
    bound_tools: list[Any] = []

    def __init__(self, responses: list[BaseMessage]) -> None:
        super().__init__(responses=responses)
        self.recorded_invocations = []
        self.bound_tools = []

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> RecordingFakeChatModel:
        self.bound_tools = list(tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.recorded_invocations.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def make_s3_settings(
    uri: str,
    *,
    endpoint: str | None = "http://minio:9000",
    access_key: str | None = "minioadmin",
    secret_key: str | None = "minioadmin",
    **extra: Any,
) -> Settings:
    """Build ``Settings`` for S3A/MinIO tests without touching ``.env``."""
    return Settings(
        _env_file=None,
        spark_storage_uri=uri,
        aws_access_key_id=SecretStr(access_key) if access_key is not None else None,
        aws_secret_access_key=SecretStr(secret_key) if secret_key is not None else None,
        s3_endpoint_url=endpoint,
        **extra,
    )


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """Session-scoped local SparkSession with 2 cores and 2 shuffle partitions."""
    settings = Settings(
        _env_file=None,
        spark_app_name="spark-data-agent-tests",
        spark_master="local[2]",
    )
    session = build_spark_session(settings, shuffle_partitions=2)
    yield session
    session.stop()


@pytest.fixture
def lakehouse_env(tmp_path: Path) -> tuple[Settings, LakehousePaths, Path]:
    """Isolated Settings, LakehousePaths, and small synthetic corpus under tmp_path."""
    storage_root = tmp_path / "data"
    settings = Settings(
        _env_file=None,
        spark_storage_uri=str(storage_root),
        spark_app_name="spark-data-agent-tests",
        spark_master="local[2]",
    )
    paths = LakehousePaths.from_settings(settings)
    sample_dir = generate_synthetic_sample(
        paths.sample_root,
        seed=42,
        subjects=10,
        posts_per_subject=8,
        include_broken=True,
    )
    return settings, paths, sample_dir


@pytest.fixture
def bronze_lakehouse(
    spark: SparkSession,
    lakehouse_env: tuple[Settings, LakehousePaths, Path],
) -> tuple[Settings, LakehousePaths, DataFrame, DataFrame]:
    """Ingest the synthetic sample into Bronze Parquet and return ``(settings, paths, posts, subjects)``."""
    settings, paths, _ = lakehouse_env
    posts_df, subjects_df = run_bronze_ingestion(
        spark,
        paths.sample_chunks,
        paths.sample_truth,
        settings=settings,
        paths=paths,
    )
    return settings, paths, posts_df, subjects_df


@pytest.fixture
def forbid_active_spark(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    """Return a helper that fails any ``SparkSession.getActiveSession()`` lookup."""

    def _activate() -> None:
        def _forbid(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("load_manifest must never invoke SparkSession")

        monkeypatch.setattr(SparkSession, "getActiveSession", staticmethod(_forbid))

    return _activate
