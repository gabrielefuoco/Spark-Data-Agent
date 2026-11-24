"""Tests for the deterministic synthetic eRISK sample generator."""

from __future__ import annotations

from pathlib import Path

from scripts.generate_dev_sample import generate_synthetic_sample, main, sample_from_corpus


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_same_seed_produces_identical_output(tmp_path: Path) -> None:
    out_a = generate_synthetic_sample(
        tmp_path / "run_a",
        seed=123,
        subjects=10,
        posts_per_subject=5,
        include_broken=True,
    )
    out_b = generate_synthetic_sample(
        tmp_path / "run_b",
        seed=123,
        subjects=10,
        posts_per_subject=5,
        include_broken=True,
    )

    assert _snapshot_tree(out_a) == _snapshot_tree(out_b)


def test_different_seeds_produce_different_output(tmp_path: Path) -> None:
    out_a = generate_synthetic_sample(tmp_path / "a", seed=1, subjects=6, posts_per_subject=4)
    out_b = generate_synthetic_sample(tmp_path / "b", seed=2, subjects=6, posts_per_subject=4)

    assert _snapshot_tree(out_a) != _snapshot_tree(out_b)


def test_cli_generates_balanced_truth_and_multi_year_posts(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_sample"
    main(
        [
            "--output-dir",
            str(out_dir),
            "--seed",
            "7",
            "--subjects",
            "12",
            "--posts-per-subject",
            "10",
            "--no-broken",
        ]
    )

    truth_lines = (out_dir / "risk-golden-truth-test.txt").read_text(encoding="utf-8").splitlines()
    labels = [int(line.split("\t")[1]) for line in truth_lines]
    assert len(truth_lines) == 12
    assert labels.count(0) == 6
    assert labels.count(1) == 6

    xml_files = sorted((out_dir / "chunk" / "chunk1").glob("*.xml"))
    assert len(xml_files) == 12
    years = {
        line.strip().removeprefix("<DATE>").removesuffix("</DATE>").strip()[:4]
        for xml in xml_files
        for line in xml.read_text(encoding="utf-8").splitlines()
        if "<DATE>" in line
    }
    assert len(years) >= 2


def test_from_corpus_mode_warns_about_gitignore(tmp_path: Path, capsys) -> None:
    fake_corpus = generate_synthetic_sample(
        tmp_path / "fake_corpus",
        seed=99,
        subjects=6,
        posts_per_subject=4,
        include_broken=False,
    )
    sampled = sample_from_corpus(
        fake_corpus,
        tmp_path / "sampled",
        seed=99,
        subjects=4,
        posts_per_subject=2,
    )

    captured = capsys.readouterr()
    assert "gitignored" in captured.err.lower()
    assert (sampled / "risk-golden-truth-test.txt").exists()
    assert len(list((sampled / "chunk" / "chunk1").glob("*.xml"))) == 4
