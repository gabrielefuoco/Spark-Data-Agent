"""Deterministic eRISK-shaped sample dataset generator for local development.

Generates a synthetic XML + TSV corpus matching the real eRISK directory
layout so tests and local workflows run quickly without touching real user
posts. Also supports stratified sampling from a local corpus directory.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

from src.lakehouse.paths import LakehousePaths

_CONTROL_VOCAB = (
    "morning coffee",
    "distributed systems",
    "weekend hike",
    "reading documentation",
    "team meeting",
    "cooking dinner",
    "python refactoring",
    "garden project",
    "sunny afternoon",
    "music practice",
)

_DEPRESSED_VOCAB = (
    "feel alone",
    "cant sleep",
    "exhausted again",
    "long night",
    "quiet room",
    "heavy thoughts",
    "sleepless hours",
    "no energy",
    "waking late",
    "empty morning",
)

_TITLE_TEMPLATES = (
    "Daily note on {topic}",
    "Thoughts about {topic}",
    "Quick update regarding {topic}",
    "   ",
)

_BODY_TEMPLATES = (
    "Spent some time thinking about {a} and {b} today.",
    "Another day dealing with {a} while trying to manage {b}.",
    "Looking back at {a}, it reminds me of {b} from earlier this year.",
    "Tried focusing on {a} before getting distracted by {b}.",
)


def _format_writing(title: str, date_str: str, info: str, text: str) -> str:
    return (
        "  <WRITING>\n"
        f"    <TITLE> {escape(title)} </TITLE>\n"
        f"    <DATE> {escape(date_str)} </DATE>\n"
        f"    <INFO> {escape(info)} </INFO>\n"
        f"    <TEXT> {escape(text)} </TEXT>\n"
        "  </WRITING>"
    )


def _write_individual_xml(path: Path, writings: list[str]) -> None:
    body = "\n".join(writings)
    path.write_text(f"<INDIVIDUAL>\n{body}\n</INDIVIDUAL>\n", encoding="utf-8")


def _synthetic_timestamp(rng: random.Random, label: int) -> str:
    base = datetime(2014, 1, 1, 0, 0, 0)
    day_offset = rng.randint(0, 365 * 3 - 1)
    # Depressed cohort posts more often in circadian night window (00:00-04:59).
    hour = rng.randint(0, 4) if label == 1 and rng.random() < 0.45 else rng.randint(8, 22)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    dt = base + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def generate_synthetic_sample(
    output_dir: str | Path,
    *,
    seed: int = 42,
    subjects: int = 100,
    posts_per_subject: int = 50,
    include_broken: bool = True,
) -> Path:
    """Create a deterministic synthetic eRISK-shaped dataset in ``output_dir``."""
    if subjects < 2:
        raise ValueError("subjects must be at least 2 to balance both classes.")
    if posts_per_subject < 1:
        raise ValueError("posts_per_subject must be at least 1.")

    rng = random.Random(seed)
    out = Path(output_dir)
    chunk_dir = out / "chunk" / "chunk1"
    if (out / "chunk").exists():
        shutil.rmtree(out / "chunk")
    chunk_dir.mkdir(parents=True, exist_ok=True)

    truth_lines: list[str] = []
    half = subjects // 2
    labels = [0] * half + [1] * (subjects - half)
    rng.shuffle(labels)

    for idx, label in enumerate(labels, start=1):
        subject_id = f"subject{idx:03d}"
        truth_lines.append(f"{subject_id}\t{label}")
        vocab = _DEPRESSED_VOCAB if label == 1 else _CONTROL_VOCAB

        writings: list[str] = []
        for _ in range(posts_per_subject):
            topic_a, topic_b = rng.sample(vocab, 2)
            title = rng.choice(_TITLE_TEMPLATES).format(topic=topic_a)
            text = rng.choice(_BODY_TEMPLATES).format(a=topic_a, b=topic_b)
            date_str = _synthetic_timestamp(rng, label)
            writings.append(_format_writing(title, date_str, "reddit post", text))

        if include_broken and idx == 1:
            writings.append(
                _format_writing(
                    "Broken date", "not-a-valid-date", "reddit post", "Valid text with bad date."
                )
            )
        if include_broken and idx == 2:
            writings.append(
                _format_writing(
                    "Blank body", _synthetic_timestamp(rng, label), "reddit post", "   "
                )
            )

        _write_individual_xml(chunk_dir / f"{subject_id}_1.xml", writings)

    if include_broken:
        _write_individual_xml(
            chunk_dir / "subject99999_1.xml",
            [
                _format_writing(
                    "Orphan", "2015-06-15 12:00:00", "reddit post", "Orphan subject post."
                )
            ],
        )
        _write_individual_xml(
            chunk_dir / "unknown_1.xml",
            [
                _format_writing(
                    "Missing id", "2015-06-16 14:30:00", "reddit post", "Post without subject id."
                )
            ],
        )

    truth_path = out / "risk-golden-truth-test.txt"
    truth_path.write_text("\n".join(truth_lines) + "\n", encoding="utf-8")
    return out


def sample_from_corpus(
    corpus_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    subjects: int = 100,
    posts_per_subject: int = 50,
) -> Path:
    """Sample balanced subjects from a local eRISK corpus directory.

    Intended strictly for local, gitignored development runs; never commit the
    resulting files.
    """
    print(
        "WARNING: --from-corpus copies real eRISK posts. Output is gitignored "
        "and MUST NEVER be committed or added to tests.",
        file=sys.stderr,
    )
    rng = random.Random(seed)
    corpus = Path(corpus_dir)
    truth_file = corpus / "risk-golden-truth-test.txt"
    if not truth_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {truth_file}")

    by_label: dict[int, list[str]] = {0: [], 1: []}
    for line in truth_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        subj, lbl = line.strip().split("\t")
        label_int = int(lbl)
        if label_int in by_label:
            by_label[label_int].append(subj)

    per_class = subjects // 2
    selected_0 = rng.sample(sorted(by_label[0]), min(per_class, len(by_label[0])))
    selected_1 = rng.sample(sorted(by_label[1]), min(subjects - len(selected_0), len(by_label[1])))
    selected = [(s, 0) for s in selected_0] + [(s, 1) for s in selected_1]
    selected_map = dict(selected)

    out = Path(output_dir)
    chunk_dir = out / "chunk" / "chunk1"
    if (out / "chunk").exists():
        shutil.rmtree(out / "chunk")
    chunk_dir.mkdir(parents=True, exist_ok=True)

    writings_by_subject: dict[str, list[ET.Element]] = {s: [] for s in selected_map}
    for xml_path in sorted((corpus / "chunk").glob("chunk*/*.xml")):
        subj_id = xml_path.stem.split("_")[0]
        if subj_id not in selected_map:
            continue
        root = ET.parse(xml_path).getroot()
        writings_by_subject[subj_id].extend(root.findall("WRITING"))

    for subj_id, writings in writings_by_subject.items():
        if not writings:
            continue
        chosen = writings[:posts_per_subject]
        root = ET.Element("INDIVIDUAL")
        for elem in chosen:
            root.append(elem)
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(
            chunk_dir / f"{subj_id}_1.xml",
            encoding="utf-8",
            xml_declaration=False,
        )

    truth_lines = [f"{s}\t{lbl}" for s, lbl in selected if writings_by_subject.get(s)]
    (out / "risk-golden-truth-test.txt").write_text("\n".join(truth_lines) + "\n", encoding="utf-8")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_out = LakehousePaths.from_settings().sample_root
    parser = argparse.ArgumentParser(description="Generate an eRISK-shaped dev sample.")
    parser.add_argument("--output-dir", default=default_out, help="Target sample directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--subjects", type=int, default=100, help="Total balanced subjects.")
    parser.add_argument(
        "--posts-per-subject",
        type=int,
        default=50,
        help="Target posts generated or sampled per subject.",
    )
    parser.add_argument(
        "--no-broken",
        action="store_true",
        help="Omit deliberately malformed rows in synthetic mode.",
    )
    parser.add_argument(
        "--from-corpus",
        type=Path,
        default=None,
        help="Optional local eRISK corpus root to sample from instead of synthetic generation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.from_corpus is not None:
        sample_from_corpus(
            args.from_corpus,
            args.output_dir,
            seed=args.seed,
            subjects=args.subjects,
            posts_per_subject=args.posts_per_subject,
        )
    else:
        generate_synthetic_sample(
            args.output_dir,
            seed=args.seed,
            subjects=args.subjects,
            posts_per_subject=args.posts_per_subject,
            include_broken=not args.no_broken,
        )


if __name__ == "__main__":
    main()
