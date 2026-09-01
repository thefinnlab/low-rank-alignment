"""Move legacy Brain Treebank epoch provenance to stimulus-scoped filenames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def subjects_for_movie(data_root: Path, movie: str) -> list[int]:
    subjects: list[int] = []
    for path in sorted((data_root / "subject_metadata").glob("sub_*_metadata.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("filename") != movie:
            continue
        subject_token = path.name.split("_trial", maxsplit=1)[0]
        subjects.append(int(subject_token.removeprefix("sub_")))
    if not subjects:
        raise ValueError(f"No Brain Treebank subjects found for movie {movie!r}")
    return sorted(set(subjects))


def legacy_filename(rereference: str) -> str:
    return (
        "config.yml"
        if rereference == "common_average"
        else f"config_ref-{rereference}.yml"
    )


def scoped_filename(movie: str, rereference: str) -> str:
    suffix = "" if rereference == "common_average" else f"_ref-{rereference}"
    return f"config_movie-{movie}{suffix}.yml"


def validate_legacy_config(path: Path, movie: str, rereference: str) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if config.get("dataset") != "brain_treebank":
        raise ValueError(f"Legacy provenance is not for brain_treebank: {path}")
    per_dataset = config.get("per_dataset_config") or {}
    actual_movie = per_dataset.get("movie")
    actual_rereference = per_dataset.get("rereference", "common_average")
    if actual_movie != movie or actual_rereference != rereference:
        raise ValueError(
            f"Legacy provenance does not match requested movie/rereference at {path}: "
            f"{actual_movie!r}/{actual_rereference!r} != {movie!r}/{rereference!r}"
        )


def migrate(
    data_root: Path,
    movie: str,
    rereference: str,
    subjects: list[int],
    *,
    dry_run: bool,
) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    for subject in subjects:
        directory = data_root / "precomputed_epochs" / f"sub_{subject}"
        source = directory / legacy_filename(rereference)
        destination = directory / scoped_filename(movie, rereference)
        if not source.exists():
            raise FileNotFoundError(f"Legacy provenance file not found: {source}")
        if destination.exists():
            raise FileExistsError(f"Destination already exists: {destination}")
        validate_legacy_config(source, movie, rereference)
        moves.append((source, destination))

    for source, destination in moves:
        print(f"{'Would move' if dry_run else 'Moving'} {source} -> {destination}")
        if not dry_run:
            source.rename(destination)
    return moves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/brain-treebank"))
    parser.add_argument("--movie", required=True)
    parser.add_argument(
        "--rereference",
        choices=("common_average", "laplacian"),
        required=True,
    )
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subjects = args.subjects or subjects_for_movie(args.data_root, args.movie)
    migrate(
        args.data_root,
        args.movie,
        args.rereference,
        subjects,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
