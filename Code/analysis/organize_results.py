from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path

from analysis_config import PROJECT_ROOT, RESULTS_DIR


SOURCE_DIR = PROJECT_ROOT / "Code" / "analysis" / "results"
ARCHIVE_DIR = RESULTS_DIR / "_archive" / "pre_cleanup_root_results"


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def plan_moves() -> list[tuple[Path, Path, str]]:
    """Return (source, destination, action) tuples without changing files."""
    if not SOURCE_DIR.exists():
        return []

    moves: list[tuple[Path, Path, str]] = []
    for source in iter_files(SOURCE_DIR):
        relative = source.relative_to(SOURCE_DIR)
        destination = RESULTS_DIR / relative

        if not destination.exists():
            moves.append((source, destination, "move"))
        elif filecmp.cmp(source, destination, shallow=False):
            moves.append((source, destination, "remove_duplicate"))
        else:
            moves.append((source, destination, "replace_root_with_source"))

    return moves


def print_plan(moves: list[tuple[Path, Path, str]]) -> None:
    if not moves:
        print("No files need migration.")
        return

    for source, destination, action in moves:
        print(f"{action:22s} {source} -> {destination}")


def apply_moves(moves: list[tuple[Path, Path, str]]) -> None:
    for source, destination, action in moves:
        if action == "remove_duplicate":
            source.unlink()
            continue

        if action == "replace_root_with_source":
            relative = source.relative_to(SOURCE_DIR)
            archive_destination = ARCHIVE_DIR / relative
            archive_destination.parent.mkdir(parents=True, exist_ok=True)

            if archive_destination.exists():
                if filecmp.cmp(destination, archive_destination, shallow=False):
                    destination.unlink()
                else:
                    raise FileExistsError(
                        "Archive conflict: a different file already exists at "
                        f"{archive_destination}"
                    )
            else:
                shutil.move(str(destination), str(archive_destination))

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

    # Remove now-empty source directories from the bottom up.
    for directory in sorted(
        (p for p in SOURCE_DIR.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass

    try:
        SOURCE_DIR.rmdir()
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Safely consolidate legacy Code/analysis/results into the canonical "
            "repository-root results directory. Defaults to a dry run."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the migration. Without this flag, only show the plan.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("RESULT DIRECTORY MIGRATION")
    print("=" * 72)
    print(f"Source : {SOURCE_DIR}")
    print(f"Target : {RESULTS_DIR}")
    print()

    moves = plan_moves()
    print_plan(moves)

    if not args.apply:
        print("\nDry run only. No files were changed.")
        print("Run with --apply after reviewing the plan.")
        return

    apply_moves(moves)
    print("\nMigration complete.")
    print(f"Canonical results root: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
