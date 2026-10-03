from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

MANIFEST_NAME = "manifest.csv"
MANIFEST_FIELDS = ["source", "destination", "category", "status", "reason"]
# "copyd" is a legacy misspelling of "copied" found in old manifests.
DONE_STATUSES = {"moved", "copied", "copyd", "skipped_missing"}


class ManifestWriter:
    def __init__(self, manifest_path: Path, flush_every: int = 100) -> None:
        self.manifest_path = manifest_path
        self.flush_every = flush_every
        self.count = 0
        self.file = None
        self.writer = None

    def __enter__(self) -> "ManifestWriter":
        is_new = not self.manifest_path.exists()
        self.file = self.manifest_path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=MANIFEST_FIELDS)
        if is_new:
            self.writer.writeheader()
        return self

    def __exit__(self, exc_type, exc, traceback_obj) -> None:
        if self.file:
            self.file.flush()
            self.file.close()

    def append(
        self,
        source: Path,
        destination: Path | None,
        category: str,
        status: str,
        reason: str = "",
    ) -> None:
        if not self.writer or not self.file:
            raise RuntimeError("ManifestWriter is not open")

        self.writer.writerow(
            {
                "source": str(source),
                "destination": str(destination) if destination else "",
                "category": category,
                "status": status,
                "reason": reason,
            }
        )
        self.count += 1
        if self.count % self.flush_every == 0:
            self.file.flush()


def read_done_manifest(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()

    done: set[str] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            source = row.get("source")
            if source and row.get("status") in DONE_STATUSES:
                done.add(source)
    return done


def read_done_manifests(manifest_paths: Iterable[Path]) -> set[str]:
    done: set[str] = set()
    for manifest_path in manifest_paths:
        done.update(read_done_manifest(manifest_path))
    return done
