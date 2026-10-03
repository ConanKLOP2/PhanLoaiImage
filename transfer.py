from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TransferJob:
    source: Path
    destination: Path
    category: str
    status: str
    reason: str
    detection_error: bool = False


@dataclass(frozen=True)
class TransferOutcome:
    job: TransferJob
    ok: bool
    error: str = ""


class DestinationAllocator:
    """Reserve collision-free destination names.

    Each destination directory is listed once and cached, so allocating a name
    costs a set lookup instead of a filesystem stat per file. Names are compared
    case-insensitively, as on Windows. Only one thread should allocate.
    """

    def __init__(self) -> None:
        self._taken: dict[Path, set[str]] = {}

    def _names(self, directory: Path) -> set[str]:
        names = self._taken.get(directory)
        if names is None:
            try:
                names = {entry.name.casefold() for entry in os.scandir(directory)}
            except FileNotFoundError:
                names = set()
            self._taken[directory] = names
        return names

    def reserve(self, directory: Path, source: Path) -> Path:
        names = self._names(directory)
        candidate = source.name
        index = 0
        while candidate.casefold() in names:
            index += 1
            candidate = f"{source.stem}_{index}{source.suffix}"
        names.add(candidate.casefold())
        return directory / candidate


def transfer_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    try:
        same_drive = os.path.splitdrive(os.fspath(source))[0].lower() == os.path.splitdrive(
            os.fspath(destination)
        )[0].lower()
        if mode == "move" and same_drive:
            os.replace(source, destination)
            return

        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        if mode == "move":
            source.unlink()
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def transfer_status(mode: str) -> str:
    return "copied" if mode == "copy" else "moved"


def run_transfer(job: TransferJob, mode: str) -> TransferOutcome:
    try:
        transfer_file(job.source, job.destination, mode)
        return TransferOutcome(job=job, ok=True)
    except Exception as exc:
        return TransferOutcome(job=job, ok=False, error=repr(exc))
