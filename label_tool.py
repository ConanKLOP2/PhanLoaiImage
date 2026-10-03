"""Label sampled images by hand (the model's prediction is hidden on purpose).

    python label_tool.py --sample sample.csv --labels labels.csv

Keys: 1 = nude (explicit), 2 = sexy (suggestive, nothing explicit),
3 = normal, Space = skip, Backspace/u = undo. Progress is saved after every
label, so you can stop and resume at any time.
"""
from __future__ import annotations

import argparse
import base64
import csv
import sys
from pathlib import Path

from accuracy import CLASSES, read_labels, read_sample

KEY_TO_LABEL = {"1": "nude", "2": "sexy", "3": "normal"}


class LabelSession:
    """Order, undo and persistence for labelling, independent of any GUI."""

    def __init__(self, paths: list[str], labels_path: Path) -> None:
        self.paths = list(paths)
        self.labels_path = Path(labels_path)
        self.labels: dict[str, str] = (
            read_labels(self.labels_path) if self.labels_path.exists() else {}
        )
        self.skipped: set[str] = set()
        self.history: list[tuple[str, str | None]] = []  # (path, label or None=skip)

    @property
    def remaining(self) -> list[str]:
        return [p for p in self.paths if p not in self.labels and p not in self.skipped]

    @property
    def current(self) -> str | None:
        remaining = self.remaining
        return remaining[0] if remaining else None

    @property
    def labelled_count(self) -> int:
        return sum(1 for p in self.paths if p in self.labels)

    def set_label(self, label: str) -> None:
        if label not in CLASSES:
            raise ValueError(f"Nhan khong hop le: {label}")
        path = self.current
        if path is None:
            return
        self.labels[path] = label
        self.history.append((path, label))
        self._save()

    def skip(self) -> None:
        path = self.current
        if path is not None:
            self.skipped.add(path)
            self.history.append((path, None))

    def undo(self) -> str | None:
        """Revert the last action and return the path that is current again."""
        if not self.history:
            return self.current
        path, label = self.history.pop()
        if label is None:
            self.skipped.discard(path)
        else:
            self.labels.pop(path, None)
            self._save()
        return path

    def counts(self) -> dict[str, int]:
        return {name: sum(1 for p in self.paths if self.labels.get(p) == name) for name in CLASSES}

    def _save(self) -> None:
        temporary = self.labels_path.with_suffix(self.labels_path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["path", "label"])
            for path in self.paths:
                if path in self.labels:
                    writer.writerow([path, self.labels[path]])
        temporary.replace(self.labels_path)


def render_png(path: str, max_width: int = 900, max_height: int = 700) -> bytes:
    """Decode (reduced for big JPEGs), fit into the box and return PNG bytes."""
    import cv2
    import numpy as np

    from fast_onnx_detector import jpeg_dimensions

    data = np.fromfile(path, dtype=np.uint8)
    image = None
    dimensions = jpeg_dimensions(data)
    if dimensions:
        width, height = dimensions
        flags = {
            2: cv2.IMREAD_REDUCED_COLOR_2,
            4: cv2.IMREAD_REDUCED_COLOR_4,
            8: cv2.IMREAD_REDUCED_COLOR_8,
        }
        for factor in (8, 4, 2):
            if width // factor >= max_width or height // factor >= max_height:
                image = cv2.imdecode(data, flags[factor] | cv2.IMREAD_IGNORE_ORIENTATION)
                break
    if image is None:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Khong doc duoc anh: {path}")
    scale = min(max_width / image.shape[1], max_height / image.shape[0], 1.0)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Khong ma hoa duoc anh: {path}")
    return buffer.tobytes()


def run_gui(session: LabelSession) -> None:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Label images")
    root.geometry("960x860")

    picture = ttk.Label(root, anchor="center")
    picture.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
    info = tk.StringVar()
    ttk.Label(root, textvariable=info).pack(pady=(0, 4))
    buttons = ttk.Frame(root)
    buttons.pack(pady=(0, 10))

    state = {"photo": None}

    def show() -> None:
        path = session.current
        counts = session.counts()
        summary = (
            f"{session.labelled_count}/{len(session.paths)} da gan nhan  "
            f"(nude {counts['nude']}, sexy {counts['sexy']}, normal {counts['normal']})"
        )
        if path is None:
            picture.configure(image="", text="Xong! Chay evaluate.py de xem ket qua.")
            info.set(summary)
            return
        try:
            state["photo"] = tk.PhotoImage(data=base64.b64encode(render_png(path)))
            picture.configure(image=state["photo"], text="")
        except Exception as exc:
            picture.configure(image="", text=f"Loi doc anh: {exc}")
        info.set(f"{summary}   |   {Path(path).name}")

    def label(name: str) -> None:
        session.set_label(name)
        show()

    def skip() -> None:
        session.skip()
        show()

    def undo() -> None:
        session.undo()
        show()

    for key, name in KEY_TO_LABEL.items():
        ttk.Button(buttons, text=f"{key} - {name}", command=lambda n=name: label(n)).pack(
            side=tk.LEFT, padx=4
        )
        root.bind(key, lambda _event, n=name: label(n))
    ttk.Button(buttons, text="Space - skip", command=skip).pack(side=tk.LEFT, padx=4)
    ttk.Button(buttons, text="u - undo", command=undo).pack(side=tk.LEFT, padx=4)
    root.bind("<space>", lambda _event: skip())
    root.bind("u", lambda _event: undo())
    root.bind("<BackSpace>", lambda _event: undo())

    show()
    root.mainloop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("labels.csv"))
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    sample = read_sample(args.sample)
    if not sample:
        raise SystemExit("File sample rong.")
    run_gui(LabelSession([row.path for row in sample], args.labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
