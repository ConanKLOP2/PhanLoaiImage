from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from classify_images import OUTPUT_DIR_NAME, scan_and_classify

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    DND_FILES = None
    TkinterDnD = None


BaseTk = TkinterDnD.Tk if TkinterDnD else tk.Tk


class ImageClassifierApp(BaseTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Image Classifier - nude / sexy / normal")
        self.geometry("1020x660")
        self.minsize(940, 600)

        self.folders: list[Path] = []

        self.mode_var = tk.StringVar(value="copy")
        self.output_strategy_var = tk.StringVar(value="root")
        self.device_var = tk.StringVar(value="gpu")
        self.engine_var = tk.StringVar(value="onnx")
        self.batch_size_var = tk.IntVar(value=256)
        self.preprocess_workers_var = tk.IntVar(value=6)
        self.transfer_workers_var = tk.IntVar(value=0)
        self.nude_threshold_var = tk.DoubleVar(value=0.8)
        self.sexy_threshold_var = tk.DoubleVar(value=0.8)
        self.status_var = tk.StringVar(value="Add one or more folders to begin.")
        self.detail_var = tk.StringVar(value="")
        self.progress_var = tk.DoubleVar(value=0)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None

        self._build_ui()
        self.after(150, self._poll_events)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        folder_frame = ttk.LabelFrame(root, text="Folders", padding=10)
        folder_frame.grid(row=0, column=0, sticky="nsew")
        folder_frame.columnconfigure(0, weight=1)
        folder_frame.rowconfigure(0, weight=1)

        self.folder_list = tk.Listbox(folder_frame, height=7, selectmode=tk.EXTENDED)
        self.folder_list.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        if DND_FILES:
            self.folder_list.drop_target_register(DND_FILES)
            self.folder_list.dnd_bind("<<Drop>>", self._drop_folders)

        folder_buttons = ttk.Frame(folder_frame)
        folder_buttons.grid(row=0, column=1, sticky="ns")
        ttk.Button(folder_buttons, text="Add folders...", command=self._choose_folders).pack(
            fill=tk.X, pady=(0, 6)
        )
        ttk.Button(folder_buttons, text="Remove selected", command=self._remove_selected).pack(
            fill=tk.X, pady=(0, 6)
        )
        ttk.Button(folder_buttons, text="Clear", command=self._clear_folders).pack(
            fill=tk.X
        )

        options = ttk.Frame(root)
        options.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        options.columnconfigure(0, weight=1)

        output_frame = ttk.LabelFrame(options, text="Output", padding=10)
        output_frame.grid(row=0, column=0, sticky="ew")
        output_frame.columnconfigure(0, weight=1)
        output_frame.columnconfigure(1, weight=1)
        output_frame.columnconfigure(2, weight=1)
        output_frame.columnconfigure(3, weight=1)
        ttk.Radiobutton(
            output_frame,
            text="Copy (keep originals)",
            value="copy",
            variable=self.mode_var,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            output_frame,
            text="Move (save disk space)",
            value="move",
            variable=self.mode_var,
        ).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Radiobutton(
            output_frame,
            text=f"Root {OUTPUT_DIR_NAME}",
            value="root",
            variable=self.output_strategy_var,
        ).grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Radiobutton(
            output_frame,
            text=f"Per-folder {OUTPUT_DIR_NAME}",
            value="per-folder",
            variable=self.output_strategy_var,
        ).grid(row=0, column=3, sticky="w", padx=(12, 0))

        performance = ttk.LabelFrame(options, text="Performance", padding=10)
        performance.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for index in range(8):
            performance.columnconfigure(index, weight=1)
        self._field(performance, "Device", self.device_var, ("gpu", "auto", "cpu"), 0, 0)
        self._field(performance, "Engine", self.engine_var, ("onnx", "nudenet"), 0, 2)
        self._spin(performance, "Batch", self.batch_size_var, 1, 1024, 0, 4)
        self._spin(performance, "Preprocess", self.preprocess_workers_var, 1, 32, 0, 6)
        self._spin(performance, "Transfer (0=Auto)", self.transfer_workers_var, 0, 16, 1, 0)

        preset_frame = ttk.Frame(performance)
        preset_frame.grid(row=1, column=2, columnspan=6, sticky="e", padx=(12, 0), pady=(8, 0))
        ttk.Button(preset_frame, text="HDD", command=lambda: self._apply_preset("hdd")).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(preset_frame, text="SSD", command=lambda: self._apply_preset("ssd")).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(preset_frame, text="CPU", command=lambda: self._apply_preset("cpu")).pack(
            side=tk.LEFT
        )

        thresholds = ttk.LabelFrame(root, text="Thresholds", padding=10)
        thresholds.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        thresholds.columnconfigure(5, weight=1)
        self._float_spin(thresholds, "Nude", self.nude_threshold_var, 0.1, 0.95, 0, 0)
        self._float_spin(thresholds, "Sexy", self.sexy_threshold_var, 0.1, 0.95, 0, 2)
        ttk.Label(
            thresholds,
            text="Lower = catches more images. Higher = fewer false positives.",
        ).grid(row=0, column=4, sticky="w", padx=(20, 0))

        status_frame = ttk.Frame(root)
        status_frame.grid(row=3, column=0, sticky="ew", pady=(16, 0))
        status_frame.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(
            status_frame, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.progress.grid(row=0, column=0, sticky="ew")
        ttk.Label(status_frame, textvariable=self.status_var).grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Label(status_frame, textvariable=self.detail_var).grid(
            row=2, column=0, sticky="w", pady=(4, 0)
        )

        actions = ttk.Frame(root)
        actions.grid(row=4, column=0, sticky="ew", pady=(18, 0))
        actions.columnconfigure(0, weight=1)
        ttk.Button(actions, text="Exit", command=self.destroy).grid(row=0, column=1)
        self.start_button = ttk.Button(
            actions, text="Start classification", command=self._start
        )
        self.start_button.grid(row=0, column=2, padx=(8, 0))

    def _field(self, parent, label, variable, values, row, column) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w")
        ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            state="readonly",
            width=10,
        ).grid(row=row, column=column + 1, sticky="w", padx=(6, 18))

    def _spin(self, parent, label, variable, start, end, row, column) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w")
        ttk.Spinbox(
            parent,
            from_=start,
            to=end,
            textvariable=variable,
            width=8,
        ).grid(row=row, column=column + 1, sticky="w", padx=(6, 18))

    def _float_spin(self, parent, label, variable, start, end, row, column) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w")
        ttk.Spinbox(
            parent,
            from_=start,
            to=end,
            increment=0.05,
            textvariable=variable,
            width=8,
        ).grid(row=row, column=column + 1, sticky="w", padx=(6, 18))

    def _apply_preset(self, preset: str) -> None:
        if preset == "hdd":
            self.device_var.set("gpu")
            self.engine_var.set("onnx")
            self.batch_size_var.set(256)
            self.preprocess_workers_var.set(4)
            self.transfer_workers_var.set(1)
        elif preset == "ssd":
            self.device_var.set("gpu")
            self.engine_var.set("onnx")
            self.batch_size_var.set(256)
            self.preprocess_workers_var.set(8)
            self.transfer_workers_var.set(1)
        else:
            self.device_var.set("cpu")
            self.engine_var.set("onnx")
            self.batch_size_var.set(64)
            self.preprocess_workers_var.set(4)
            self.transfer_workers_var.set(1)

    def _choose_folders(self) -> None:
        folder = filedialog.askdirectory(title="Select image folder")
        if folder:
            self._add_folder(Path(folder))

    def _add_folder(self, folder: Path) -> None:
        folder = folder.resolve()
        if not folder.exists() or not folder.is_dir():
            return
        if folder in self.folders:
            return
        self.folders.append(folder)
        self.folder_list.insert(tk.END, str(folder))

    def _remove_selected(self) -> None:
        selected = list(self.folder_list.curselection())
        for index in reversed(selected):
            self.folder_list.delete(index)
            del self.folders[index]

    def _clear_folders(self) -> None:
        self.folder_list.delete(0, tk.END)
        self.folders.clear()

    def _drop_folders(self, event) -> None:
        for item in self.tk.splitlist(event.data):
            self._add_folder(Path(item))

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        if not self.folders:
            messagebox.showerror("Error", "Add at least one folder.")
            return

        self.start_button.configure(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_var.set("Loading model and scanning files...")
        self.detail_var.set("")

        self.worker = threading.Thread(
            target=self._run_worker,
            args=(list(self.folders),),
            daemon=True,
        )
        self.worker.start()

    def _run_worker(self, folders: list[Path]) -> None:
        def progress(done: int, total: int, path: Path, category: str) -> None:
            self.events.put(("progress", (done, total, path.name, category)))

        try:
            results = []
            for index, folder in enumerate(folders, start=1):
                self.events.put(("folder", (index, len(folders), folder)))
                result = scan_and_classify(
                    root=folder,
                    mode=self.mode_var.get(),
                    batch_size=int(self.batch_size_var.get()),
                    nude_threshold=float(self.nude_threshold_var.get()),
                    sexy_threshold=float(self.sexy_threshold_var.get()),
                    progress=progress,
                    log_path=folder / OUTPUT_DIR_NAME / "debug.log",
                    progress_interval=25,
                    device=self.device_var.get(),
                    transfer_workers=int(self.transfer_workers_var.get()),
                    engine=self.engine_var.get(),
                    preprocess_workers=int(self.preprocess_workers_var.get()),
                    output_strategy=self.output_strategy_var.get(),
                )
                results.append((folder, result))
            self.events.put(("done", results))
        except Exception as exc:
            self.events.put(("error", exc))

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "folder":
                    index, total, folder = payload
                    self.status_var.set(f"Folder {index}/{total}: {folder}")
                elif event == "progress":
                    done, total, filename, category = payload
                    percent = (done / total * 100) if total else 0
                    self.progress_var.set(percent)
                    self.status_var.set(f"{done}/{total} - {filename} -> {category}")
                    if category == "errors":
                        self.detail_var.set("Errors found. See debug.log under _classified.")
                elif event == "done":
                    self.start_button.configure(state=tk.NORMAL)
                    self.progress_var.set(100)
                    total_processed = sum(result.processed for _, result in payload)
                    total_errors = sum(result.errors for _, result in payload)
                    total_batch_errors = sum(result.batch_errors for _, result in payload)
                    self.status_var.set(
                        "Done: "
                        f"folders={len(payload)}, processed={total_processed}, "
                        f"errors={total_errors}, batch_errors={total_batch_errors}"
                    )
                    if total_errors or total_batch_errors:
                        self.detail_var.set("Warnings/errors found. See debug.log under _classified.")
                    messagebox.showinfo("Done", self.status_var.get())
                elif event == "error":
                    self.start_button.configure(state=tk.NORMAL)
                    self.status_var.set(f"Error: {payload}")
                    self.detail_var.set("Fatal error. See debug.log under _classified if it exists.")
                    messagebox.showerror("Error", str(payload))
        except queue.Empty:
            pass
        self.after(150, self._poll_events)


if __name__ == "__main__":
    app = ImageClassifierApp()
    app.mainloop()
