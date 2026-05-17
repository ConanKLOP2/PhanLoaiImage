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
        self.title("Phan loai anh nude / sexy / normal")
        self.geometry("860x450")
        self.minsize(760, 420)

        self.folder_var = tk.StringVar()
        self.folders: list[Path] = []
        self.mode_var = tk.StringVar(value="copy")
        self.device_var = tk.StringVar(value="gpu")
        self.engine_var = tk.StringVar(value="onnx")
        self.batch_size_var = tk.IntVar(value=250)
        self.transfer_workers_var = tk.IntVar(value=0)
        self.preprocess_workers_var = tk.IntVar(value=10)
        self.nude_threshold_var = tk.DoubleVar(value=0.8)
        self.sexy_threshold_var = tk.DoubleVar(value=0.8)
        self.status_var = tk.StringVar(value="Chon thu muc anh de bat dau.")
        self.error_var = tk.StringVar(value="")
        self.progress_var = tk.DoubleVar(value=0)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None

        self._build_ui()
        self.after(150, self._poll_events)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=18)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)

        ttk.Label(root, text="Folders").grid(row=0, column=0, sticky="w")
        self.folder_list = tk.Listbox(root, height=5, selectmode=tk.EXTENDED)
        self.folder_list.grid(row=0, column=1, sticky="nsew", padx=8)
        root.rowconfigure(0, weight=1)
        if DND_FILES:
            self.folder_list.drop_target_register(DND_FILES)
            self.folder_list.dnd_bind("<<Drop>>", self._drop_folders)

        folder_buttons = ttk.Frame(root)
        folder_buttons.grid(row=0, column=2, sticky="ne")
        ttk.Button(folder_buttons, text="Add...", command=self._choose_folder).pack(
            fill=tk.X, pady=(0, 6)
        )
        ttk.Button(folder_buttons, text="Remove", command=self._remove_selected).pack(
            fill=tk.X, pady=(0, 6)
        )
        ttk.Button(folder_buttons, text="Clear", command=self._clear_folders).pack(
            fill=tk.X
        )

        mode_frame = ttk.LabelFrame(root, text="Che do", padding=12)
        mode_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(16, 8))
        ttk.Radiobutton(
            mode_frame,
            text="Move file vao _classified",
            value="move",
            variable=self.mode_var,
        ).pack(side=tk.LEFT, padx=(0, 24))
        ttk.Radiobutton(
            mode_frame,
            text="Copy file, giu file goc",
            value="copy",
            variable=self.mode_var,
        ).pack(side=tk.LEFT)

        settings = ttk.LabelFrame(root, text="Cau hinh", padding=12)
        settings.grid(row=2, column=0, columnspan=3, sticky="ew", pady=8)
        for index in range(8):
            settings.columnconfigure(index, weight=1)

        ttk.Label(settings, text="Device").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.device_var,
            values=("auto", "gpu", "cpu"),
            state="readonly",
            width=8,
        ).grid(row=0, column=1, sticky="w")

        ttk.Label(settings, text="Engine").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.engine_var,
            values=("onnx", "nudenet"),
            state="readonly",
            width=8,
        ).grid(row=0, column=3, sticky="w")

        ttk.Label(settings, text="Batch size").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(
            settings,
            from_=1,
            to=1024,
            textvariable=self.batch_size_var,
            width=8,
        ).grid(row=0, column=5, sticky="w")

        ttk.Label(settings, text="Nude threshold").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(
            settings,
            from_=0.1,
            to=0.95,
            increment=0.05,
            textvariable=self.nude_threshold_var,
            width=8,
        ).grid(row=1, column=1, sticky="w")

        ttk.Label(settings, text="Sexy threshold").grid(row=1, column=2, sticky="w")
        ttk.Spinbox(
            settings,
            from_=0.1,
            to=0.95,
            increment=0.05,
            textvariable=self.sexy_threshold_var,
            width=8,
        ).grid(row=1, column=3, sticky="w")

        ttk.Label(settings, text="Preprocess workers").grid(row=1, column=4, sticky="w")
        ttk.Spinbox(
            settings,
            from_=1,
            to=32,
            textvariable=self.preprocess_workers_var,
            width=8,
        ).grid(row=1, column=5, sticky="w")

        ttk.Label(settings, text="Transfer workers").grid(row=1, column=6, sticky="w")
        ttk.Spinbox(
            settings,
            from_=0,
            to=16,
            textvariable=self.transfer_workers_var,
            width=8,
        ).grid(row=1, column=7, sticky="w")

        output_text = (
            f"Output mac dinh: <thu muc anh>\\{OUTPUT_DIR_NAME}\\"
            "nude, sexy, normal, errors"
        )
        ttk.Label(root, text=output_text).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

        self.progress = ttk.Progressbar(
            root, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.progress.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(18, 8))

        ttk.Label(root, textvariable=self.status_var).grid(
            row=5, column=0, columnspan=3, sticky="w"
        )

        ttk.Label(root, textvariable=self.error_var).grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )

        button_frame = ttk.Frame(root)
        button_frame.grid(row=7, column=0, columnspan=3, sticky="e", pady=(24, 0))
        self.start_button = ttk.Button(
            button_frame, text="Bat dau phan loai", command=self._start
        )
        self.start_button.pack(side=tk.RIGHT)

    def _choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="Chon thu muc anh")
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
            messagebox.showerror("Loi", "Chua chon thu muc.")
            return

        self.start_button.configure(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_var.set("Dang nap model va quet file...")
        self.error_var.set("")

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
                )
                results.append((folder, result))
            self.events.put(("done", results))
        except Exception as exc:
            self.events.put(("error", exc))

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "progress":
                    done, total, filename, category = payload
                    percent = (done / total * 100) if total else 0
                    self.progress_var.set(percent)
                    self.status_var.set(
                        f"{done}/{total} - {filename} -> {category}"
                    )
                    if category == "errors":
                        self.error_var.set(
                            f"Co file loi. Xem debug.log trong thu muc _classified."
                        )
                elif event == "folder":
                    index, total, folder = payload
                    self.status_var.set(f"Folder {index}/{total}: {folder}")
                elif event == "done":
                    self.start_button.configure(state=tk.NORMAL)
                    self.progress_var.set(100)
                    total_processed = sum(result.processed for _, result in payload)
                    total_errors = sum(result.errors for _, result in payload)
                    total_batch_errors = sum(result.batch_errors for _, result in payload)
                    self.status_var.set(
                        "Hoan tat: "
                        f"folders={len(payload)}, processed={total_processed}, "
                        f"errors={total_errors}, batch_errors={total_batch_errors}"
                    )
                    if total_errors or total_batch_errors:
                        self.error_var.set("Co canh bao/loi. Xem debug.log trong cac thu muc _classified.")
                    messagebox.showinfo("Hoan tat", self.status_var.get())
                elif event == "error":
                    self.start_button.configure(state=tk.NORMAL)
                    self.status_var.set(f"Loi: {payload}")
                    self.error_var.set(
                        "Loi nghiem trong. Xem debug.log trong thu muc _classified neu co."
                    )
                    messagebox.showerror("Loi", str(payload))
        except queue.Empty:
            pass
        self.after(150, self._poll_events)


if __name__ == "__main__":
    app = ImageClassifierApp()
    app.mainloop()
