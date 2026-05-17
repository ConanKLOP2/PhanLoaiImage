from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from classify_images import OUTPUT_DIR_NAME, scan_and_classify


class ImageClassifierApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Phan loai anh nude / sexy / normal")
        self.geometry("860x450")
        self.minsize(760, 420)

        self.folder_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="copy")
        self.device_var = tk.StringVar(value="gpu")
        self.batch_size_var = tk.IntVar(value=128)
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

        ttk.Label(root, text="Thu muc anh").grid(row=0, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.folder_var).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(root, text="Chon...", command=self._choose_folder).grid(
            row=0, column=2, sticky="e"
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

        ttk.Label(settings, text="Batch size").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(
            settings,
            from_=1,
            to=512,
            textvariable=self.batch_size_var,
            width=8,
        ).grid(row=0, column=3, sticky="w")

        ttk.Label(settings, text="Nude threshold").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(
            settings,
            from_=0.1,
            to=0.95,
            increment=0.05,
            textvariable=self.nude_threshold_var,
            width=8,
        ).grid(row=0, column=5, sticky="w")

        ttk.Label(settings, text="Sexy threshold").grid(row=0, column=6, sticky="w")
        ttk.Spinbox(
            settings,
            from_=0.1,
            to=0.95,
            increment=0.05,
            textvariable=self.sexy_threshold_var,
            width=8,
        ).grid(row=0, column=7, sticky="w")

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
            self.folder_var.set(folder)

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return

        folder = Path(self.folder_var.get().strip())
        if not folder.exists() or not folder.is_dir():
            messagebox.showerror("Loi", "Thu muc khong ton tai.")
            return

        self.start_button.configure(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_var.set("Dang nap model va quet file...")
        self.error_var.set("")

        self.worker = threading.Thread(
            target=self._run_worker,
            args=(folder,),
            daemon=True,
        )
        self.worker.start()

    def _run_worker(self, folder: Path) -> None:
        def progress(done: int, total: int, path: Path, category: str) -> None:
            self.events.put(("progress", (done, total, path.name, category)))

        try:
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
            )
            self.events.put(("done", result))
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
                            f"Co file loi. Xem: {self.folder_var.get()}\\{OUTPUT_DIR_NAME}\\debug.log"
                        )
                elif event == "done":
                    self.start_button.configure(state=tk.NORMAL)
                    self.progress_var.set(100)
                    self.status_var.set(
                        "Hoan tat: "
                        f"processed={payload.processed}, "
                        f"skipped={payload.skipped}, errors={payload.errors}, "
                        f"batch_errors={payload.batch_errors}, "
                        f"providers={payload.providers}"
                    )
                    if payload.errors or payload.batch_errors:
                        self.error_var.set(f"Co file loi. Xem: {payload.log_path}")
                    messagebox.showinfo("Hoan tat", self.status_var.get())
                elif event == "error":
                    self.start_button.configure(state=tk.NORMAL)
                    self.status_var.set(f"Loi: {payload}")
                    self.error_var.set(
                        f"Loi nghiem trong. Xem: {self.folder_var.get()}\\{OUTPUT_DIR_NAME}\\debug.log"
                    )
                    messagebox.showerror("Loi", str(payload))
        except queue.Empty:
            pass
        self.after(150, self._poll_events)


if __name__ == "__main__":
    app = ImageClassifierApp()
    app.mainloop()
