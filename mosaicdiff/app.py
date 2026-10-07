"""Dark window: add videos, set a few options, start."""

from __future__ import annotations

import queue
import threading
import webbrowser
import tkinter.filedialog as filedialog
from pathlib import Path

import customtkinter as ctk

from mosaicdiff.comfy_setup import comfy_instructions, comfy_ready, discover_comfy, ensure_rtx_package
from mosaicdiff.fetch import FetchError, ensure_weights
from mosaicdiff.paths import MODEL_LABELS, default_output_dir, is_frozen
from mosaicdiff.pipeline import Cancelled, process_video, unique_path
from mosaicdiff.settings import PATH_KEYS, Settings

BG = "#14161c"
CARD = "#1e2128"
RAISED = "#2a2e37"
TEXT = "#f2efe9"
MUTED = "#a39e94"
LINE = "#3a3f4a"
ACCENT = "#e0a15a"
ACCENT_TEXT = "#1c140c"
DANGER = "#d4645a"
OK = "#8dcea8"

VIDEO_TYPES = (
    ("Videos", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v"),
    ("All files", "*.*"),
)
PATREON_URL = "https://www.patreon.com/cw/hometogether"


class MosaicDiffApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.settings = Settings.load()
        self.title("MosaicDiff")
        self.geometry("980x720")
        self.minsize(860, 620)
        self.configure(fg_color=BG)
        self._files: list[dict] = []
        self._events: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(80, self._poll)

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(5, weight=1)

        header = ctk.CTkFrame(self, fg_color=BG)
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(22, 8))
        promo = ctk.CTkFrame(header, fg_color=CARD, corner_radius=10)
        promo.pack(side="right", anchor="n", padx=(16, 0))
        ctk.CTkLabel(promo, text="Home Together", text_color=TEXT, font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=12, pady=(8, 0))
        ctk.CTkLabel(promo, text="Adult game on Patreon", text_color=MUTED, font=ctk.CTkFont(size=12)).pack(anchor="w", padx=12)
        ctk.CTkButton(
            promo,
            text="Open Patreon",
            fg_color="#FF424D",
            hover_color="#e23b45",
            text_color="#ffffff",
            command=self._open_patreon,
        ).pack(padx=12, pady=(6, 10))
        ctk.CTkLabel(header, text="MosaicDiff", text_color=TEXT, font=ctk.CTkFont(size=28, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text="BasicVSR++ finds the mosaic. Eros Max repaints it.",
            text_color=MUTED,
            font=ctk.CTkFont(size=14),
        ).pack(anchor="w", pady=(2, 0))

        toolbar = ctk.CTkFrame(self, fg_color=BG)
        toolbar.grid(row=1, column=0, sticky="ew", padx=28, pady=(8, 8))
        ctk.CTkButton(toolbar, text="Add videos", fg_color=ACCENT, text_color=ACCENT_TEXT, hover_color="#c98b45", command=self._add).pack(side="left")
        ctk.CTkButton(toolbar, text="Clear", fg_color=RAISED, text_color=TEXT, hover_color=LINE, command=self._clear).pack(side="left", padx=(8, 0))

        self.list_frame = ctk.CTkScrollableFrame(self, fg_color=CARD, corner_radius=12)
        self.list_frame.grid(row=2, column=0, sticky="nsew", padx=28, pady=(0, 12))
        self._empty = ctk.CTkLabel(self.list_frame, text="No videos yet", text_color=MUTED)
        self._empty.pack(anchor="w", padx=8, pady=12)

        options = ctk.CTkFrame(self, fg_color=CARD, corner_radius=12)
        options.grid(row=3, column=0, sticky="ew", padx=28, pady=(0, 12))
        options.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(options, text="Output folder", text_color=MUTED).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 4))
        self.output_entry = ctk.CTkEntry(options, fg_color=BG, border_color=LINE, text_color=TEXT)
        self.output_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8), pady=(14, 4))
        self.output_entry.insert(0, self.settings.output_dir or str(default_output_dir()))
        browse = ctk.CTkButton(options, text="Browse", width=90, fg_color=RAISED, text_color=TEXT, hover_color=LINE, command=self._browse_output)
        browse.grid(row=0, column=2, padx=(0, 16), pady=(14, 4))
        if is_frozen():
            self.output_entry.configure(state="disabled")
            browse.configure(state="disabled")

        ctk.CTkLabel(options, text="H3 length", text_color=MUTED).grid(row=1, column=0, sticky="w", padx=16, pady=8)
        self.seconds = ctk.CTkSlider(options, from_=1, to=15, number_of_steps=14, command=self._on_seconds, button_color=ACCENT, progress_color=ACCENT)
        self.seconds.grid(row=1, column=1, sticky="ew", padx=8, pady=8)
        self.seconds.set(self.settings.h3_seconds)
        self.seconds_label = ctk.CTkLabel(options, text=f"{self.settings.h3_seconds}s", text_color=TEXT, width=70)
        self.seconds_label.grid(row=1, column=2, padx=(0, 16))

        ctk.CTkLabel(options, text="H3 resolution", text_color=MUTED).grid(row=2, column=0, sticky="w", padx=16, pady=8)
        self.resolution = ctk.CTkSlider(options, from_=512, to=1280, number_of_steps=24, command=self._on_resolution, button_color=ACCENT, progress_color=ACCENT)
        self.resolution.grid(row=2, column=1, sticky="ew", padx=8, pady=8)
        self.resolution.set(self.settings.h3_resolution)
        self.resolution_label = ctk.CTkLabel(options, text=str(self.settings.h3_resolution), text_color=TEXT, width=70)
        self.resolution_label.grid(row=2, column=2, padx=(0, 16), pady=(8, 14))

        actions = ctk.CTkFrame(self, fg_color=BG)
        actions.grid(row=4, column=0, sticky="ew", padx=28, pady=(0, 8))
        self.start_button = ctk.CTkButton(actions, text="Start", width=120, fg_color=ACCENT, text_color=ACCENT_TEXT, hover_color="#c98b45", command=self._start)
        self.start_button.pack(side="left")
        self.stop_button = ctk.CTkButton(actions, text="Stop", width=90, fg_color=RAISED, text_color=TEXT, hover_color=LINE, command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        ctk.CTkButton(actions, text="Models", width=90, fg_color=RAISED, text_color=TEXT, hover_color=LINE, command=self._models).pack(side="left", padx=(8, 0))
        self.progress = ctk.CTkProgressBar(actions, progress_color=ACCENT, fg_color=RAISED)
        self.progress.pack(side="left", fill="x", expand=True, padx=(16, 0))
        self.progress.set(0)

        self.log = ctk.CTkTextbox(self, fg_color=CARD, text_color=TEXT, font=ctk.CTkFont(family="Consolas", size=12))
        self.log.grid(row=5, column=0, sticky="nsew", padx=28, pady=(0, 22))
        self.log.insert("end", "Add videos and press Start.\n")
        self.log.configure(state="disabled")

    def _open_patreon(self) -> None:
        webbrowser.open(PATREON_URL)

    def _add(self) -> None:
        names = filedialog.askopenfilenames(title="Add videos", filetypes=VIDEO_TYPES)
        for name in names:
            path = Path(name)
            if any(item["path"] == path for item in self._files):
                continue
            self._files.append({"path": path, "status": "Waiting"})
        self._refresh_list()

    def _clear(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._files.clear()
        self._refresh_list()

    def _refresh_list(self) -> None:
        for child in self.list_frame.winfo_children():
            child.destroy()
        if not self._files:
            self._empty = ctk.CTkLabel(self.list_frame, text="No videos yet", text_color=MUTED)
            self._empty.pack(anchor="w", padx=8, pady=12)
            return
        for index, item in enumerate(self._files):
            row = ctk.CTkFrame(self.list_frame, fg_color=BG, corner_radius=8)
            row.pack(fill="x", padx=6, pady=4)
            ctk.CTkLabel(row, text=item["path"].name, text_color=TEXT, anchor="w").pack(side="left", padx=10, pady=8)
            color = OK if item["status"] == "Done" else DANGER if item["status"].startswith("Failed") else MUTED
            ctk.CTkLabel(row, text=item["status"], text_color=color).pack(side="right", padx=10)
            if not (self._worker and self._worker.is_alive()):
                ctk.CTkButton(
                    row,
                    text="Remove",
                    width=70,
                    fg_color=RAISED,
                    text_color=TEXT,
                    hover_color=LINE,
                    command=lambda i=index: self._remove(i),
                ).pack(side="right")

    def _remove(self, index: int) -> None:
        if 0 <= index < len(self._files):
            del self._files[index]
            self._refresh_list()

    def _browse_output(self) -> None:
        folder = filedialog.askdirectory(title="Output folder")
        if folder:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, folder)

    def _on_seconds(self, value: float) -> None:
        self.seconds_label.configure(text=f"{int(round(value))}s")

    def _on_resolution(self, value: float) -> None:
        snapped = int(round(float(value) / 32) * 32)
        snapped = min(1280, max(512, snapped))
        self.resolution_label.configure(text=str(snapped))

    def _collect_settings(self) -> Settings:
        if is_frozen():
            self.settings.output_dir = str(default_output_dir())
        else:
            self.settings.output_dir = self.output_entry.get().strip()
        self.settings.h3_seconds = int(round(self.seconds.get()))
        snapped = int(round(float(self.resolution.get()) / 32) * 32)
        self.settings.h3_resolution = min(1280, max(512, snapped))
        self.settings.compare = False
        self.settings.save()
        return self.settings

    def _start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        if not self._files:
            self._write_log("Add a video first.")
            return
        settings = self._collect_settings()
        output = Path(settings.output_dir or default_output_dir())
        output.mkdir(parents=True, exist_ok=True)
        self._cancel.clear()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress.set(0)
        files = [item["path"] for item in self._files]
        for item in self._files:
            item["status"] = "Waiting"
        self._refresh_list()
        self._worker = threading.Thread(target=self._run, args=(files, output, settings), daemon=True)
        self._worker.start()

    def _stop(self) -> None:
        self._cancel.set()
        self._write_log("Stopping after the current step.")

    def _run(self, files: list[Path], output: Path, settings: Settings) -> None:
        try:
            try:
                log = lambda message: self._events.put(("log", message))

                def download_progress(fraction: float) -> None:
                    self._events.put(("progress", fraction))

                discover_comfy(settings, log)
                if not comfy_ready(settings):
                    raise RuntimeError(comfy_instructions(settings))
                ensure_weights(settings, log, self._cancel, download_progress)
                ensure_rtx_package(settings, log, self._cancel)
            except (FetchError, RuntimeError) as exc:
                self._events.put(("log", str(exc)))
                return
            for index, path in enumerate(files):
                if self._cancel.is_set():
                    self._events.put(("status", index, "Stopped"))
                    continue
                self._events.put(("status", index, "Running"))
                self._events.put(("log", f"Started {path.name}"))
                destination = unique_path(output / f"{path.stem}_mosaicdiff.mp4")

                def log(message, _index=index):
                    self._events.put(("log", message))

                def progress(fraction, _message, _index=index):
                    self._events.put(("progress", fraction))

                try:
                    written = process_video(path, destination, settings, log, progress, self._cancel)
                    self._events.put(("status", index, "Done"))
                    self._events.put(("log", f"Finished {written.name}"))
                except Cancelled:
                    self._events.put(("status", index, "Stopped"))
                    self._events.put(("log", f"Stopped {path.name}"))
                    break
                except Exception as exc:
                    self._events.put(("status", index, "Failed"))
                    self._events.put(("log", f"Failed {path.name}: {exc}"))
        finally:
            self._events.put(("idle",))

    def _poll(self) -> None:
        try:
            while True:
                kind, *payload = self._events.get_nowait()
                if kind == "log":
                    self._write_log(payload[0])
                elif kind == "status":
                    index, text = payload
                    if 0 <= index < len(self._files):
                        self._files[index]["status"] = text
                        self._refresh_list()
                elif kind == "progress":
                    self.progress.set(max(0.0, min(1.0, float(payload[0]))))
                elif kind == "idle":
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self._write_log("Ready.")
        except queue.Empty:
            pass
        self.after(80, self._poll)

    def _write_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _models(self) -> None:
        dialog = ctk.CTkToplevel(self)
        dialog.title("Models")
        dialog.geometry("760x520")
        dialog.configure(fg_color=BG)
        dialog.transient(self)
        entries = {}
        for row, key in enumerate(PATH_KEYS):
            ctk.CTkLabel(dialog, text=MODEL_LABELS[key], text_color=MUTED, anchor="w").grid(row=row, column=0, sticky="ew", padx=16, pady=6)
            entry = ctk.CTkEntry(dialog, fg_color=CARD, border_color=LINE, text_color=TEXT)
            entry.grid(row=row, column=1, sticky="ew", padx=8, pady=6)
            entry.insert(0, str(self.settings.resolved(key)))
            entries[key] = entry
            ctk.CTkButton(
                dialog,
                text="Browse",
                width=80,
                fg_color=RAISED,
                text_color=TEXT,
                hover_color=LINE,
                command=lambda k=key, e=entry: self._browse_model(k, e),
            ).grid(row=row, column=2, padx=(0, 16), pady=6)
        dialog.grid_columnconfigure(1, weight=1)

        def save() -> None:
            self.settings.paths = {key: entries[key].get().strip() for key in PATH_KEYS}
            self.settings.save()
            dialog.destroy()

        ctk.CTkButton(dialog, text="Save", fg_color=ACCENT, text_color=ACCENT_TEXT, hover_color="#c98b45", command=save).grid(row=len(PATH_KEYS), column=1, sticky="e", padx=8, pady=16)
        dialog.grab_set()

    def _browse_model(self, key: str, entry: ctk.CTkEntry) -> None:
        if key == "comfy_root":
            folder = filedialog.askdirectory(title=MODEL_LABELS[key])
            chosen = folder
        elif key == "comfy_python":
            chosen = filedialog.askopenfilename(title=MODEL_LABELS[key], filetypes=(("Python", "python.exe"), ("All files", "*.*")))
        else:
            chosen = filedialog.askopenfilename(title=MODEL_LABELS[key], filetypes=(("Model", "*.pth *.engine *.safetensors"), ("All files", "*.*")))
        if chosen:
            entry.delete(0, "end")
            entry.insert(0, chosen)

    def _close(self) -> None:
        self._collect_settings()
        self._cancel.set()
        self.destroy()


def run() -> None:
    MosaicDiffApp().mainloop()
