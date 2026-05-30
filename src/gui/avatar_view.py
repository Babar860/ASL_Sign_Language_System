"""Tkinter frame for Text-to-Sign avatar playback."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Any, Callable

import cv2
from PIL import Image, ImageTk

from src.avatar import Avatar


class AvatarView(tk.Frame):
    """Text input and animation controls for avatar mode."""

    def __init__(self, parent: tk.Misc, controller: Any, config: dict[str, Any]) -> None:
        """Build the avatar UI."""
        super().__init__(parent, padx=16, pady=16)
        self.controller = controller
        self.config_data = config
        self.photo: ImageTk.PhotoImage | None = None
        self.video_capture: cv2.VideoCapture | None = None
        self.video_job: str | None = None
        self.text_var = tk.StringVar()
        self.text_var.trace_add("write", self._enforce_limit)

        toolbar = tk.Frame(self)
        toolbar.pack(fill="x")
        tk.Button(toolbar, text="Back to Menu", command=lambda: controller.show_frame("MainMenu")).pack(side="left")

        input_row = tk.Frame(self)
        input_row.pack(fill="x", pady=12)
        self.entry = tk.Entry(input_row, textvariable=self.text_var, font=("Segoe UI", 12))
        self.entry.pack(side="left", fill="x", expand=True)
        tk.Button(input_row, text="Play", command=self._play).pack(side="left", padx=6)
        tk.Button(input_row, text="Pause", command=self._pause).pack(side="left", padx=6)
        tk.Button(input_row, text="Resume", command=self._resume).pack(side="left", padx=6)
        tk.Button(input_row, text="Replay", command=self._replay).pack(side="left", padx=6)

        self.limit_label = tk.Label(self, text="", fg="#b00020")
        self.limit_label.pack(anchor="w")

        self.image_label = tk.Label(self, bg="#f5f5f5", width=400, height=400)
        self.image_label.pack(pady=16)
        self.overlay_label = tk.Label(self, text="", font=("Segoe UI", 16, "bold"))
        self.overlay_label.pack()
        self.avatar = Avatar(config, self._display_image, self._set_overlay, self._play_video)

    def on_show(self) -> None:
        """Focus the text field when the avatar view opens."""
        self.entry.focus_set()

    def on_hide(self) -> None:
        """Pause playback when leaving the avatar view."""
        self.avatar.pause()
        self._stop_video()

    def _play(self) -> None:
        text = self.text_var.get()
        if text:
            self.entry.configure(state="disabled")
            self.avatar.play(text, self._after)

    def _pause(self) -> None:
        self.avatar.pause()
        if self.video_job is not None:
            try:
                self.after_cancel(self.video_job)
            except tk.TclError:
                pass
            self.video_job = None

    def _resume(self) -> None:
        if self.video_capture is not None and self.video_job is None:
            self.avatar.state = "PLAYING"
            self._show_next_video_frame()
            return
        self.avatar.resume()

    def _replay(self) -> None:
        self.avatar.replay(self._after)

    def _after(self, delay_ms: int, callback: Callable[[], None]) -> Any:
        return self.after(delay_ms, callback)

    def _display_image(self, image: Image.Image | None) -> None:
        if image is None:
            self.image_label.configure(image="")
            self.entry.configure(state="normal")
            return
        self.photo = ImageTk.PhotoImage(image=image)
        self.image_label.configure(image=self.photo)

    def _set_overlay(self, text: str) -> None:
        self.overlay_label.configure(text=text)
        if text == "Done":
            self.entry.configure(state="normal")

    def _play_video(self, video_path: Path | None) -> None:
        self._stop_video()
        if video_path is None:
            return
        self.entry.configure(state="disabled")
        self.video_capture = cv2.VideoCapture(str(video_path))
        if not self.video_capture.isOpened():
            self.video_capture.release()
            self.video_capture = None
            self._set_overlay(f"Could not open {video_path.name}")
            return
        self._show_next_video_frame()

    def _show_next_video_frame(self) -> None:
        if self.video_capture is None:
            return
        ok, frame = self.video_capture.read()
        if not ok or frame is None:
            self._stop_video()
            self.avatar.state = "DONE"
            self._set_overlay("Done")
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize(
            (int(self.config_data["avatar"]["image_size"]), int(self.config_data["avatar"]["image_size"])),
            Image.Resampling.LANCZOS,
        )
        self.photo = ImageTk.PhotoImage(image=image)
        self.image_label.configure(image=self.photo)
        fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 24
        delay_ms = max(1, int(1000 / fps))
        self.video_job = str(self.after(delay_ms, self._show_next_video_frame))

    def _stop_video(self) -> None:
        if self.video_job is not None:
            try:
                self.after_cancel(self.video_job)
            except tk.TclError:
                pass
            self.video_job = None
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

    def _enforce_limit(self, *_: Any) -> None:
        max_chars = int(self.config_data["avatar"]["max_input_chars"])
        value = self.text_var.get()
        if len(value) > max_chars:
            self.text_var.set(value[:max_chars])
            self.limit_label.configure(text="Character limit reached")
        elif len(value) == max_chars:
            self.limit_label.configure(text="Character limit reached")
        else:
            self.limit_label.configure(text="")
