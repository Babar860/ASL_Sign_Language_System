"""Tkinter frame for live Sign-to-Text recognition."""

from __future__ import annotations

import tkinter as tk
from typing import Any

import cv2
from PIL import Image, ImageTk

from src.recognizer import Recognizer, load_model_for_inference


class RecognizerView(tk.Frame):
    """Display camera frames, predictions, and the sentence buffer."""

    def __init__(self, parent: tk.Misc, controller: Any, config: dict[str, Any]) -> None:
        """Build the recognizer UI."""
        super().__init__(parent, padx=16, pady=16)
        self.controller = controller
        self.config_data = config
        self.recognizer: Recognizer | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.running = False

        toolbar = tk.Frame(self)
        toolbar.pack(fill="x")
        tk.Button(toolbar, text="Back to Menu", command=lambda: controller.show_frame("MainMenu")).pack(side="left")
        tk.Button(toolbar, text="Show", command=self._on_show_letter).pack(side="left", padx=(12, 4))
        tk.Button(toolbar, text="Space", command=self._on_space_button).pack(side="left", padx=4)
        tk.Button(toolbar, text="Backspace", command=self._on_backspace_button).pack(side="left", padx=4)
        tk.Button(toolbar, text="Clear", command=self._on_clear).pack(side="left", padx=4)
        self.prediction_label = tk.Label(toolbar, text="Prediction: ", font=("Segoe UI", 14, "bold"))
        self.prediction_label.pack(side="right")

        self.video_label = tk.Label(self, bg="black", width=800, height=520)
        self.video_label.pack(fill="both", expand=True, pady=12)

        tk.Label(self, text="Recognized Text", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.buffer_text = tk.Text(self, height=3, state="disabled", font=("Segoe UI", 12))
        self.buffer_text.pack(fill="x")

        self.bind_all("<space>", self._on_space)
        self.bind_all("<Return>", self._on_enter)
        self.bind_all("<BackSpace>", self._on_backspace)
        self.bind_all("<q>", self._on_q)
        self.bind_all("<Q>", self._on_q)

    def on_show(self) -> None:
        """Start camera recognition when the frame becomes visible."""
        if self.running:
            return
        try:
            model = load_model_for_inference(self.config_data)
            self.recognizer = Recognizer(self.config_data, model)
            self.recognizer.start()
            self.running = True
            self._update_frame()
        except RuntimeError as exc:
            if str(exc) == "Camera not available":
                try:
                    self.controller.show_error("Camera not available")
                except Exception:
                    self._show_blank_state()
            else:
                self.controller.show_error(exc)
        except Exception as exc:
            self.controller.show_error(exc)

    def on_hide(self) -> None:
        """Stop camera recognition when leaving the frame."""
        self.running = False
        if self.recognizer is not None:
            self.recognizer.stop()
            self.recognizer = None

    def _update_frame(self) -> None:
        if not self.running or self.recognizer is None:
            return
        try:
            frame = self.recognizer.step()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            self.photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=self.photo)
            self.prediction_label.configure(text=f"Prediction: {self.recognizer.current_prediction}")
            self._set_buffer(self.recognizer.sentence_buffer)
            self.after(33, self._update_frame)
        except Exception as exc:
            self.controller.show_error(exc)

    def _set_buffer(self, text: str) -> None:
        self.buffer_text.configure(state="normal")
        self.buffer_text.delete("1.0", "end")
        self.buffer_text.insert("1.0", text)
        self.buffer_text.configure(state="disabled")

    def _on_space(self, event: tk.Event) -> str | None:
        self._on_space_button()
        return "break"

    def _on_enter(self, event: tk.Event) -> str | None:
        if self.recognizer is not None:
            self.recognizer.finalize_word()
            self._set_buffer(self.recognizer.sentence_buffer)
        return "break"

    def _on_backspace(self, event: tk.Event) -> str | None:
        self._on_backspace_button()
        return "break"

    def _on_q(self, event: tk.Event) -> str | None:
        self.controller.show_frame("MainMenu")
        return "break"

    def _show_blank_state(self) -> None:
        self.running = True
        self.video_label.configure(image="", text="")

    def _on_show_letter(self) -> None:
        if self.recognizer is not None:
            self.recognizer.show_current_prediction()
            self._set_buffer(self.recognizer.sentence_buffer)

    def _on_space_button(self) -> None:
        if self.recognizer is not None:
            self.recognizer.append_space()
            self._set_buffer(self.recognizer.sentence_buffer)

    def _on_backspace_button(self) -> None:
        if self.recognizer is not None:
            self.recognizer.backspace()
            self._set_buffer(self.recognizer.sentence_buffer)

    def _on_clear(self) -> None:
        if self.recognizer is not None:
            self.recognizer.clear()
            self._set_buffer(self.recognizer.sentence_buffer)
