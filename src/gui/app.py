"""Tkinter application root and frame manager."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from typing import Any

from src.gui.avatar_view import AvatarView
from src.gui.main_menu import MainMenuFrame
from src.gui.recognizer_view import RecognizerView


class App(tk.Tk):
    """Root window for the ASL Sign Language System."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Create the window and all application frames."""
        super().__init__()
        self.config_data = config
        self.title("ASL Sign Language System")
        self.geometry("960x720")
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.report_callback_exception = self._handle_callback_exception

        container = tk.Frame(self)
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames: dict[str, tk.Frame] = {}
        for name, frame_cls in (
            ("MainMenu", MainMenuFrame),
            ("Recognizer", RecognizerView),
            ("Avatar", AvatarView),
        ):
            frame = frame_cls(container, self, config)
            frame.grid(row=0, column=0, sticky="nsew")
            self.frames[name] = frame

        self.show_frame("MainMenu")

    def show_frame(self, name: str) -> None:
        """Raise one frame and call lifecycle hooks."""
        for frame_name, frame in self.frames.items():
            if frame_name != name and hasattr(frame, "on_hide"):
                frame.on_hide()
        frame = self.frames[name]
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()

    def show_error(self, exc: Exception | str) -> None:
        """Display a modal error dialog and return to the main menu."""
        messagebox.showerror("ASL Sign Language System", str(exc))
        self.show_frame("MainMenu")

    def _handle_callback_exception(self, exc_type: type[BaseException], exc: BaseException, traceback: Any) -> None:
        self.show_error(exc)
