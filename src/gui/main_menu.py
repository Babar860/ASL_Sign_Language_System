"""Main menu frame for choosing application modes."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from typing import Any


class MainMenuFrame(tk.Frame):
    """Main menu with buttons for recognition and avatar modes."""

    def __init__(self, parent: tk.Misc, controller: Any, config: dict[str, Any]) -> None:
        """Build the main menu widgets."""
        super().__init__(parent, padx=48, pady=48)
        self.controller = controller
        self.config_data = config

        tk.Label(self, text="ASL Sign Language System", font=("Segoe UI", 24, "bold")).pack(pady=(40, 12))
        self.model_status = tk.Label(self, text="", font=("Segoe UI", 11))
        self.model_status.pack(pady=(0, 24))

        self.recognition_button = tk.Button(
            self,
            text="Sign Recognition (Camera)",
            command=lambda: controller.show_frame("Recognizer"),
            width=32,
            height=2,
        )
        self.recognition_button.pack(pady=10)

        tk.Button(
            self,
            text="Sign Avatar (Text-to-Sign)",
            command=lambda: controller.show_frame("Avatar"),
            width=32,
            height=2,
        ).pack(pady=10)

    def on_show(self) -> None:
        """Refresh checkpoint-dependent recognition availability."""
        checkpoint = Path(self.config_data["model"]["checkpoint_path"])
        word_checkpoint = Path(self.config_data["model"]["word_checkpoint_path"])
        if word_checkpoint.exists():
            self.recognition_button.configure(state="normal")
            self.model_status.configure(text=f"Word model loaded from {word_checkpoint}")
        elif checkpoint.exists():
            self.recognition_button.configure(state="normal")
            self.model_status.configure(text=f"Model loaded from {checkpoint}")
        else:
            self.recognition_button.configure(state="disabled")
            self.model_status.configure(text="No trained model found. Please run train.py or train_words.py first.")
