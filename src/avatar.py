"""Text-to-Sign avatar sequencing and gesture image loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

from src.model import STATIC_LETTERS

DisplayCallback = Callable[[Image.Image | None], None]
VideoCallback = Callable[[Path | None], None]
OverlayCallback = Callable[[str], None]
AfterCallback = Callable[[int, Callable[[], None]], Any]


class Avatar:
    """Load gesture images and animate typed text letter by letter."""

    def __init__(
        self,
        config: dict[str, Any],
        display_callback: DisplayCallback,
        overlay_callback: OverlayCallback,
        video_callback: VideoCallback | None = None,
    ) -> None:
        """Create an avatar sequencer."""
        self.config = config
        self.display_callback = display_callback
        self.video_callback = video_callback
        self.overlay_callback = overlay_callback
        self.state = "IDLE"
        self.sequence = ""
        self.index = 0
        self.after_fn: AfterCallback | None = None
        self.setup_required = False
        self._images: dict[str, Image.Image] = {}
        self._word_videos: dict[str, Path] = {}
        self.current_word_video: Path | None = None
        self.placeholder = self._make_placeholder("No Sign Available")
        self.blank = self._make_placeholder("")
        self._load_images()
        self._load_word_videos()

    def play(self, text: str, after_fn: AfterCallback) -> None:
        """Start playback for up to the configured 200-character limit."""
        max_chars = int(self.config["avatar"]["max_input_chars"])
        entered_text = text[:max_chars]
        word_video = self._find_word_video(entered_text)
        if word_video is not None:
            self.sequence = entered_text.strip()
            self.index = 0
            self.after_fn = after_fn
            self.state = "PLAYING"
            self.current_word_video = word_video
            self.display_callback(None)
            if self.video_callback is not None:
                self.video_callback(word_video)
            self.overlay_callback(f"{word_video.stem}")
            return

        if self.video_callback is not None:
            self.video_callback(None)
        self.current_word_video = None
        self.sequence = entered_text.upper()
        self.index = 0
        self.after_fn = after_fn
        if not self.sequence:
            self.overlay_callback("Done")
            self.state = "DONE"
            return
        self.state = "PLAYING"
        self._advance()

    def pause(self) -> None:
        """Pause playback at the current index."""
        if self.state == "PLAYING":
            self.state = "PAUSED"

    def resume(self) -> None:
        """Resume playback after a pause."""
        if self.state == "PAUSED":
            self.state = "PLAYING"
            self._schedule_current()

    def replay(self, after_fn: AfterCallback | None = None) -> None:
        """Restart playback from the beginning of the current input."""
        if after_fn is not None:
            self.after_fn = after_fn
        if self.current_word_video is not None and self.video_callback is not None:
            self.state = "PLAYING"
            self.index = 0
            self.display_callback(None)
            self.video_callback(self.current_word_video)
            self.overlay_callback(f"{self.current_word_video.stem}")
            return
        if self.sequence and self.after_fn is not None:
            self.index = 0
            self.state = "PLAYING"
            self._advance()

    def _load_images(self) -> None:
        signs_dir = Path(self.config["avatar"]["signs_dir"])
        signs_dir.mkdir(parents=True, exist_ok=True)
        image_files = [p for p in signs_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        self.setup_required = len(image_files) == 0
        if self.setup_required:
            self.overlay_callback("Please add ASL gesture images to the assets/signs/ directory.")

        for letter in STATIC_LETTERS:
            image_path = self._find_image(signs_dir, letter)
            if image_path is None:
                logging.warning("Missing gesture image for %s", letter)
                self._images[letter] = self.placeholder
                continue
            with Image.open(image_path) as image:
                self._images[letter] = image.convert("RGB").resize(
                    (int(self.config["avatar"]["image_size"]), int(self.config["avatar"]["image_size"])),
                    Image.Resampling.LANCZOS,
                )

    def _load_word_videos(self) -> None:
        videos_dir = Path(self.config["avatar"]["word_videos_dir"])
        videos_dir.mkdir(parents=True, exist_ok=True)
        for video_path in videos_dir.rglob("*.mp4"):
            for key in self._word_keys(video_path.stem):
                self._word_videos.setdefault(key, video_path)

    def _find_word_video(self, text: str) -> Path | None:
        clean_text = text.replace("%0A", "").replace("\n", "").strip()
        words = clean_text.split()
        if not words:
            return None

        candidates = {" ".join(words), "-".join(words), "_".join(words)}
        for key in {word_key for candidate in candidates for word_key in self._word_keys(candidate)}:
            video_path = self._word_videos.get(key)
            if video_path is not None:
                return video_path
        return None

    def _word_keys(self, word: str) -> set[str]:
        lowered = word.strip().lower()
        compact = "".join(ch for ch in lowered if ch.isalnum())
        keys = {key for key in (lowered, compact) if key}
        if lowered.endswith("%0"):
            keys.update(self._word_keys(lowered[:-2]))
        return keys

    def _find_image(self, signs_dir: Path, letter: str) -> Path | None:
        candidates = [path for path in signs_dir.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        normalized_letter = letter.upper()
        preferred_names = {
            normalized_letter,
            f"ASL{normalized_letter}",
            f"SIGN{normalized_letter}",
            f"LETTER{normalized_letter}",
            f"{normalized_letter}SIGN",
            f"{normalized_letter}ASL",
        }
        for path in candidates:
            if path.stem.upper() == normalized_letter:
                return path
        for path in candidates:
            compact_stem = "".join(ch for ch in path.stem.upper() if ch.isalnum())
            if compact_stem in preferred_names:
                return path
        return None

    def _advance(self) -> None:
        if self.state != "PLAYING":
            return
        if self.index >= len(self.sequence):
            self.state = "DONE"
            self.display_callback(None)
            if self.video_callback is not None:
                self.video_callback(None)
            self.overlay_callback("Done")
            return

        current = self.sequence[self.index]
        image, duration = self._image_and_duration(current)
        self.display_callback(image)
        self.overlay_callback(f"{current} ({self.index + 1}/{len(self.sequence)})")
        self.index += 1
        if self.after_fn is not None:
            self.after_fn(duration, self._advance)

    def _schedule_current(self) -> None:
        if self.after_fn is not None:
            self.after_fn(0, self._advance)

    def _image_and_duration(self, char: str) -> tuple[Image.Image, int]:
        if char == " ":
            return self.blank, int(self.config["avatar"]["space_duration_ms"])
        if char in self._images and char in STATIC_LETTERS:
            return self._images[char], int(self.config["avatar"]["display_duration_ms"])
        return self.placeholder, int(self.config["avatar"]["no_sign_duration_ms"])

    def _make_placeholder(self, text: str) -> Image.Image:
        size = int(self.config["avatar"]["image_size"])
        image = Image.new("RGB", (size, size), color=(245, 245, 245))
        if text:
            draw = ImageDraw.Draw(image)
            draw.text((size // 2, size // 2), text, fill=(30, 30, 30), anchor="mm")
        return image
