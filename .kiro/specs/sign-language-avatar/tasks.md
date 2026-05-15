# Implementation Plan: Sign Language Avatar System

## Overview

Incremental implementation of a local Python desktop application for two-way ASL communication. Tasks build from infrastructure upward: config and project scaffold → core ML components → training pipeline → live recognition → avatar animation → GUI → entry points → documentation and tests. Each step integrates with the previous so there is no orphaned code.

Language: **Python**

---

## Tasks

- [x] 1. Project scaffold — config, requirements, and directory structure
  - Create `config.yaml` at the project root with all sections from the design schema (dataset, model, training, augmentation, inference, avatar, logging)
  - Create `requirements.txt` with pinned exact versions for: torch, torchvision, opencv-python, Pillow, pandas, numpy, PyYAML, hypothesis
  - Create empty placeholder `__init__.py` files for `src/` and `src/gui/`
  - Create `src/config.py` with `load_config(path)` that reads `config.yaml`, merges with hardcoded defaults, and returns a plain `dict`; missing keys must fall back to defaults without raising
  - Create `src/utils/logger.py` with `setup_logger(config)` that configures Python `logging` at the level from config and returns the root logger
  - Create stub empty files for all remaining source modules so imports resolve: `src/model.py`, `src/preprocessor.py`, `src/trainer.py`, `src/recognizer.py`, `src/avatar.py`, `src/gui/app.py`, `src/gui/main_menu.py`, `src/gui/recognizer_view.py`, `src/gui/avatar_view.py`
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [-] 2. SignNet CNN model (`src/model.py`)
  - Implement `SignNet(nn.Module)` with exactly three convolutional blocks (Conv2d → BatchNorm2d → ReLU → MaxPool2d) with filter counts 32, 64, 128
  - Implement the fully connected classifier: Flatten → Linear(1152→512) → ReLU → Dropout(p) → Linear(512→256) → ReLU → Dropout(p) → Linear(256→24)
  - Implement `forward(x)` with shape validation: raise `ValueError` if input is not `(B, 1, 28, 28)`
  - Implement `predict(x)` convenience method returning `(class_index: int, confidence: float)` via softmax; must call `model.eval()` and use `torch.no_grad()`
  - Add Google-style docstrings to the class and all public methods
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.7, 3.2, 3.6_

  - [ ]* 2.1 Write property test — SignNet shape invariant
    - **Property: Shape invariant** — for any batch size B in [1, 8], `SignNet.forward(rand(B,1,28,28))` returns shape `(B, 24)`
    - **Validates: Requirements 2.1, 2.5**

  - [ ]* 2.2 Write property test — SignNet determinism in eval mode
    - **Property: Determinism** — calling `forward(x)` twice with the same tensor in `eval()` mode returns identical tensors
    - **Validates: Requirements 2.5**

  - [ ]* 2.3 Write property test — confidence bounds
    - **Property: Confidence bounds** — `predict(x)` always returns `confidence` in `[0.0, 1.0]` and `class_index` in `[0, 23]`
    - **Validates: Requirements 3.2**

  - [ ]* 2.4 Write unit tests for SignNet shape rejection
    - Test that `forward(x)` raises `ValueError` for inputs with wrong channel count, wrong spatial size, and wrong number of dimensions
    - _Requirements: 3.6_

- [ ] 3. Preprocessor (`src/preprocessor.py`)
  - Implement `Preprocessor` class with static method `prepare_dataset_image(image, augment=False)`: accepts a `(28, 28)` uint8 numpy array, applies optional augmentation (rotation ≤10°, horizontal flip, brightness/contrast jitter), normalizes to `[0.0, 1.0]`, returns float32 tensor of shape `(1, 28, 28)`
  - Implement static method `prepare_frame(roi_array)`: accepts a BGR numpy array of any size, converts to grayscale, resizes to 28×28, normalizes by dividing by 255.0, returns float32 tensor of shape `(1, 1, 28, 28)` (adds batch dimension)
  - Ensure both methods apply identical normalization (pixel / 255.0)
  - Add Google-style docstrings
  - _Requirements: 1.2, 1.3, 1.10, 3.1, 3.4, 3.5, 4.3_

  - [ ]* 3.1 Write property test — output shape
    - **Property: Output shape** — `prepare_frame(roi)` always returns a tensor of shape `(1, 1, 28, 28)` for any non-empty BGR array
    - **Validates: Requirements 3.1, 3.5**

  - [ ]* 3.2 Write property test — value range
    - **Property: Value range** — all values in the output tensor of `prepare_frame` are in `[0.0, 1.0]`
    - **Validates: Requirements 1.2, 3.4**

  - [ ]* 3.3 Write property test — idempotency of normalization
    - **Property: Idempotency** — applying `prepare_frame` twice to the same ROI produces the same tensor
    - **Validates: Requirements 3.4**

- [ ] 4. Dataset and Trainer (`src/trainer.py` and `train.py`)
  - Implement `SignMNISTDataset(Dataset)` in `src/trainer.py`: reads CSV (label + 784 pixel columns), reshapes pixels to `(N, 28, 28)`, applies `Preprocessor.prepare_dataset_image` in `__getitem__`
  - Implement `Trainer` class with `train(config)` method: loads train/test CSVs, creates `DataLoader`s, instantiates `SignNet`, Adam optimizer with weight decay ≥ 1e-4, and LR scheduler (ReduceLROnPlateau or CosineAnnealingLR per config)
  - Implement training loop: for each epoch, run `train_epoch`, run `evaluate`, call `scheduler.step`, append row to `logs/training_log.csv` (columns: epoch, train_loss, val_accuracy), create `logs/` if missing
  - Implement `save_if_better`: save checkpoint to `models/signnet_best.pth` only when new accuracy ≥ previous best; create `models/` if missing
  - Print per-class accuracy and overall accuracy after final evaluation; print device string at startup
  - Implement `train.py` at project root as CLI entry point: calls `load_config`, instantiates `Trainer`, runs training
  - Add Google-style docstrings to all public classes and methods
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.11, 1.12, 2.6, 9.4, 9.5_

  - [ ]* 4.1 Write unit tests for dataset loading
    - Test that `SignMNISTDataset` loads the correct number of samples from the CSV
    - Test that `__getitem__` returns a float32 tensor of shape `(1, 28, 28)` and a valid label in `[0, 23]`
    - _Requirements: 1.1, 1.10_

  - [ ]* 4.2 Write unit tests for checkpoint saving logic
    - Test `save_if_better`: verify checkpoint is written when accuracy improves, not written when it does not, and `models/` directory is created if absent
    - _Requirements: 1.6, 1.7_

- [~] 5. Checkpoint — ensure scaffold, model, preprocessor, and trainer compile and tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Recognizer (`src/recognizer.py`)
  - Implement `Recognizer` class: constructor accepts `config` and a loaded `SignNet` model; opens `cv2.VideoCapture(camera_index)` on `start()`
  - Implement `step()` method: reads one frame, calls `extract_roi`, calls `Preprocessor.prepare_frame`, calls `model.predict`, overlays predicted letter or "Low Confidence" on the annotated frame, returns the annotated BGR frame
  - Implement `extract_roi(frame)` and `draw_roi_rectangle(frame)` as per the design algorithms (centered 300×300 square)
  - Implement `Sentence Buffer` management: `append_letter(letter)`, `append_space()`, `backspace()`, `clear()`; buffer initializes to empty string on session start
  - Implement `stop()`: always calls `cap.release()` even if an exception occurred (use try/finally)
  - Handle camera-unavailable case: if `VideoCapture.isOpened()` returns False, raise a descriptive exception that the GUI can catch and display as "Camera not available"
  - Add Google-style docstrings
  - _Requirements: 3.3, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 4.11, 4.12_

  - [ ]* 6.1 Write property test — sentence buffer monotonicity
    - **Property: Sentence buffer monotonicity** — the buffer only grows by one character (append) or shrinks by one (backspace) per action; it never resets mid-session except on explicit `clear()`
    - **Validates: Requirements 4.5, 4.6, 4.7**

  - [ ]* 6.2 Write unit tests for ROI extraction
    - Test `extract_roi` returns an array of shape `(roi_size, roi_size, C)` for various frame sizes
    - Test that the ROI is centered in the frame
    - _Requirements: 4.2, 4.3_

  - [ ]* 6.3 Write unit tests for confidence gate
    - Test that a letter is only appended to the buffer when confidence ≥ threshold
    - Test that "Low Confidence" is returned when confidence < threshold
    - _Requirements: 3.3, 4.4, 4.11_

  - [ ]* 6.4 Write unit tests for camera release guarantee
    - Test that `stop()` calls `cap.release()` even when an exception is raised during `step()`
    - _Requirements: 4.8_

- [ ] 7. Avatar (`src/avatar.py`)
  - Implement `Avatar` class: constructor accepts `config`, `display_callback(PIL.Image | None)`, and `overlay_callback(str)`; calls `_load_images()` at startup
  - Implement `_load_images()`: iterate over all 24 static letters, try `.png` then `.jpg`/`.jpeg`; resize to `image_size × image_size` using `Image.LANCZOS`; log a warning for any missing letter; if `assets/signs/` contains zero images, set a flag to display setup instruction
  - Implement `play(text, after_fn)`: uppercase the text, tokenize to letter list, set state to PLAYING, call `_advance()`
  - Implement `_advance()`: display current letter's image (or placeholder for space/J/Z/non-alpha), call `overlay_callback("X (n/total)")`, schedule next `_advance()` via `after_fn` with the appropriate duration from config
  - Implement `pause()`, `resume()`, `replay(after_fn)` state transitions as per the design state machine (IDLE → PLAYING → PAUSED → PLAYING → DONE)
  - When sequence completes, call `display_callback(None)` and `overlay_callback("Done")`
  - Add Google-style docstrings
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11, 5.12, 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 7.1 Write property test — sequence completeness
    - **Property: Sequence completeness** — `play(text)` eventually calls `overlay_callback("Done")` for any finite uppercase string, provided state remains PLAYING
    - **Validates: Requirements 5.12**

  - [ ]* 7.2 Write property test — letter coverage
    - **Property: Letter coverage** — for every letter in `"ABCDEFGHIKLMNOPQRSTUVWXY"`, if the image file exists, `Avatar._images[L]` is a non-None PIL.Image of size `(400, 400)`
    - **Validates: Requirements 6.1, 6.3**

  - [ ]* 7.3 Write unit tests for avatar state machine
    - Test IDLE → PLAYING on `play()`
    - Test PLAYING → PAUSED on `pause()`, then PAUSED → PLAYING on `resume()` from same index
    - Test DONE → PLAYING on `replay()`
    - _Requirements: 5.8, 5.9, 5.10_

  - [ ]* 7.4 Write unit tests for special character handling
    - Test that space produces a 400ms blank placeholder
    - Test that J, Z, and non-alpha characters produce "No Sign Available" placeholder for 800ms
    - _Requirements: 5.5, 5.6_

- [ ] 8. GUI — App, MainMenu, RecognizerView, AvatarView
  - Implement `src/gui/app.py` — `App(tk.Tk)`: set window title to "ASL Sign Language System", build all three frames in a stacked container using `grid(row=0, column=0, sticky="nsew")`, implement `show_frame(name)` and `show_error(exc)` (modal error dialog → return to MainMenu)
  - Implement `src/gui/main_menu.py` — `MainMenuFrame(tk.Frame)`: two buttons labeled exactly "Sign Recognition (Camera)" and "Sign Avatar (Text-to-Sign)"; disable "Sign Recognition (Camera)" if checkpoint file does not exist
  - Implement `src/gui/recognizer_view.py` — `RecognizerView(tk.Frame)`: `tk.Label` for video feed (updated via `PIL.ImageTk`), read-only `tk.Text` for Sentence Buffer, prediction label, "Back to Menu" button; start `Recognizer` on show, stop on hide; drive camera loop via `after(33, _update_frame)`; handle "Camera not available" by showing error dialog
  - Implement `src/gui/avatar_view.py` — `AvatarView(tk.Frame)`: `tk.Entry` (max 200 chars with visible "Character limit reached" indicator), Play/Pause/Resume/Replay buttons, 400×400 gesture image label, progress overlay label, "Back to Menu" button; wire buttons to `Avatar` methods
  - Add Google-style docstrings to all public classes and methods
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 5.1, 5.7, 5.8, 5.9, 5.10, 5.11, 5.12, 9.1, 9.2, 9.3_

  - [ ]* 8.1 Write unit tests for GUI frame navigation
    - Test that `show_frame("Recognizer")` raises the correct frame
    - Test that `show_error(exc)` calls `show_frame("MainMenu")` after displaying the dialog
    - _Requirements: 7.2, 7.3, 7.4, 7.8_

  - [ ]* 8.2 Write unit tests for checkpoint-conditional button state
    - Test that "Sign Recognition (Camera)" is disabled when checkpoint file is absent
    - Test that it is enabled when checkpoint file exists
    - _Requirements: 9.1, 9.3_

- [~] 9. Entry point (`main.py`)
  - Implement `main.py` at the project root: call `load_config("config.yaml")`, check for checkpoint existence, instantiate `App(config)`, call `app.mainloop()`
  - Ensure `python main.py` launches the GUI without errors when the checkpoint is absent (button disabled) and when it is present (button enabled)
  - Add a `if __name__ == "__main__":` guard
  - _Requirements: 8.5, 9.1, 9.2, 9.3, 9.4, 9.5_

- [~] 10. Checkpoint — ensure full application wires together and all existing tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [~] 11. README.md documentation
  - Write `README.md` at the project root with the following sections (exact titles required):
    - Project overview
    - Prerequisites (Python version, OS)
    - Installation steps
    - Dataset setup instructions
    - Training instructions (`python train.py`)
    - Usage instructions for Sign Recognition mode and Avatar mode
    - **Model Architecture** — describe SignNet layers and expected test accuracy
    - **Project Structure** — list top-level directories and their purpose
    - **Troubleshooting** — entries for: missing model checkpoint, camera not found, missing gesture image assets
  - _Requirements: 10.1, 10.2, 10.3, 10.5_

- [ ] 12. Unit tests (`tests/` directory)
  - Create `tests/__init__.py` and the following test files, each with at least the test cases described in the design's Testing Strategy (Section 11):
    - `tests/test_model.py` — SignNet shape, output range, ValueError on bad input
    - `tests/test_preprocessor.py` — tensor shape, value range, grayscale conversion
    - `tests/test_trainer.py` — dataset loading, checkpoint saving logic
    - `tests/test_recognizer.py` — ROI extraction, confidence scoring, sentence buffer ops
    - `tests/test_avatar.py` — image loading, duration logic, state machine transitions
    - `tests/test_config.py` — key loading, fallback defaults, missing key handling
  - Add property-based tests using `hypothesis` for the properties defined in Section 10 of the design (SignNet shape invariant, determinism, confidence bounds; Preprocessor output shape, value range, idempotency; Avatar sequence completeness, letter coverage; Recognizer sentence buffer monotonicity)
  - _Requirements: 10.4_ (docstrings already added per-module above)

  - [ ]* 12.1 Write integration smoke test — training pipeline
    - Run 1 epoch on a 100-sample subset; verify `training_log.csv` is created and checkpoint is saved
    - _Requirements: 1.8, 1.6_

  - [ ]* 12.2 Write integration smoke test — inference pipeline
    - Load checkpoint (or a freshly initialized model), pass a random tensor, verify output shape `(1, 24)` and confidence in `[0.0, 1.0]`
    - _Requirements: 3.2, 3.5_

- [~] 13. Final checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints (tasks 5, 10, 13) ensure incremental validation at natural integration boundaries
- Property tests use `hypothesis` and validate the universal invariants from design Section 10
- Unit tests validate specific examples and edge cases
- The `assets/signs/` directory must be populated manually with ASL gesture images before running the Avatar; the README covers this in the dataset setup section

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2", "3"] },
    { "id": 2, "tasks": ["2.1", "2.2", "2.3", "2.4", "3.1", "3.2", "3.3"] },
    { "id": 3, "tasks": ["4"] },
    { "id": 4, "tasks": ["4.1", "4.2"] },
    { "id": 5, "tasks": ["6", "7"] },
    { "id": 6, "tasks": ["6.1", "6.2", "6.3", "6.4", "7.1", "7.2", "7.3", "7.4"] },
    { "id": 7, "tasks": ["8"] },
    { "id": 8, "tasks": ["8.1", "8.2"] },
    { "id": 9, "tasks": ["9"] },
    { "id": 10, "tasks": ["11", "12"] },
    { "id": 11, "tasks": ["12.1", "12.2"] }
  ]
}
```
