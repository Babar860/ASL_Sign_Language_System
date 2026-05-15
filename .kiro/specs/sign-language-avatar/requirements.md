# Requirements Document

## Introduction

This document defines the requirements for a two-way Sign Language Recognition and Word-to-Sign Avatar System. The system enables real-time American Sign Language (ASL) communication in both directions: it recognizes hand gestures captured via a webcam and converts them to text (Sign-to-Text), and it animates an avatar to display ASL gestures for any typed text (Text-to-Sign). The system is built on a PyTorch CNN model trained on the Sign Language MNIST dataset, with OpenCV handling live video capture, and a Python-based GUI tying both modes together.

---

## Glossary

- **SignNet**: The PyTorch convolutional neural network model that classifies ASL hand gesture images into one of 24 letter classes (A–Z, excluding J and Z which require motion).
- **Recognizer**: The Sign-to-Text subsystem that captures webcam frames, extracts the hand region, and feeds it to SignNet for classification.
- **Avatar**: The Text-to-Sign subsystem that renders a sequence of ASL gesture images or animations in response to typed text input.
- **Trainer**: The training pipeline that loads the Sign Language MNIST dataset, trains SignNet, evaluates accuracy, and saves the model checkpoint.
- **Preprocessor**: The image processing component that normalizes, resizes, and transforms raw webcam frames or dataset images into the tensor format expected by SignNet.
- **GUI**: The graphical user interface that hosts both the Recognizer and Avatar modes and allows the user to switch between them.
- **Sign Language MNIST**: The dataset of 28×28 grayscale images of ASL hand gestures, with 24 classes (letters A–Z excluding J and Z), stored as CSV files.
- **ROI**: Region of Interest — the bounding box within a webcam frame that contains the user's hand for gesture recognition.
- **Checkpoint**: A saved file containing the trained weights of SignNet, used to restore the model without retraining.
- **Confidence Score**: The probability value (0.0–1.0) output by SignNet for the predicted class, computed via softmax over the output logits.
- **Data Augmentation**: Transformations applied to training images (rotation, flipping, brightness changes) to improve model generalization.
- **Sentence Buffer**: An in-memory string that accumulates recognized letters or words during a live recognition session.

---

## Requirements

### Requirement 1: Model Training Pipeline

**User Story:** As a developer, I want to train an improved SignNet model on the Sign Language MNIST dataset, so that the system achieves higher than 92.12% test accuracy and produces a reusable checkpoint.

#### Acceptance Criteria

1. THE Trainer SHALL load training data from `archive/sign_mnist_train/sign_mnist_train.csv` and test data from `archive/sign_mnist_test/sign_mnist_test.csv`.
2. THE Trainer SHALL normalize pixel values to the range [0.0, 1.0] before training.
3. THE Trainer SHALL apply data augmentation (random rotation up to 10°, random horizontal flip, random brightness and contrast jitter) to training images.
4. THE Trainer SHALL use a learning rate scheduler (ReduceLROnPlateau or CosineAnnealingLR) during training.
5. WHEN training completes, THE Trainer SHALL evaluate SignNet on the test set and print the per-class accuracy alongside the overall accuracy; if evaluation is skipped or fails, THE Trainer MAY still print the most recent available accuracy metrics or indicate that evaluation was not completed.
6. WHEN training completes, THE Trainer SHALL save the model checkpoint to `models/signnet_best.pth`, overwriting any previous checkpoint only if the new model achieves test accuracy greater than or equal to the previous checkpoint's accuracy (including when both are 0%).
7. IF the `models/` directory does not exist, THEN THE Trainer SHALL create it before saving the checkpoint.
8. THE Trainer SHALL log training loss and validation accuracy for each epoch to a file at `logs/training_log.csv` with at minimum the columns: epoch, train_loss, val_accuracy.
9. THE Trainer SHALL achieve a test set accuracy of at least 95% on the Sign Language MNIST test set.
10. THE Preprocessor SHALL format all dataset images as float32 tensors of shape (1, 28, 28) before passing them to SignNet.
11. IF the `logs/` directory does not exist, THEN THE Trainer SHALL create it before writing the training log.
12. THE Trainer SHALL run for a configurable number of epochs (default: 20) as specified in `config.yaml`, and training SHALL terminate after that many epochs have completed.

---

### Requirement 2: Improved SignNet Architecture

**User Story:** As a developer, I want an improved CNN architecture for SignNet, so that the model generalizes better and exceeds the baseline accuracy.

#### Acceptance Criteria

1. THE SignNet SHALL accept input tensors of shape (batch_size, 1, 28, 28) and output logits of shape (batch_size, 24), corresponding to the 24 static ASL letter classes (A–Z excluding J and Z).
2. THE SignNet SHALL include at least three convolutional blocks, each containing a Conv2d layer, Batch Normalization, ReLU activation, and MaxPooling.
3. THE SignNet SHALL include at least one Dropout layer with a rate in the range [0.3, 0.5] in the fully connected classifier section.
4. THE SignNet SHALL use Batch Normalization after each Conv2d layer.
5. WHEN given an input tensor of shape (batch_size, 1, 28, 28) with float32 values in [0.0, 1.0], THE SignNet SHALL produce a deterministic output tensor of shape (batch_size, 24) in evaluation mode (`model.eval()`).
6. THE Trainer SHALL use the Adam optimizer with weight decay (L2 regularization) of at least 1e-4 when training SignNet.
7. THE SignNet's fully connected classifier SHALL contain at least one hidden linear layer with ReLU activation between the flattened convolutional output and the final 24-class output layer.

---

### Requirement 3: Model Inference and Prediction

**User Story:** As a developer, I want a reliable inference interface for SignNet, so that both the Recognizer and Avatar subsystems can obtain predictions consistently.

#### Acceptance Criteria

1. THE Preprocessor SHALL resize any input image to 28×28 pixels and convert it to a single-channel grayscale float32 tensor before inference.
2. WHEN an image tensor of shape (1, 1, 28, 28) is passed to SignNet in evaluation mode, THE SignNet SHALL return the predicted class index (0–23) and the associated Confidence Score computed via softmax.
3. WHEN the Confidence Score for the top prediction is below 0.60, THE Recognizer SHALL display "Low Confidence" instead of the predicted letter.
4. THE Preprocessor SHALL apply the same normalization (pixel values divided by 255.0) during inference as during training.
5. WHEN a non-null single-channel 28×28 float32 tensor with pixel values in [0.0, 1.0] is passed through the Preprocessor and then SignNet, THE system SHALL produce a class index in the range [0, 23].
6. IF the tensor passed to SignNet does not conform to shape (batch_size, 1, 28, 28), THEN SignNet SHALL raise a ValueError indicating invalid input shape without producing a class index or Confidence Score.

---

### Requirement 4: Sign-to-Text Recognition (Live Camera Mode)

**User Story:** As a user, I want to hold my hand in front of a webcam and see the recognized ASL letter or word displayed on screen in real time, so that I can communicate using sign language without typing.

#### Acceptance Criteria

1. WHEN the user activates Recognition Mode, THE Recognizer SHALL open the default system webcam using OpenCV and begin capturing frames at a minimum of 15 frames per second.
2. WHILE Recognition Mode is active, THE Recognizer SHALL display a fixed ROI rectangle of 300×300 pixels centered in the video frame on the live video feed.
3. WHEN a frame is captured, THE Preprocessor SHALL extract the ROI, convert it to grayscale, resize it to 28×28 pixels, and normalize it before passing it to SignNet.
4. WHEN SignNet returns a prediction with Confidence Score ≥ 0.60, THE Recognizer SHALL display the predicted letter overlaid on the video feed within 200ms of frame capture.
5. WHEN the user presses the spacebar, THE Recognizer SHALL append a space to the Sentence Buffer.
6. WHEN the user presses the Enter key, THE Recognizer SHALL finalize the current word in the Sentence Buffer and display the complete sentence.
7. WHEN the user presses the Backspace key, THE Recognizer SHALL remove the last character from the Sentence Buffer.
8. WHEN the user presses 'Q' or closes the window, THE Recognizer SHALL release the webcam and close the video feed; in other situations, closing the video feed alone does not require webcam release.
9. IF the webcam cannot be opened, THEN THE Recognizer SHALL attempt to display an error message stating "Camera not available" and return to the main menu without crashing; if the error display itself fails, THE Recognizer SHALL continue with recognition mode in a blank or default state.
10. WHILE Recognition Mode is active, THE Recognizer SHALL display the current contents of the Sentence Buffer on screen.
11. WHEN SignNet returns a prediction with Confidence Score below 0.60, THE Recognizer SHALL display "Low Confidence" overlaid on the video feed instead of a letter.
12. WHEN a Recognition Mode session starts, THE Sentence Buffer SHALL be initialized to an empty string.

---

### Requirement 5: Text-to-Sign Avatar Display

**User Story:** As a user, I want to type a word or sentence and see an avatar animate the corresponding ASL signs, so that I can learn or communicate sign language visually.

#### Acceptance Criteria

1. WHEN the user activates Avatar Mode, THE Avatar SHALL display a text input field and a "Play" button.
2. WHEN the user clicks the "Play" button with non-empty text in the input field, THE Avatar SHALL display the corresponding ASL gesture image for each letter in sequence.
3. THE Avatar SHALL normalize all input text to uppercase before processing.
4. THE Avatar SHALL display each letter's gesture image for a duration between 800ms and 1500ms before advancing to the next letter.
5. WHEN the input text contains a space character, THE Avatar SHALL display a blank placeholder frame for 400ms before continuing.
6. WHEN the input text contains a character that has no corresponding ASL static gesture (J, Z, or non-alphabetic characters), THE Avatar SHALL display a "No Sign Available" placeholder image for 800ms and continue to the next character.
7. WHILE the Avatar is animating, THE Avatar SHALL display the current letter being signed and its position in the sequence (e.g., "A (1/5)") as an overlay on the gesture image.
8. WHEN the user clicks a "Pause" button during playback, THE Avatar SHALL pause the animation at the current letter.
9. WHEN the user clicks a "Resume" button while paused, THE Avatar SHALL continue the animation from the paused letter.
10. WHEN the user clicks a "Replay" button, THE Avatar SHALL restart the animation from the first letter of the current input.
11. THE Avatar SHALL accept input text of up to 200 characters; WHEN the input field contains 200 characters, THE Avatar SHALL display a visible "Character limit reached" indicator, hard-stop at 200 characters, and ignore any excess characters beyond the limit.
12. WHEN the animation completes all letters, THE Avatar SHALL display a "Done" message and re-enable the text input field for new input.

---

### Requirement 6: Gesture Image Assets

**User Story:** As a developer, I want a complete set of ASL reference images for all 24 static letters, so that the Avatar can display accurate gesture visuals.

#### Acceptance Criteria

1. THE Avatar SHALL load gesture images from the `assets/signs/` directory, with one image per letter named `A.png` through `Z.png` (excluding `J.png` and `Z.png` which are not static gestures).
2. WHEN an expected gesture image file is missing from `assets/signs/`, THE Avatar SHALL log a warning to the application log AND display a "No Sign Available" placeholder image in its place; if either logging or placeholder display cannot be completed, THE system SHALL fail rather than silently degrade.
3. THE Avatar SHALL resize all loaded gesture images to exactly 400×400 pixels before rendering.
4. THE Avatar SHALL support gesture images in PNG format and JPEG format (`.jpg` or `.jpeg` extensions).
5. WHEN the application starts and the `assets/signs/` directory contains zero image files, THE Avatar SHALL display a setup instruction message: "Please add ASL gesture images to the assets/signs/ directory."

---

### Requirement 7: Graphical User Interface

**User Story:** As a user, I want a unified GUI that lets me switch between Sign-to-Text and Text-to-Sign modes, so that I can use both features from a single application window.

#### Acceptance Criteria

1. THE GUI SHALL display a main menu with two buttons labeled exactly "Sign Recognition (Camera)" and "Sign Avatar (Text-to-Sign)".
2. WHEN the user clicks "Sign Recognition (Camera)", THE GUI SHALL transition to the Recognizer view and display the live camera feed within the application window.
3. WHEN the user clicks "Sign Avatar (Text-to-Sign)", THE GUI SHALL transition to the Avatar view displaying a text input field and an animation panel.
4. THE GUI SHALL provide a "Back to Menu" button in both the Recognizer view and the Avatar view that returns the user to the main menu without closing the application window.
5. THE GUI SHALL display the text "ASL Sign Language System" in the window title bar.
6. WHILE the Recognizer is active, THE GUI SHALL display the Sentence Buffer contents and the current predicted letter in a dedicated read-only text area.
7. THE GUI SHALL be implemented using Tkinter or PyQt5 and SHALL NOT require a web browser to operate.
8. IF an unhandled exception propagates to the GUI event loop, THEN THE GUI SHALL display a modal error dialog containing the exception message and return the user to the main menu without terminating the process.
9. THE GUI MAY display error dialogs during normal operation even when no exception has occurred.

---

### Requirement 8: Project Structure and Configuration

**User Story:** As a developer, I want a well-organized project structure with a configuration file, so that paths, hyperparameters, and settings are easy to find and modify.

#### Acceptance Criteria

1. THE system SHALL organize source code into the following directory structure:
   - `src/` — all Python source modules
   - `models/` — saved model checkpoints
   - `assets/signs/` — ASL gesture images
   - `logs/` — training logs
   - `archive/` — dataset CSV files (existing)
2. THE system SHALL provide a `config.yaml` file at the project root containing all configurable parameters: dataset paths, model checkpoint path, ROI coordinates, confidence threshold, avatar display duration (min and max ms), and training hyperparameters (epochs, batch size, learning rate, weight decay).
3. WHEN any module reads a configurable parameter, THE module SHALL read it from `config.yaml`; all required parameter keys MUST be present in `config.yaml`, and the module SHALL NOT silently fall back to hardcoded default values if a parameter is missing.
4. THE system SHALL provide a `requirements.txt` file listing all Python dependencies with pinned exact versions (e.g., `torch==2.1.0`).
5. THE system SHALL provide a `main.py` entry point at the project root that, when executed with `python main.py`, launches the GUI.

---

### Requirement 9: Model Persistence and Loading

**User Story:** As a user, I want the application to load a pre-trained model on startup without retraining, so that I can use the system immediately.

#### Acceptance Criteria

1. WHEN the application starts, THE GUI SHALL check for the existence of the file at the checkpoint path specified in `config.yaml` (default: `models/signnet_best.pth`).
2. IF the checkpoint file exists, THEN THE Recognizer SHALL load the checkpoint and initialize SignNet weights from it before opening the camera.
3. IF the checkpoint file does not exist, THEN THE GUI SHALL display the message "No trained model found. Please run train.py first." and disable the "Sign Recognition (Camera)" button.
4. WHEN loading a checkpoint, THE system SHALL use CUDA if available, otherwise CPU, and SHALL map all tensor operations to the selected device automatically.
5. THE Trainer SHALL print a line stating the device being used (e.g., "Using device: cuda" or "Using device: cpu") at the start of both training and inference.

---

### Requirement 10: Documentation

**User Story:** As a developer or end user, I want comprehensive documentation, so that I can set up, train, and use the system without external assistance.

#### Acceptance Criteria

1. THE system SHALL include a `README.md` file at the project root covering: project overview, prerequisites (Python version, OS), installation steps, dataset setup instructions, training instructions, and usage instructions for both Sign Recognition and Avatar modes.
2. THE README.md SHALL include a section titled "Model Architecture" describing the SignNet layers and the accuracy achieved on the test set.
3. THE README.md SHALL include a section titled "Troubleshooting" with entries for at least: missing model checkpoint, camera not found, and missing gesture image assets.
4. THE system SHALL include Google-style docstrings for all public functions and classes in the `src/` directory.
5. THE README.md SHALL include a section titled "Project Structure" listing the top-level directories and their purpose.
