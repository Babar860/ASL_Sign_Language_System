# ASL Sign Language System

## Project overview

This is a local desktop application for two-way American Sign Language workflows. It trains a PyTorch CNN on Sign Language MNIST, recognizes static ASL letters from a webcam ROI, and animates typed text as letter-by-letter ASL gesture images.

## Prerequisites (Python version, OS)

Use Python 3.10 or 3.11 on Windows, macOS, or Linux. A webcam is required for Sign Recognition mode. CUDA is used automatically when PyTorch detects it; otherwise the app runs on CPU.

## Installation steps

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Dataset setup instructions

The repository expects Sign Language MNIST CSV files at:

- `archive/sign_mnist_train/sign_mnist_train.csv`
- `archive/sign_mnist_test/sign_mnist_test.csv`

Text-to-Sign assets live in `assets/text_to_sign/`. Letter images belong in `assets/text_to_sign/signs/`, named `A.png` through `Y.png`, excluding `J.png` and `Z.png`. Word or phrase avatar videos belong in `assets/text_to_sign/words/` as MP4 files named after the text, for example `hello.mp4` or `I am fine.mp4`.

Sign-to-Text word video samples are kept separately in `assets/sign_to_text/words/`. Use `python scripts/prepare_wlasl_subset.py --download` to inspect the WLASL processed dataset metadata and extract only videos whose gloss matches the current Text-to-Sign MP4 vocabulary. The script writes `assets/sign_to_text/word_manifest.csv`, which is used by `python train_words.py` to train the word-level CNN checkpoint at `models/word_signnet_best.pth`.

## Training instructions (`python train.py`)

Run:

```powershell
python train.py
```

Training reads `config.yaml`, logs per-epoch metrics to `logs/training_log.csv`, evaluates per-class and overall accuracy, and saves the best checkpoint to `models/signnet_best.pth` when the new accuracy is greater than or equal to the previous checkpoint accuracy.

## Usage instructions for Sign Recognition mode and Avatar mode

Launch the GUI:

```powershell
python main.py
```

Sign Recognition mode is enabled only when `models/signnet_best.pth` exists. Place your hand inside the 300x300 ROI. The current prediction and sentence buffer are displayed in the app. Use Space for a space, Enter to finalize a word, Backspace to delete, and Q or Back to Menu to close the video feed.

Avatar mode accepts up to 200 characters. If the input has a matching MP4 in `assets/text_to_sign/words/`, it plays that avatar video. Otherwise it normalizes input to uppercase and plays static gesture images one at a time. Pause, Resume, and Replay controls are available during playback.

## Model Architecture

SignNet accepts `(batch_size, 1, 28, 28)` grayscale tensors and outputs 24 logits for static ASL letters A-Z excluding J and Z. It uses three convolution blocks, each with Conv2d, BatchNorm2d, ReLU, and MaxPool2d, followed by a classifier with two hidden Linear layers and Dropout. The training target is at least 95% test accuracy on Sign Language MNIST.

## Project Structure

- `src/`: Python source modules for model, preprocessing, training, recognition, avatar, GUI, config, and logging.
- `models/`: saved model checkpoints.
- `assets/text_to_sign/signs/`: ASL gesture images for letter-by-letter Avatar mode.
- `assets/text_to_sign/words/`: MP4 avatar videos for word or phrase Text-to-Sign playback.
- `assets/sign_to_text/words/`: WLASL word-video samples filtered to the supported Text-to-Sign vocabulary.
- `logs/`: training logs.
- `archive/`: Sign Language MNIST CSV data.
- `.kiro/`: requirements, design, and implementation planning documents.

## Troubleshooting

Missing model checkpoint: run `python train.py`. Until `models/signnet_best.pth` exists, the Sign Recognition button remains disabled.

Camera not found: confirm the webcam is connected, not already in use, and that `inference.camera_index` in `config.yaml` matches the camera device.

Missing gesture image assets: add PNG or JPG files to `assets/text_to_sign/signs/`. If the directory is empty, Avatar mode shows a setup instruction. Missing individual letters display a placeholder and log a warning.
