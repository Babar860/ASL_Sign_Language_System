import numpy as np

from src.preprocessor import Preprocessor


def test_prepare_frame_shape_and_range():
    tensor = Preprocessor.prepare_frame(np.full((300, 300, 3), 128, dtype=np.uint8))
    assert tuple(tensor.shape) == (1, 1, 28, 28)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_prepare_dataset_image_shape():
    tensor = Preprocessor.prepare_dataset_image(np.zeros((28, 28), dtype=np.uint8))
    assert tuple(tensor.shape) == (1, 28, 28)


def test_prepare_word_frames_shape():
    frames = [np.zeros((80, 80, 3), dtype=np.uint8) for _ in range(2)]
    tensor = Preprocessor.prepare_word_frames(frames, frame_count=4, image_size=32)
    assert tuple(tensor.shape) == (4, 1, 32, 32)
