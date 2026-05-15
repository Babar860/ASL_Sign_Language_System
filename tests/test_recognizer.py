import numpy as np

from src.config import load_config
from src.model import SignNet
from src.recognizer import Recognizer


def test_roi_is_centered_and_sized():
    config = load_config("config.yaml")
    recognizer = Recognizer(config, SignNet())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    roi, coords = recognizer.extract_roi(frame)
    assert roi.shape == (300, 300, 3)
    assert coords == (170, 90, 470, 390)


def test_sentence_buffer_operations():
    config = load_config("config.yaml")
    recognizer = Recognizer(config, SignNet())
    recognizer.current_prediction = "H"
    assert recognizer.show_current_prediction() is True
    recognizer.append_letter("A")
    recognizer.append_space()
    recognizer.append_letter("B")
    recognizer.backspace()
    assert recognizer.sentence_buffer == "HA "
    recognizer.clear()
    assert recognizer.sentence_buffer == ""


def test_show_current_prediction_ignores_low_confidence():
    config = load_config("config.yaml")
    recognizer = Recognizer(config, SignNet())
    recognizer.current_prediction = "Low Confidence"
    assert recognizer.show_current_prediction() is False
    assert recognizer.sentence_buffer == ""
