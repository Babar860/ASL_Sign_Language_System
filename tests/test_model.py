import pytest
import torch

from src.model import SignNet
from src.keypoint_model import KEYPOINT_DIM, KeypointWordNet
from src.word_model import WordSignNet


def test_signnet_output_shape():
    model = SignNet().eval()
    assert tuple(model(torch.rand(4, 1, 28, 28)).shape) == (4, 24)


def test_signnet_eval_determinism():
    model = SignNet().eval()
    x = torch.rand(2, 1, 28, 28)
    assert torch.equal(model(x), model(x))


def test_predict_bounds():
    model = SignNet().eval()
    index, confidence = model.predict(torch.rand(1, 1, 28, 28))
    assert 0 <= index <= 23
    assert 0.0 <= confidence <= 1.0


def test_bad_shape_rejected():
    with pytest.raises(ValueError):
        SignNet()(torch.rand(1, 3, 28, 28))


def test_word_signnet_output_shape():
    model = WordSignNet(num_classes=3).eval()
    assert tuple(model(torch.rand(2, 4, 1, 64, 64)).shape) == (2, 3)


def test_keypoint_wordnet_output_shape():
    model = KeypointWordNet(num_classes=5).eval()
    assert tuple(model(torch.rand(2, 16, KEYPOINT_DIM)).shape) == (2, 5)
