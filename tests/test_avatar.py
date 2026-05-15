from src.avatar import Avatar
from src.config import load_config
from PIL import Image


def test_avatar_hard_stops_at_limit():
    overlays = []
    config = load_config("config.yaml")
    avatar = Avatar(config, lambda image: None, overlays.append)
    avatar.play("a" * 250, lambda delay, callback: None)
    assert len(avatar.sequence) == 200


def test_avatar_space_duration():
    config = load_config("config.yaml")
    avatar = Avatar(config, lambda image: None, lambda text: None)
    _, duration = avatar._image_and_duration(" ")
    assert duration == 400


def test_avatar_finds_lowercase_letter_image(tmp_path):
    config = load_config("config.yaml")
    config["avatar"]["signs_dir"] = str(tmp_path)
    Image.new("RGB", (20, 20), "white").save(tmp_path / "h.png")
    avatar = Avatar(config, lambda image: None, lambda text: None)
    assert avatar._images["H"].size == (400, 400)
