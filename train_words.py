"""Command-line training entry point for word-level Sign-to-Text."""

from src.config import load_config
from src.word_trainer import WordTrainer


def main() -> None:
    """Train the word-level video classifier."""
    WordTrainer(load_config("config.yaml")).train()


if __name__ == "__main__":
    main()
