"""Shared filesystem paths for the backend package."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent

CONFIG_PATH = PROJECT_ROOT / "config.json"
CONFIG_EXAMPLE_PATH = PROJECT_ROOT / "configs" / "config.example.json"
