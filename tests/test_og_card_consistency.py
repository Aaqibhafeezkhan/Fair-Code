"""Keep the generated benchmark card aligned with the benchmark definitions."""

import importlib
import pytest

pytest.importorskip("sklearn", reason="benchmark definitions require the benchmark extra")
pytest.importorskip("fairlearn", reason="benchmark definitions require the benchmark extra")

from faircode.models import MODEL_FAMILIES  # noqa: E402
from faircode.strategies import STRATEGIES  # noqa: E402
generate_og_images = importlib.import_module("scripts.generate_og_images")


def test_og_card_copy_tracks_audit_and_benchmark_definitions():
    audits = list(generate_og_images.ROOT.glob("*/audit.yaml"))
    count_words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                   6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    subtitle = generate_og_images.benchmark_card_subtitle()
    assert f"{count_words[len(audits)]} audits" in subtitle
    assert f"{count_words[len(STRATEGIES)]} mitigation strategies" in subtitle
    assert f"{count_words[len(MODEL_FAMILIES)]} model families" in subtitle
    assert "real-world domains" in generate_og_images.home_card_subtitle()
    assert "criminal justice, hiring, lending" not in generate_og_images.home_card_subtitle()
