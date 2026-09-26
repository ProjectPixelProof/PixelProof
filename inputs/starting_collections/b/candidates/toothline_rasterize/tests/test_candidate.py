from pathlib import Path

from PIL import Image

from oracle import decision_from_image, measure_from_image
from renderer import analytic_gold, margin, render, sample_scene
from verify import verify


ROOT = Path(__file__).resolve().parents[1]


def test_exact_evidence():
    verify(ROOT / "evidence")


def test_balanced_pixel_inverse():
    labels = []
    for seed in range(101, 117):
        scene = sample_scene(seed)
        image = render(scene)
        assert decision_from_image(image) == analytic_gold(scene)
        assert measure_from_image(image)["runner_up_margin"] >= 2
        assert margin(scene) >= 2
        labels.append(analytic_gold(scene))
    assert {name: labels.count(name) for name in set(labels)} == {name: 4 for name in set(labels)}


def test_blank_abstains():
    try:
        decision_from_image(Image.new("RGB", (512, 512), "white"))
    except ValueError:
        return
    raise AssertionError("blank image did not force oracle abstention")
