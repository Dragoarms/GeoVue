from __future__ import annotations

from dataclasses import dataclass, field
import io
from pathlib import Path
from unittest.mock import patch

from ml_pipeline.lithology_manifest import (
    build_manifest_rows_from_images,
    load_manifest_csv,
    normalise_label,
    normalise_tags,
    summarise_rows,
    write_manifest_csv,
)


@dataclass
class FakeImage:
    image_path: str
    classification: str
    hole_id: str = "BA0001"
    depth_from: float = 0.0
    depth_to: float = 1.0
    moisture_status: str = "Dry"
    tags: set[str] = field(default_factory=set)
    consensus: str = ""
    peer_tags: set[str] = field(default_factory=set)


def _image(name: str) -> str:
    return str(Path("C:/fake/geovue") / name)


def test_normalise_label_matches_case_and_punctuation_variants():
    labels = ["BIFf", "BIFf-s", "BIFhm"]

    assert normalise_label("BIFHm", labels) == "BIFhm"
    assert normalise_label("biff_s", labels) == "BIFf-s"
    assert normalise_label("ClassificationCategory.UNASSIGNED", labels) is None


def test_normalise_tags_handles_review_storage_shapes():
    assert normalise_tags("['jasperlitic', 'gold_assays']") == ("gold_assays", "jasperlitic")
    assert normalise_tags("jasperlitic|not_gold_assays") == ("jasperlitic", "not_gold_assays")
    assert normalise_tags({"gold_assays": True, "jasperlitic": False}) == ("gold_assays",)


def test_build_manifest_uses_consensus_and_aggregated_tags():
    images = [
        FakeImage(
            _image("BA0001_CC_001_Dry.png"),
            "Other",
            consensus="BIFHm",
            tags={"current_tag"},
            peer_tags={"peer_tag"},
        ),
        FakeImage(_image("BA0001_CC_002_Dry.png"), "BIFf-s", depth_to=2.0),
        FakeImage(_image("BA0001_CC_003_Dry.png"), "Greenstone", depth_to=3.0),
    ]

    rows = build_manifest_rows_from_images(
        images,
        ["BIFf", "BIFf-s", "BIFhm"],
        consensus_fn=lambda img: img.consensus,
        metadata_fn=lambda img: {"tags": img.tags | img.peer_tags},
        include_missing_images=True,
    )

    assert [row.label for row in rows] == ["BIFhm", "BIFf-s"]
    assert rows[0].source == "consensus"
    assert rows[0].tags == ("current_tag", "peer_tag")

    summary = summarise_rows(rows)
    assert summary["class_counts"] == {"BIFf-s": 1, "BIFhm": 1}
    assert summary["tag_counts"] == {"current_tag": 1, "peer_tag": 1}


def test_build_manifest_can_cap_each_class_deterministically():
    images = []
    for idx in range(5):
        images.append(
            FakeImage(
                _image(f"BA0001_CC_{idx + 1:03d}_Dry.png"),
                "BIFf",
                depth_to=float(idx + 1),
            )
        )
    for idx in range(4):
        images.append(
            FakeImage(
                _image(f"BA0002_CC_{idx + 1:03d}_Dry.png"),
                "BIFf-s",
                hole_id="BA0002",
                depth_to=float(idx + 1),
            )
        )

    rows = build_manifest_rows_from_images(
        images,
        ["BIFf", "BIFf-s"],
        max_per_class=2,
        seed=7,
        include_missing_images=True,
    )

    summary = summarise_rows(rows)
    assert summary["class_counts"] == {"BIFf": 2, "BIFf-s": 2}


def test_write_and_load_manifest_round_trip():
    image = FakeImage(_image("BA0001_CC_001_Dry.png"), "BIFf", tags={"jasperlitic"})
    rows = build_manifest_rows_from_images([image], ["BIFf"], include_missing_images=True)
    memory: dict[str, str] = {}

    class MemoryWriter(io.StringIO):
        def __init__(self, key: str):
            super().__init__()
            self.key = key

        def close(self):
            memory[self.key] = self.getvalue()
            super().close()

    def fake_open(path, mode="r", *args, **kwargs):
        key = str(path)
        if "w" in mode:
            return MemoryWriter(key)
        return io.StringIO(memory[key])

    with patch.object(Path, "mkdir", lambda self, parents=False, exist_ok=False: None):
        with patch.object(Path, "open", fake_open):
            path = write_manifest_csv(rows, Path("memory_manifest.csv"))
            loaded = load_manifest_csv(path)

    assert loaded == rows
