from processing import logging_review_html_report as html_report


class _FakeImageCoordinator:
    def __init__(self, image_path: str = r"C:\images\wet tray 010.png"):
        self.image_path = image_path
        self.calls = []

    def get_image_path(self, key):
        self.calls.append(key)
        return self.image_path

    def get_keys_for_hole(self, hole_id):
        return []


def test_thumbnail_mode_returns_relative_deduped_asset_urls(monkeypatch):
    coordinator = _FakeImageCoordinator()
    thumbnail_writes = []

    def fake_write_thumbnail(path, output_dir, relative_dir, asset_cache=None, **kwargs):
        if asset_cache is not None and path in asset_cache:
            return asset_cache[path]
        thumbnail_writes.append(path)
        url = "assets/Logger_A/images/wet_tray_010_thumb.jpg"
        if asset_cache is not None:
            asset_cache[path] = url
        return url

    monkeypatch.setattr(html_report, "_write_thumbnail_asset", fake_write_thumbnail)

    result = html_report._build_intervals_with_images(
        coordinator,
        [
            {"hole_id": "h1", "depth_to": 10.0},
            {"hole_id": "H1", "depth_to": 10},
            {"hole_id": "H1", "depth_to": 11.0},
        ],
        include_images=True,
        image_cache={},
        image_mode="thumbnail",
        image_output_dir=r"C:\reports\assets\Logger_A\images",
        image_relative_dir="assets/Logger_A/images",
        image_asset_cache={},
    )

    image_urls = [item["image"] for item in result]
    assert image_urls == [
        "assets/Logger_A/images/wet_tray_010_thumb.jpg",
        "assets/Logger_A/images/wet_tray_010_thumb.jpg",
        "assets/Logger_A/images/wet_tray_010_thumb.jpg",
    ]
    assert len(coordinator.calls) == 2
    assert thumbnail_writes == [coordinator.image_path]


def test_none_image_mode_skips_lookup():
    coordinator = _FakeImageCoordinator()

    result = html_report._build_intervals_with_images(
        coordinator,
        [{"hole_id": "H1", "depth_to": 10.0}],
        include_images=True,
        image_mode="none",
        image_cache={},
    )

    assert result[0]["image"] is None
    assert coordinator.calls == []
