from __future__ import annotations

import logging
from pathlib import Path

from gui import logging_review_dialog as lrd
from gui.ReviewDialog.filter_pipeline import FilterResult


class FakeCriteria:
    dynamic_filters = [{"column": "hole_id", "operator": "=", "value": "BA0001"}]

    def get_hash(self):
        return "hole-filter"


class FakePipeline:
    def __init__(self, filtered_images):
        self.filtered_images = filtered_images
        self.seen_all_images = None
        self.seen_criteria = None

    def execute(self, all_images, criteria):
        self.seen_all_images = list(all_images)
        self.seen_criteria = criteria
        return FilterResult(
            images=list(self.filtered_images),
            total_before=len(all_images),
            total_after=len(self.filtered_images),
            csv_matches=0,
            filter_descriptions=["hole_id = BA0001"],
            execution_time_ms=1.0,
        )


class FakeGridCanvas:
    def __init__(self):
        self.loaded_images = None
        self.displayed_images = []
        self.canvas = FakeCanvas()

    def load_images(self, images):
        self.loaded_images = list(images)


class FakeCanvas:
    def __init__(self):
        self.cursor = ""

    def config(self, **kwargs):
        if "cursor" in kwargs:
            self.cursor = kwargs["cursor"]


def image(path, hole="BA0001", depth=1.0):
    return lrd.CompartmentImage(
        filename=path.rsplit("/", 1)[-1],
        hole_id=hole,
        depth_from=depth - 1.0,
        depth_to=depth,
        image_path=path,
    )


def bare_dialog(images):
    dialog = object.__new__(lrd.LoggingReviewDialog)
    dialog.logger = logging.getLogger("test_logging_review_similarity_bridge")
    dialog.all_images = list(images)
    dialog.displayed_images = []
    dialog.last_filter_hash = None
    dialog.grid_canvas = FakeGridCanvas()
    dialog.grid_canvas.displayed_images = list(images)
    dialog.updated_filter_descriptions = None
    dialog.last_status = None
    dialog._pending_similarity_options = None
    dialog.update_active_filters_display = lambda descriptions: setattr(
        dialog, "updated_filter_descriptions", list(descriptions)
    )
    dialog._update_statistics = lambda: None
    dialog._update_status = lambda status: setattr(dialog, "last_status", status)
    return dialog


def test_similarity_layer_respects_existing_filters_and_sorts_remaining_hits(monkeypatch):
    query = image("C:/chips/query.png", depth=1.0)
    kept_lower_rank = image("C:/chips/kept_lower.png", depth=2.0)
    kept_higher_rank = image("C:/chips/kept_higher.png", depth=3.0)
    filtered_out_best = image("C:/chips/filtered_out_best.png", "BA0002", depth=4.0)
    dialog = bare_dialog([query, kept_lower_rank, kept_higher_rank, filtered_out_best])
    dialog.filter_pipeline = FakePipeline([kept_lower_rank, kept_higher_rank])
    dialog._similarity_filter = {
        "query_path": query.image_path,
        "mode": "hybrid",
        "indexed_hits": 3,
        "review_match_count": 3,
    }
    dialog._similarity_rank_by_key = {
        dialog._image_similarity_key(filtered_out_best): 0,
        dialog._image_similarity_key(kept_higher_rank): 1,
        dialog._image_similarity_key(kept_lower_rank): 2,
    }
    dialog._similarity_score_by_key = {}
    dialog._similarity_explanation_by_key = {}
    dialog._similarity_query_path = query.image_path

    monkeypatch.setattr(lrd, "create_filter_criteria_from_dialog", lambda _dialog: FakeCriteria())

    lrd.LoggingReviewDialog._apply_filters(dialog, force=True)

    assert dialog.displayed_images == [kept_higher_rank, kept_lower_rank]
    assert dialog.grid_canvas.loaded_images == dialog.displayed_images
    assert dialog.filter_pipeline.seen_all_images == dialog.all_images
    assert dialog.updated_filter_descriptions == [
        "hole_id = BA0001",
        "Hybrid similar to query.png: 2 shown from 3 review matches (3 visual neighbours)",
    ]
    assert dialog.last_status == "Displaying 2 images | similarity: 2 -> 2"


def test_similarity_candidate_scope_is_all_loaded_images_not_current_grid():
    all_images = [
        image("C:/chips/a.png", depth=1.0),
        image("C:/chips/b.png", depth=2.0),
        image("C:/chips/c.png", depth=3.0),
    ]
    dialog = bare_dialog(all_images)
    dialog.displayed_images = [all_images[1]]

    assert lrd.LoggingReviewDialog._similarity_candidate_images(dialog) == all_images


def test_similarity_candidate_builder_hydrates_chemistry_only_when_requested():
    sample = image("C:/chips/sample.png", depth=2.0)
    dialog = bare_dialog([sample])
    calls = []
    dialog._build_similarity_xyz_cache = lambda images, use_xyz: {}
    dialog._get_csv_data_cached = lambda hole, depth: calls.append((hole, depth)) or ({"Fe_pct_BEST": 60.0}, True)
    dialog._get_classification_string = lambda classification: str(classification)

    visual_candidates = lrd.LoggingReviewDialog._build_similarity_candidates(
        dialog,
        include_chemistry=False,
        use_xyz=False,
    )

    assert calls == []
    assert visual_candidates[0].chemistry == {}

    chemical_candidates = lrd.LoggingReviewDialog._build_similarity_candidates(
        dialog,
        include_chemistry=True,
        use_xyz=False,
    )

    assert calls == [(sample.hole_id, sample.depth_to)]
    assert chemical_candidates[0].chemistry == {"Fe_pct_BEST": 60.0}
    assert sample.in_csv is True


def test_similarity_search_can_arm_next_clicked_grid_image_as_query():
    query = image("C:/chips/query.png", depth=10.0)
    dialog = bare_dialog([query])
    starts = []
    dialog._start_similarity_search = lambda query_path, selected_img, options: starts.append(
        (query_path, selected_img, options)
    )

    lrd.LoggingReviewDialog._arm_similarity_query_pick(dialog, {"mode": "hybrid"})

    assert lrd.LoggingReviewDialog._has_pending_similarity_query(dialog) is True
    assert dialog.grid_canvas.canvas.cursor == "crosshair"
    assert dialog.last_status == "Similarity hybrid search armed - click one grid image to use as the query."

    handled = lrd.LoggingReviewDialog._consume_pending_similarity_query_image(dialog, query)

    assert handled is True
    assert lrd.LoggingReviewDialog._has_pending_similarity_query(dialog) is False
    assert dialog.grid_canvas.canvas.cursor == ""
    assert starts == [(Path(query.image_path), query, {"mode": "hybrid"})]
