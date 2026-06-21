from __future__ import annotations

import pytest

from ml_pipeline.similarity_search import (
    HybridSimilarityRanker,
    SimilarityCandidate,
    SimilaritySearchConfig,
    build_chemical_space,
    clr_transform,
    spatial_similarity,
)


def candidate(path, hole, depth_from, depth_to, chemistry=None, xyz=None):
    return SimilarityCandidate(
        image_path=path,
        hole_id=hole,
        depth_from=depth_from,
        depth_to=depth_to,
        chemistry=chemistry or {},
        xyz=xyz,
    )


def test_clr_transform_keeps_zero_handling_finite_and_compositional():
    import pandas as pd

    df = pd.DataFrame(
        {
            "Fe_pct_BEST": [60.0, 60.0, 60.0],
            "SiO2_pct_BEST": [5.0, 0.0, 10.0],
            "Al2O3_pct_BEST": [2.0, 2.0, 2.0],
        }
    )

    transformed = clr_transform(df)

    assert transformed.notna().all().all()
    # CLR rows sum to approximately zero across available components.
    assert transformed.sum(axis=1).abs().max() < 1e-9


def test_chemical_similarity_uses_aliases_and_ranks_near_assays_first():
    query = candidate("C:/chips/q.png", "BA0001", 0, 1, {
        "Fe_pct_BEST": 60.0,
        "SiO2_pct_BEST": 5.0,
        "Al2O3_pct_BEST": 2.0,
    })
    near = candidate("C:/chips/near.png", "BA0002", 0, 1, {
        "Fe_pct_BEST": 59.5,
        "SiO2_pct_BEST": 5.3,
        "Al2O3_pct_BEST": 2.1,
    })
    far = candidate("C:/chips/far.png", "BA0003", 0, 1, {
        "Fe_pct_BEST": 35.0,
        "SiO2_pct_BEST": 35.0,
        "Al2O3_pct_BEST": 10.0,
    })
    config = SimilaritySearchConfig(mode="chemical")

    results = HybridSimilarityRanker([query, far, near], config=config).rank(query)

    assert [result.candidate.image_path for result in results] == ["C:/chips/near.png", "C:/chips/far.png"]
    assert results[0].chemical_score > results[1].chemical_score
    assert results[0].chemical_coverage == pytest.approx(1.0)


def test_chemical_similarity_refuses_low_dimension_coverage():
    query = candidate("q.png", "BA0001", 0, 1, {"Fe_pct_BEST": 60.0, "SiO2_pct_BEST": 5.0})
    sparse = candidate("sparse.png", "BA0002", 0, 1, {"Fe_pct_BEST": 60.2})
    config = SimilaritySearchConfig(
        mode="chemical",
        chemistry_columns=("fe_pct", "sio2_pct"),
        min_chemical_coverage=0.75,
    )

    results = HybridSimilarityRanker([query, sparse], config=config).rank(query)

    assert results == []


def test_spatial_similarity_prefers_surveyed_xyz_over_depth_fallback():
    query = candidate("q.png", "BA0001", 0, 1, xyz=(100.0, 100.0, 50.0))
    nearby_other_hole = candidate("near.png", "BA0002", 100, 101, xyz=(103.0, 104.0, 50.0))
    same_hole_far_xyz = candidate("far.png", "BA0001", 1, 2, xyz=(200.0, 200.0, 50.0))
    config = SimilaritySearchConfig(mode="spatial", spatial_range_m=10.0, depth_range_m=10.0, use_xyz=True)

    near_score, near_dist, _ = spatial_similarity(query, nearby_other_hole, config)
    far_score, far_dist, _ = spatial_similarity(query, same_hole_far_xyz, config)

    assert near_dist == pytest.approx(5.0)
    assert far_dist > 100.0
    assert near_score > far_score


def test_spatial_similarity_falls_back_to_same_hole_depth_when_xyz_missing():
    query = candidate("q.png", "BA0001", 10, 11)
    same_hole = candidate("same.png", "BA0001", 12, 13)
    other_hole = candidate("other.png", "BA0002", 12, 13)
    config = SimilaritySearchConfig(mode="spatial", depth_range_m=5.0, use_xyz=True)

    same_score, same_dist, same_depth = spatial_similarity(query, same_hole, config)
    other_score, other_dist, other_depth = spatial_similarity(query, other_hole, config)

    assert same_dist is None
    assert same_depth == pytest.approx(2.0)
    assert same_score is not None and same_score > 0
    assert other_score is None
    assert other_dist is None
    assert other_depth == pytest.approx(2.0)


def test_continuity_rewards_candidate_neighbourhoods_that_match_query():
    query = candidate("q.png", "BA0001", 10, 11, {"Fe_pct_BEST": 60, "SiO2_pct_BEST": 5, "Al2O3_pct_BEST": 2})
    isolated = candidate("isolated.png", "BA0002", 10, 11, {"Fe_pct_BEST": 60, "SiO2_pct_BEST": 5, "Al2O3_pct_BEST": 2})
    isolated_bad_neighbour = candidate("isolated_n.png", "BA0002", 11, 12, {"Fe_pct_BEST": 35, "SiO2_pct_BEST": 35, "Al2O3_pct_BEST": 10})
    continuous = candidate("continuous.png", "BA0003", 10, 11, {"Fe_pct_BEST": 60, "SiO2_pct_BEST": 5, "Al2O3_pct_BEST": 2})
    continuous_neighbour = candidate("continuous_n.png", "BA0003", 11, 12, {"Fe_pct_BEST": 59, "SiO2_pct_BEST": 5.5, "Al2O3_pct_BEST": 2.2})
    visual = {
        query.image_path: 1.0,
        isolated.image_path: 0.95,
        isolated_bad_neighbour.image_path: 0.05,
        continuous.image_path: 0.90,
        continuous_neighbour.image_path: 0.88,
    }
    config = SimilaritySearchConfig(mode="continuity", continuity_window_m=2.0, use_xyz=False)

    results = HybridSimilarityRanker(
        [query, isolated, isolated_bad_neighbour, continuous, continuous_neighbour],
        visual_scores_by_key=visual,
        config=config,
    ).rank(query)
    by_path = {result.candidate.image_path: result for result in results}

    assert by_path["continuous.png"].continuity_score > by_path["isolated.png"].continuity_score
    assert by_path["continuous.png"].combined_score > by_path["isolated.png"].combined_score


def test_build_chemical_space_returns_empty_when_no_requested_columns():
    rows = [candidate("a.png", "BA0001", 0, 1, {"not_assay": 1})]
    space = build_chemical_space(rows, SimilaritySearchConfig(chemistry_columns=("fe_pct",)))

    assert space.columns == ()
    assert space.matrix.shape == (1, 0)
