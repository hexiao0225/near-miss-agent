"""Offline checks: python3 test_main.py"""

import main

VSS_RESPONSE = {
    "results": [
        {"source": "s3://vss-chunks-segments/team-5/a_seg3.mp4", "similarity_score": 0.41,
         "reasoning_content": "A pickup turns while a pedestrian steps back to the curb.",
         "camera_id": "sf_streets_cam-1", "location": "san_francisco",
         "tags": ["person 10", "truck 2", "car"]},
    ],
    "chunk_results": [
        {"original_video": "s3://vss-chunks/team-5/a.mp4", "preview_source": "s3://vss-chunks-segments/team-5/a_seg3.mp4",
         "best_match_start_sec": 10, "best_match_end_sec": 15, "matched_segment_count": 1},
    ],
}


def test_normalise():
    [hit] = main.normalise(VSS_RESPONSE, "q")
    assert hit["camera_id"] == "sf_streets_cam-1"
    assert hit["start"] == 10 and hit["end"] == 15
    assert hit["score"] == 0.41
    assert hit["objects"] == {"person": 10, "truck": 2, "car": 1}
    assert "pedestrian" in hit["caption"]


def test_heuristic():
    [hit] = main.normalise(VSS_RESPONSE, "q")
    assert main.heuristic(hit)["risk"] == "high"
    calm = dict(hit, caption="Dense highway traffic; no pedestrians visible.", objects={"car": 30})
    assert main.heuristic(calm)["risk"] == "none"


def test_parse_json():
    assert main._parse_json('Sure! {"risk": "high", "near_miss": true}')["risk"] == "high"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
