from validation_v2.modern.tuning import candidates, select_candidate


def test_each_modern_model_has_four_stable_candidates():
    for model in ("brits", "saits", "csdi", "sssd"):
        values = candidates(model)
        assert len(values) == 4
        assert len({item["configuration_id"] for item in values}) == 4


def test_selection_uses_rmse_then_parameters_then_latency_then_id():
    rows = [
        {
            "configuration_id": "b",
            "missing_rmse": 0.2,
            "parameters": 20,
            "latency_s": 1.0,
        },
        {
            "configuration_id": "a",
            "missing_rmse": 0.2,
            "parameters": 10,
            "latency_s": 2.0,
        },
    ]
    assert select_candidate(rows)["configuration_id"] == "a"
