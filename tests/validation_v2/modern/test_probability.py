import numpy as np

from validation_v2.modern.probability import (
    empirical_crps,
    interval_metrics,
    stitch_samples,
)


def test_stitch_samples_averages_each_sample_before_quantiles():
    windows = np.array(
        [
            [[[[0.0], [2.0], [4.0]]], [[[10.0], [12.0], [14.0]]]],
            [[[[6.0], [8.0], [10.0]]], [[[16.0], [18.0], [20.0]]]],
        ]
    ).reshape(2, 2, 3, 1)
    result = stitch_samples(windows, starts=(0, 2), length=5)
    np.testing.assert_allclose(
        result[:, :, 0], [[0, 2, 5, 8, 10], [10, 12, 15, 18, 20]]
    )


def test_empirical_crps_matches_two_sample_closed_form():
    samples = np.array([[[0.0]], [[2.0]]])
    target = np.array([[1.0]])
    mask = np.array([[0]], dtype=np.uint8)
    assert empirical_crps(samples, target, mask) == 0.5


def test_interval_metrics_report_coverage_and_width():
    samples = np.array([[[-1.0]], [[0.0]], [[1.0]]])
    coverage, width = interval_metrics(
        samples, np.array([[0.5]]), np.array([[0]]), level=0.95
    )
    assert coverage == 1.0
    assert width > 1.0
