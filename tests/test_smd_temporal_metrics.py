import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from smd_temporal_metrics import DT_S, reconstructed_grouped_state, temporal_metrics


def test_linear_motion_uses_released_five_over_sixty_four_timing():
    positions = np.zeros((64, 2, 2))
    positions[:, 0, 0] = np.arange(64) * .02
    positions[:, 1, 1] = np.arange(64) * .02
    grouped = reconstructed_grouped_state(positions)
    report = temporal_metrics(grouped)
    assert report["dt_s"] == 5 / 64
    assert report["last_support_time_s"] == 63 * DT_S
    np.testing.assert_allclose(grouped[0, :, 2:], 0)
    np.testing.assert_allclose(grouped[-1, :, 2:], 0)
    assert report["mean_acceleration_mps2"] < 1e-10
    assert report["stored_velocity_consistency_rms_mps"] < 1e-10
    assert abs(report["path_m"] - 1.26) < 1e-10
