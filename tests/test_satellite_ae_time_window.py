import numpy as np
import pandas as pd

from appa.data.datasets import SatelliteAEDataset


def _stub_dataset():
    center = pd.Timestamp("2025-01-01T06:00:00")
    matching = (
        np.array([[[2.0, 2.0]]], dtype=np.float32),
        np.array([[[1.0, 1.0]]], dtype=np.float32),
        None,
    )
    frames = {center: matching}
    calls = []

    dataset = SatelliteAEDataset.__new__(SatelliteAEDataset)
    dataset.obs_channel = 1
    dataset.use_satzen = False
    dataset._read_one_by_time = lambda timestamp: (
        calls.append(timestamp) or frames.get(timestamp)
    )
    return dataset, center, calls


def test_dataset_reads_only_matching_hour_file():
    dataset, center, calls = _stub_dataset()

    frame = dataset._read_frame_at_time(center)

    assert calls == [center]
    assert frame.shape == (2, 1, 2)
    np.testing.assert_array_equal(frame[0], [[2.0, 2.0]])
    np.testing.assert_array_equal(frame[1], [[1.0, 1.0]])
