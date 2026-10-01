import numpy as np
import pytest
import os
from srcs.csp import CustomCSP
from srcs.pipeline import create_bci_pipeline, save_model, load_model
from srcs.stream import edf_playback_generator

def test_custom_csp_shape():
    """Verify CustomCSP output dimensions."""
    n_trials, n_channels, n_times = 10, 64, 641
    X = np.random.randn(n_trials, n_channels, n_times)
    y = np.array([0, 1] * 5)
    
    n_comp = 6
    csp = CustomCSP(n_components=n_comp)
    csp.fit(X, y)
    X_transformed = csp.transform(X)
    
    assert X_transformed.shape == (n_trials, n_comp)

def test_pipeline_serialization(tmp_path):
    """Verify that the pipeline can be saved and reloaded with metadata."""
    pipeline = create_bci_pipeline(n_components=4)
    model_path = os.path.join(tmp_path, "test_model.joblib")
    
    metadata = {"accuracy": 0.85, "test_subjects": [1, 2, 3]}
    save_model(pipeline, model_path, metadata=metadata)
    
    reloaded_pipeline, reloaded_meta, timestamp = load_model(model_path)
    
    assert reloaded_pipeline.get_params()['csp__n_components'] == 4
    assert reloaded_meta["accuracy"] == 0.85
    assert timestamp is not None

def test_stream_generator_integrity():
    """Verify that the playback generator yields correctly shaped chunks."""
    # Note: This test requires local data to be present
    data_path = "doc/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0"
    if not os.path.exists(data_path):
        pytest.skip("Local dataset not found for integration test.")
        
    stream = edf_playback_generator(subject_id=1, run_id=4, data_path=data_path)
    chunk, label, ts = next(stream)
    
    assert chunk.shape == (64, 641)
    assert label in [0, 1]
    assert ts >= 0.0
