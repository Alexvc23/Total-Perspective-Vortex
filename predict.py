import argparse
import time
import numpy as np
from srcs.pipeline import load_model
from srcs.stream import edf_playback_generator, stream_prediction_logger

def run_prediction_session(args):
    """
    Execute a simulated real-time prediction session using playback mode.
    """
    print(f"=== Total Perspective Vortex: Real-Time Playback & Prediction ===")
    
    # 1. Load Model
    print(f"\n[Step 1] Loading serialized model from: {args.model}")
    pipeline, metadata, timestamp = load_model(args.model)
    print(f"  Model Timestamp: {timestamp}")
    print(f"  Trained on Runs: {metadata.get('runs')}")
    print(f"  Training Accuracy: {metadata.get('test_acc', 0.0):.4f}")
    
    # 2. Initialize Playback Stream
    print(f"\n[Step 2] Initializing playback stream for Subject {args.subject}, Run {args.run}...")
    stream = edf_playback_generator(
        subject_id=args.subject,
        run_id=args.run,
        data_path=args.data_path,
        chunk_duration=args.chunk_duration
    )
    
    # Class names for logging
    class_names = ["Left Fist", "Right Fist", "Both Fists", "Both Feet"]
    
    print(f"\n[Step 3] Starting Inference Loop (Max Latency: {args.max_latency_ms}ms)...")
    print("-" * 85)
    
    # 3. Prediction Loop
    latencies = []
    try:
        for i, (chunk, truth, timestamp) in enumerate(stream):
            # Start High-Precision Timer
            t_start = time.perf_counter()
            
            # Prepare chunk for scikit-learn (N, C, T) -> (1, 64, 641)
            X_chunk = chunk[np.newaxis, ...]
            
            # Execute Pipeline
            prediction = pipeline.predict(X_chunk)[0]
            
            # Stop Timer
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000.0
            latencies.append(latency_ms)
            
            # Rigorous Latency Assertion
            if latency_ms > args.max_latency_ms:
                raise AssertionError(f"Latency threshold exceeded! {latency_ms:.2f}ms > {args.max_latency_ms}ms")
            
            # Log Output
            stream_prediction_logger(
                epoch_idx=i+1,
                prediction=prediction,
                truth=truth,
                latency_ms=latency_ms,
                class_names=class_names
            )
            
    except Exception as e:
        print(f"\n[Error] Stream interrupted: {e}")
        return

    # 4. Final Audit
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    print("-" * 85)
    print(f"\n[Step 4] Latency Audit Complete")
    print(f"  Mean Latency: {avg_latency:.2f} ms")
    print(f"  Peak Latency: {max_latency:.2f} ms")
    print(f"  Status: {'PASSED' if max_latency < args.max_latency_ms else 'FAILED'}")
    print("\n=== Playback Session Finished ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCI Playback Prediction Script")
    
    # Data arguments
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (e.g. 1)")
    parser.add_argument("--run", type=int, default=4, help="Run ID (e.g. 4 for Left/Right Imagery)")
    parser.add_argument("--data-path", type=str, default="doc/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0",
                        help="Path to dataset")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="models/bci_pipeline.joblib",
                        help="Path to serialized model")
    
    # Performance arguments
    parser.add_argument("--chunk-duration", type=float, default=4.0, help="Window duration in seconds")
    parser.add_argument("--max-latency-ms", type=float, default=2000.0, help="Max allowed latency in ms")
    
    args = parser.parse_args()
    run_prediction_session(args)
