import mne
import numpy as np
import time
from typing import Generator, Tuple
from srcs.data import load_raw_eeg, preprocess_raw

def edf_playback_generator(
    subject_id: int,
    run_id: int,
    data_path: str,
    chunk_duration: float = 4.0,
    sfreq: float = 160.0
) -> Generator[Tuple[np.ndarray, int, float], None, None]:
    """
    Custom generator to simulate a real-time EEG stream from an EDF file.

    Summary:
        Loads a raw EDF file, applies full-signal preprocessing (CAR, Notch, Band-pass),
        and yields data chunks corresponding to trial epochs defined in the annotations.
        This simulates a "playback" mode where each trial is treated as a live chunk.

    Args:
        subject_id (int): Subject ID to stream.
        run_id (int): Run ID to stream.
        data_path (str): Path to the dataset.
        chunk_duration (float): Duration of each data chunk in seconds.
        sfreq (float): Sampling frequency (default 160 Hz).

    Yields:
        tuple: (chunk_data, ground_truth_label, timestamp)
            - chunk_data: NumPy array of shape (64, n_samples).
            - ground_truth_label: The class label (0, 1, 2, or 3).
            - timestamp: The onset time of the chunk in seconds.

    Remarks:
        Alexander VALENCIA - 01/10/2026
    """
    # 1. Load and Preprocess (simulate pre-processing the stream or buffered data)
    raw = load_raw_eeg(subject_id, run_id, data_path)
    raw = preprocess_raw(raw)

    # 2. Extract events to align chunks with ground truth
    # Task mapping consistent with srcs/data.py
    if run_id in [4, 8, 12]:
        mapping = {"T1": 0, "T2": 1} # Left vs Right Fist
    elif run_id in [6, 10, 14]:
        mapping = {"T1": 2, "T2": 3} # Both Fists vs Both Feet
    else:
        mapping = {"T1": 0, "T2": 1}

    events, _ = mne.events_from_annotations(raw, event_id=mapping, verbose=False)
    
    # Calculate the number of samples per chunk based on the specified duration and sampling frequency
        # which is 4 seconds * 160 Hz = 640 samples, plus 1 for no event overlap, resulting in 641 samples per chunk.
    samples_per_chunk = int(chunk_duration * sfreq) + 1

    # 3. Iterate through events (trials) and yield data
    for i, event in enumerate(events):
        onset_sample = event[0]
        label = event[2]
        
        # Calculate the stop sample for the chunk
        stop_sample = onset_sample + samples_per_chunk
        
        # Boundary check
        # If the stop sample exceeds the total number of samples in the raw data, skip this trial.
        if stop_sample > raw.n_times:
            continue
            
        #! Extract chunk: (channels, samples)
        # Note: Scikit-learn pipeline expects (trials, channels, samples)
        # but the generator yields one chunk (channels, samples) at a time.
        chunk_data = raw.get_data(start=onset_sample, stop=stop_sample)
        
        # Calculate the timestamp in seconds for the chunk onset
        timestamp = onset_sample / sfreq
        
        
        # yield differs from return in that it allows the function to produce a series of values over time,
            # rather than computing them all at once.
            # e.g: here the data is streamed in chunks, simulating real-time EEG acquisition 
        # Yield the chunk, label, and timestamp( in seconds)
            # *  The yielded chunk is of shape (64, 641) for 64 channels and 641 samples 
            # * even if the pipeline expects (1, 64, 641) for a single trial. The pipeline will handle the reshaping.
        yield chunk_data, label, timestamp

def stream_prediction_logger(epoch_idx: int, prediction: int, truth: int, latency_ms: float, class_names: list = None):
    """
    Formatted terminal output for real-time prediction logs.
    """
    match = (prediction == truth)
    status = "SUCCESS" if match else "FAILURE"
    color = "\033[92m" if match else "\033[91m"
    reset = "\033[0m"
    
    pred_name = class_names[prediction] if class_names else str(prediction)
    truth_name = class_names[truth] if class_names else str(truth)
    
    print(f"epoch {epoch_idx:02d}: [prediction: {pred_name:^12}] "
          f"[truth: {truth_name:^12}] "
          f"equal? {color}{str(match):<5}{reset} | "
          f"Latency: {latency_ms:6.2f} ms")
