import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc
from tqdm.auto import tqdm
from typing import List, Tuple

# Suppress some MNE warnings for cleaner notebook output
mne.set_log_level('WARNING')

# Define exclusions based on known dataset issues
# Subject 88: Sampled at 128 Hz instead of 160 Hz
# Subjects 38, 89, 92, 100, 104, 106: Known annotation/event errors
# EXCLUDED_SUBJECTS = [38, 88, 89, 92, 100, 104, 106]
EXCLUDED_SUBJECTS = [ 88, 89, 92, 100, 104, 106]

# Parameters for our test subject
SUBJECT_TO_TEST = 1
RUN_TO_TEST = 3  # Motor Imagery: Left vs Right Fist

# 106 ok,

# Base directory for the unzipped dataset
# BASE_DATA_PATH = "mne_data"
# BASE_DATA_PATH = "mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0"

# ──────────────────────────────────────────────────────────────────────────────


def load_eeg_data(subject_id, run_id, base_path, montage_type="standard_1020"):
    """
    Loads an EDF+ file for a specific subject and run, verifying
        metadata constraints.
    args:
        subject_id (int): The subject number (e.g., 1 for S001)
        run_id (int): The run number (e.g., 4 for R04)
        base_path (str): The base directory where the dataset is stored.
        montage_type (str): The type of montage to apply (default is "standard_1020").
    returns:
        raw (mne.io.Raw): The loaded EEG() (electroencephalography data) with verified metadata
            in form of an MNE Raw object.
    """

    if subject_id in EXCLUDED_SUBJECTS:
        raise ValueError(
            f"Subject {subject_id} is excluded from this analysis due to dataset anomalies."
        )

    # Format strings to match PhysioNet standard (e.g., S001/S001R04.edf)
    subj_str = f"S{subject_id:03d}"
    run_str = f"R{run_id:02d}"
    file_path = os.path.join(base_path, subj_str, f"{subj_str}{run_str}.edf")

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    # Load the EDF file using MNE this should respect path standards (e.g., S001/S001R04.edf)
    print(f"Loading data from {file_path}...")
    raw = mne.io.read_raw_edf(file_path, preload=True)

    # 1. Verify Sampling Rate
    if raw.info["sfreq"] != 160.0:
        raise ValueError(
            f"Sampling rate mismatch! Expected 160 Hz, got {raw.info['sfreq']} Hz"
        )

    # 2. Verify Channel Count
    if len(raw.ch_names) != 64:
        raise ValueError(
            f"Channel count mismatch! Expected 64, got {len(raw.ch_names)}"
        )

    # 3. Standardize Channel Names & Set Montage
    # PhysioNet EDF files often have trailing dots in channel names (e.g., 'Fc5.')
    # Set the standard 10-20 montage for EEG channel locations
    mne.datasets.eegbci.standardize(raw)
    # apply spacial montage to the raw data
    # Load the standard 3D coordinate dictionary built into MNE
    montage = mne.channels.make_standard_montage(montage_type)
    raw.set_montage(montage, match_case=False)

    print(
        f"Metadata verified: 160 Hz sampling rate, 64 channels, {montage_type} montage applied."
    )
    return raw

    # ──────────────────────────────────────────────────────────────────────


def extract_and_map_events(raw, run_id):
    """
    Extracts events from annotations and maps them to specific tasks based on the run number.
    args:
        raw (mne.io.Raw): The loaded EEG data with annotations.
        run_id (int): The run number to determine task mapping.
    returns:
        events (np.ndarray): Array of events extracted from annotations.
        event_dict (dict): Mapping of event codes to their descriptions.
        descriptions (dict): Contextual descriptions for the events.
    """
    # Base mapping: T0 is always rest
    # ! T1: (imagery/execution) left fist / both fists
    # ! T2: (imagery/execution) right fist / both feet
    event_id = {"T0": 1, "T1": 2, "T2": 3}

    #? Define task descriptions based on the run number
    # 1: Rest (T0) by default
    descriptions = {1: 'Rest (T0)'}

    # assign task descriptions based on run_id for T1 and T2
    if run_id in [3,7, 11]:
        descriptions[2] = 'Opening/closing left fist (T1)'
        descriptions[3] = 'Opening/closing right fist (T2)'
    elif run_id in [4, 8, 12]:
        descriptions[2] = 'Imagine Left Fist (T1)'
        descriptions[3] = 'Imagine Right Fist (T2)'
    elif run_id in [5, 9, 13]:
        descriptions[2] = 'Opening/closing both fists (T1)'
        descriptions[3] = 'Opening/closing both feet (T2)'
    elif run_id in [6, 10, 14]:
        descriptions[2] = 'Imagine Both Fists (T1)'
        descriptions[3] = 'Imagine Both Feet (T2)'
    else:
        descriptions[2] = 'Task 1 (T1)'
        descriptions[3] = 'Task 2 (T2)'

    # Extract events from the EDF+ annotations
    events, event_dict = mne.events_from_annotations(raw, event_id=event_id)

    print(f"Mapped Events for Run {run_id}:")
    for event_code, desc in descriptions.items():
        print(f"  Code {event_code}: {desc}")

    return events, event_dict, descriptions


def load_and_parse_eeg(
    subject_ids: List[int],
    run_ids: List[int],
    base_path: str,
    tmin: float = 0.0,
    tmax: float = 4.0,
    include_rest: bool = False
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extracts, filters, remaps, and epochs PhysioNet EEG data into trial tensors.

    Args:
        subject_ids: List of subject integers to process.
        run_ids: List of run integers to process.
        base_path: Absolute path to the MNE data directory.
        tmin: Start time of the epoch in seconds.
        tmax: End time of the epoch in seconds.
        include_rest: Whether to include T0 (Rest) as a separate class (label 4).

    Returns:
        X: NumPy array of shape (n_trials, 64, n_samples).
        y: Target label vector of shape (n_trials,).
        metadata: DataFrame containing trial provenance (subject, run, event, class_name).
    """
    # Lists to accumulate trials, labels, and metadata across all subjects/runs
    X_list, y_list, metadata_records = [], [], []

    # Iterate through each subject requested
    for subject in tqdm(subject_ids, desc="Processing Subjects"):
        # Iterate through each run requested for this subject
        for run in run_ids:
            try:
                # 1. Load data with hardware validation (160Hz, 64 channels, standard_1020)
                raw = load_eeg_data(subject, run, base_path=base_path)

                # 2. Preprocessing: Common Average Reference (CAR)
                # Re-references the signal to the global average to reduce noise
                raw.set_eeg_reference("average", projection=False, verbose=False)

                # 3. Preprocessing: Notch Filter (60 Hz)
                # Removes power-line interference (hum) at 60 Hz
                raw.notch_filter(60.0, fir_design="firwin", verbose=False)

                # 4. Preprocessing: Band-pass Filter (8.0 - 30.0 Hz)
                # Keeps Alpha/Beta motor rhythms and removes low-frequency drift/high-frequency noise
                raw.filter(8.0, 30.0, fir_design="firwin", skip_by_annotation="edge", verbose=False)

                # 5. Dynamic Event Remapping
                # Define mappings based on the PhysioNet task protocol
                if run in [4, 8, 12]:
                    # Unilateral Imagery: Left (0) vs Right (1)
                    mapping = {"T1": 0, "T2": 1}
                elif run in [6, 10, 14]:
                    # Bilateral Imagery: Both Fists (2) vs Both Feet (3)
                    mapping = {"T1": 2, "T2": 3}
                else:
                    # Default generic mapping for other runs
                    mapping = {"T1": 0, "T2": 1}

                # Add rest period (Class 4) if requested
                if include_rest:
                    mapping["T0"] = 4

                # Extract events from annotations based on our specific mapping
                events, event_id = mne.events_from_annotations(raw, event_id=mapping, verbose=False)

                # 6. Trial Epoching
                # Segment the signal; baseline=None as drift is already filtered out
                epochs = mne.Epochs(
                    raw, events, event_id=event_id,
                    tmin=tmin, tmax=tmax,
                    baseline=None, preload=True, verbose=False
                )

                # 7. Convert to NumPy for scikit-learn integration
                X_run = epochs.get_data(copy=True)
                y_run = epochs.events[:, -1]

                # Store data if any trials were found
                if len(y_run) > 0:
                    X_list.append(X_run)
                    y_list.append(y_run)

                    # 8. Record Metadata for every trial to track subject/run/class
                    # Reverse map the label to get the human-readable class name
                    inv_mapping = {v: k for k, v in mapping.items()}
                    for i, label in enumerate(y_run):
                        metadata_records.append({
                            "subject_id": subject,
                            "run_id": run,
                            "trial_id": i,
                            "label": label,
                            "class_name": inv_mapping[label]
                        })

                # Force memory cleanup
                del raw, epochs
                gc.collect()

            except Exception as e:
                # Log error and skip to the next run
                print(f"Error processing S{subject:03d}R{run:02d}: {e}")
                continue

    # Final Check: Did we find any data?
    if not X_list:
        raise ValueError("No valid EEG trials were found for the provided subject/run criteria.")

    # Concatenate all trial arrays into a single unified 3D tensor
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    metadata = pd.DataFrame(metadata_records)

    return X, y, metadata
