import os
import mne
import numpy as np
import pandas as pd
import gc
from tqdm.auto import tqdm
from typing import List, Tuple, Dict

# Suppress MNE warnings for production output
mne.set_log_level('ERROR')

# Subjects with hardware or annotation desynchronization issues
EXCLUDED_SUBJECTS = [38, 88, 89, 92, 100, 104, 106]

def load_raw_eeg(subject_id: int, run_id: int, data_path: str):
    """
    Load a single EDF+ file and validate hardware constraints.

    Summary:
        Reads a PhysioNet EDF+ file for a specific subject and run, verifying
        that the sampling rate is exactly 160 Hz and channel count is 64, then
        applies the standard 10-20 EEG montage.

    Args:
        subject_id (int): Subject number (e.g., 1 for S001).
        run_id (int): Run number (e.g., 4 for R04).
        data_path (str): Base directory where the dataset is stored.

    Returns:
        mne.io.Raw: The loaded MNE raw object with verified metadata and montage.

    Remarks:
        Alexander VALENCIA - 01/10/2026
    """
    if subject_id in EXCLUDED_SUBJECTS:
        raise ValueError(f"Subject {subject_id} is excluded due to known anomalies.")

    subj_str = f"S{subject_id:03d}"
    run_str = f"R{run_id:02d}"
    file_path = os.path.join(data_path, subj_str, f"{subj_str}{run_str}.edf")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EDF file not found: {file_path}")

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # Hardware Assertions
    if raw.info["sfreq"] != 160.0:
        raise ValueError(f"S{subject_id:03d}R{run_id:02d}: Expected 160Hz, got {raw.info['sfreq']}Hz")
    if len(raw.ch_names) != 64:
        raise ValueError(f"S{subject_id:03d}R{run_id:02d}: Expected 64 channels, got {len(raw.ch_names)}")

    # Standardize and set montage
    mne.datasets.eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, verbose=False)
    
    return raw

def preprocess_raw(raw):
    """
    Apply standard BCI preprocessing to raw EEG signals.

    Summary:
        Applies Common Average Reference (CAR), a 60 Hz notch filter to remove
        power-line interference, and an 8.0 - 30.0 Hz FIR band-pass filter to isolate
        Alpha/Beta motor rhythms.

    Args:
        raw (mne.io.Raw): The loaded MNE raw object.

    Returns:
        mne.io.Raw: The preprocessed MNE raw object.

    Remarks:
        Alexander VALENCIA - 01/10/2026
    """
    # 1. Common Average Reference (CAR)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    
    # 2. Notch Filter (60 Hz)
    raw.notch_filter(60.0, fir_design="firwin", verbose=False)
    
    # 3. Band-pass Filter (8.0 - 30.0 Hz) for Alpha/Beta motor rhythms
    raw.filter(8.0, 30.0, fir_design="firwin", skip_by_annotation="edge", verbose=False)
    
    return raw

def load_cohort(
    subject_ids: List[int],
    run_ids: List[int],
    data_path: str,
    tmin: float = 0.0,
    tmax: float = 4.0
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load and process a cohort of subjects and runs into trial tensors.

    Summary:
        Iterates through specified subjects and runs, loads raw EDFs, applies
        preprocessing and event mapping, segments trials into epochs, and
        concatenates them into unified numpy arrays and metadata tables.

    Args:
        subject_ids (List[int]): List of subject IDs to process.
        run_ids (List[int]): List of run IDs to process.
        data_path (str): Base path to the dataset directory.
        tmin (float): Start time of the epoch relative to event onset in seconds.
        tmax (float): End time of the epoch relative to event onset in seconds.

    Returns:
        Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
            - X: Epoch tensor of shape (n_trials, n_channels, n_samples).
            - y: Label array of shape (n_trials,).
            - metadata: DataFrame tracking subject, run, label, and class name per trial.

    Remarks:
        Alexander VALENCIA - 01/10/2026
    """
    X_list, y_list, metadata_records = [], [], []

    for subject in tqdm(subject_ids, desc="Loading Cohort", leave=False):
        for run in run_ids:
            try:
                raw = load_raw_eeg(subject, run, data_path)
                raw = preprocess_raw(raw)

                # Event Mapping based on PhysioNet protocol
                if run in [4, 8, 12]:
                    mapping = {"T1": 0, "T2": 1} # Left vs Right Fist
                elif run in [6, 10, 14]:
                    mapping = {"T1": 2, "T2": 3} # Both Fists vs Both Feet
                else:
                    mapping = {"T1": 0, "T2": 1}

                events, event_id = mne.events_from_annotations(raw, event_id=mapping, verbose=False)
                
                epochs = mne.Epochs(
                    raw, events, event_id=event_id,
                    tmin=tmin, tmax=tmax,
                    baseline=None, preload=True, verbose=False
                )

                X_run = epochs.get_data(copy=True)
                y_run = epochs.events[:, -1]

                if len(y_run) > 0:
                    X_list.append(X_run)
                    y_list.append(y_run)
                    
                    inv_mapping = {v: k for k, v in mapping.items()}
                    for i, label in enumerate(y_run):
                        metadata_records.append({
                            "subject_id": subject,
                            "run_id": run,
                            "label": label,
                            "class_name": inv_mapping[label]
                        })

                del raw, epochs
                gc.collect()

            except Exception as e:
                print(f"  Warning: Skipping S{subject:03d}R{run:02d} due to error: {e}")
                continue

    if not X_list:
        raise ValueError("No trials were loaded for the specified cohort.")

    # turn list into numpy array 
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    # create metadata DataFrame
    metadata = pd.DataFrame(metadata_records)

    return X, y, metadata

# ──────────────────────────────────────────────────────────────────────────────

def get_subject_split(
    data_path: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Partition available subjects into training and test cohorts.

    Summary:
        Discovers all valid subjects in the dataset directory (excluding known anomalous
        subjects) and randomly splits them at the subject level to guarantee zero trial
        leakage between training and testing sets.

    Args:
        data_path (str): Base path to the dataset directory.
        test_size (float): Proportion of subjects to allocate to the test cohort.
        random_state (int): Random seed for reproducible subject shuffling.

    Returns:
        Tuple[List[int], List[int]]:
            - train_subjects: List of subject IDs for training/cross-validation.
            - test_subjects: List of subject IDs reserved for final holdout evaluation.

    Remarks:
        Alexander VALENCIA - 01/10/2026
    """
    # Discover available subjects (S001 - S109)
    all_dirs = [d for d in os.listdir(data_path) if d.startswith('S') and d[1:].isdigit()]
    all_subjects = sorted([int(d[1:]) for d in all_dirs if int(d[1:]) not in EXCLUDED_SUBJECTS])
    
    np.random.seed(random_state)
    np.random.shuffle(all_subjects)
    
    # calculate split index based on test_size: 80% of subject for training and 20% for testing
    split_idx = int(len(all_subjects) * (1 - test_size))
    # set the training and testing subjects based on the split index
    train_subjects = sorted(all_subjects[:split_idx])
    # set the testing subjects based on the split index
    test_subjects = sorted(all_subjects[split_idx:])
    
    return train_subjects, test_subjects
