
import os
import mne
import matplotlib.pyplot as plt
import numpy as np

# Suppress some MNE warnings for cleaner notebook output
mne.set_log_level('WARNING')

# Define exclusions based on known dataset issues
# Subject 88: Sampled at 128 Hz instead of 160 Hz
# Subjects 38, 89, 92, 100, 104, 106: Known annotation/event errors
EXCLUDED_SUBJECTS = [38, 88, 89, 92, 100, 104, 106]

# Parameters for our test subject
SUBJECT_TO_TEST = 1
RUN_TO_TEST = 3  # Motor Imagery: Left vs Right Fist

# 106 ok, 

# Base directory for the unzipped dataset
BASE_DATA_PATH = "./data/files"

# ──────────────────────────────────────────────────────────────────────────────

def load_eeg_data(
    subject_id, run_id, base_path=BASE_DATA_PATH, montage_type="standard_1020"
):
    """
    Loads an EDF+ file for a specific subject and run, verifying
        metadata constraints.
    args:
        subject_id (int): The subject number (e.g., 1 for S001)
        run_id (int): The run number (e.g., 4 for R04)
        base_path (str): The base directory where the dataset is stored.
        montage_type (str): The type of montage to apply (default is "standard_1020").
    returns:
        raw (mne.io.Raw): The loaded EEG data with verified metadata
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
