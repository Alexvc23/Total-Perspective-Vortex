
## Dataset Architecture: The PhysioNet EEG Structure

This project processes the **PhysioNet EEG Motor Movement/Imagery Dataset**, a benchmark collection of brain-computer interface (BCI) data recorded using the BCI2000 system.

The data is structured hierarchically across four dimensions:

### 1. Cohort & Sessions (Subjects and Runs)

* **Subjects:** 109 volunteers.
* **Runs:** Each subject performed 14 experimental runs (1-2 minutes each).
* **Protocol Map:**
* **Runs 1 & 2:** Baseline (Eyes open / Eyes closed).
* **Runs 3, 7, 11:** Motor Execution (Left vs. Right fist).
* **Runs 4, 8, 12:** Motor Imagery (Imagine Left vs. Right fist).
* **Runs 5, 9, 13:** Motor Execution (Both fists vs. Both feet).
* **Runs 6, 10, 14:** Motor Imagery (Imagine Both fists vs. Both feet).



### 2. Experimental Trials (Events)

Within each run, the data contains embedded annotations marking the exact onset of a stimulus or action.

* **T0:** Rest period.
* **T1:** Left fist action (in unilateral runs) OR Both fists action (in bilateral runs).
* **T2:** Right fist action (in unilateral runs) OR Both feet action (in bilateral runs).

### 3. Spatial & Temporal Granularity (Channels and Sampling)

* **Channels (Space):** Brain activity is captured across 64 electrodes positioned according to the International 10-10 system. *(Note: MNE and digital arrays index these from 0 to 63, while clinical diagrams label them 1 to 64)*.
* **Sampling Rate (Time):** The data is sampled at **160 Hz**, meaning the system captures 160 discrete voltage measurements (snapshots) per second across all 64 channels simultaneously.
* **Format:** Data is stored in EDF+ (European Data Format).
---

## Preprocessing & Data Engineering Pipeline

To transform raw brainwaves into a high-quality dataset suitable for Machine Learning, we implemented a multi-stage pipeline designed for **signal integrity** and **tensor uniformity**.

### Phase 1: Cohort Refinement (The "Clean Cohort" Analysis)
EEG datasets are often prone to acquisition errors. Before any processing, we performed a lightweight audit of all 109 subjects:

1.  **Hardware Validation**: We enforced a strict requirement of **160 Hz** sampling and **64 channels**. 
    *   *Result*: Subject 88 was excluded due to a non-standard 128 Hz sampling rate, which would cause spectral distortion in a unified pipeline.
2.  **Event Integrity Audit**: We verified trial counts (30 events per task run).
    *   *Result*: Several subjects (38, 89, 92, 100, 104, 106) were excluded due to documented annotation drift or physiologically unreliable markers.
3.  **The 102-Subject Benchmark**: By systematically removing hardware and annotation anomalies, we established a "Clean Cohort" of 102 subjects, ensuring our models learn from reliable, synchronized data.

### Phase 2: Signal Processing & 3D Tensor Structuring
Once the cohort was finalized, we refactored the parsing logic to transform continuous raw signals into segmented 3D NumPy arrays ($X \in \mathbb{R}^{N_{trials} \times 64 \times 641}$).

#### 1. Spatial & Temporal Filtering (The "Why")
*   **Common Average Reference (CAR)**: We re-referenced all electrodes to the global average. This acts as a spatial high-pass filter, removing noise common to all sensors (like muscle artifacts or hardware hum) to highlight local brain activity.
*   **60 Hz Notch Filter**: Specifically targets and removes power-line interference without affecting the surrounding brain frequencies.
*   **8.0–30.0 Hz Band-pass Filter**: We isolate the **Alpha (8-12Hz)** and **Beta (13-30Hz)** bands. These are the primary "Mu" rhythms used in Motor Imagery; when you imagine moving, these specific frequencies change in the motor cortex.

#### 2. Dynamic Task Remapping
Since the dataset uses generic markers ($T1, T2$), we dynamically remap them into explicit categorical labels based on the run type:
*   **Unilateral Runs (4, 8, 12)**: $T1 \to 0$ (Left Fist), $T2 \to 1$ (Right Fist).
*   **Bilateral Runs (6, 10, 14)**: $T1 \to 2$ (Both Fists), $T2 \to 3$ (Both Feet).
*   **Rest ($T0$)**: Excluded by default to focus on active intent classification.

#### 3. Trial Epoching
We extract **4-second windows** ($t=0.0$ to $t=4.0$ s) relative to each stimulus trigger. This results in uniform data tensors that can be fed directly into Scikit-Learn pipelines.

---

## Technical Implementation
The core logic is encapsulated in `srcs/misc.py` via the `load_and_parse_eeg()` function. This function automates:
1.  **Loading**: EDF+ parsing with `mne`.
2.  **Cleaning**: Automated filtering and re-referencing.
3.  **Structuring**: Transformation into $X$ (3D tensor), $y$ (target vector), and a `metadata` DataFrame for provenance tracking (Subject/Run/Trial mapping).
