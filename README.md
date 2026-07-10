# Total-Perspective-Vortex

## Total Perspective Vortex: Dataset Overview

This project builds a **Brain-Computer Interface (BCI)** by analyzing brain activity recorded via **Electroencephalography (EEG)**. If you are new to this field, think of it as "plugging your brain into a shell" to infer what a person is thinking or doing based on the electrical signals from their scalp.

The primary data source is the **PhysioNet EEG Motor Movement/Imagery Dataset**, a global benchmark containing over 1,500 recordings from **109 volunteers**.

---

## 1. Data Nature: What are we looking at?
The dataset captures brainwaves while subjects either **performed** (Motor Execution) or **imagined** (Motor Imagery) specific movements. 

*   **Motor Execution:** The subject actually moved their hands or feet.
*   **Motor Imagery:** The subject only *thought* about moving their hands or feet without physical action.
*   **Significance:** Because imagined movement activates similar brain dynamics to real movement, machine learning can be trained to "read" these intentions from the signals.

---

## 2. Data Architecture: How is it organized?

The data is structured hierarchically by **Subjects**, **Runs**, and **Events**.

### A. Subjects and Files
There are **109 subjects**. Each subject has a set of files named following the convention `S[subject_id]R[run_id].edf` (e.g., `S001R04.edf` is Subject 1, Run 4).

### B. The 14 Experimental Runs
Each subject completed 14 separate "runs" or recording sessions, each lasting 1–2 minutes. You must map the **Run Number** to the correct task:

| Run # | Category | Task Description |
| :--- | :--- | :--- |
| **1 & 2** | Baseline | Resting with eyes open/closed. |
| **3, 7, 11** | Execution | Opening/closing Left vs. Right fist. |
| **4, 8, 12** | **Imagery** | **Imagining** opening/closing Left vs. Right fist. |
| **5, 9, 13** | Execution | Opening/closing Both Fists vs. Both Feet. |
| **6, 10, 14** | **Imagery** | **Imagining** opening/closing Both Fists vs. Both Feet. |

### C. Event Codes (Labels)
Based on the dataset architecture and experimental protocol detailed in the sources, here is the summary table for the **Annotations (Events)** used in the PhysioNet EEG Motor Movement/Imagery Dataset.

### **PhysioNet EEG Event Codes Mapping**

The interpretation of the event codes (**T0, T1, T2**) is strictly dependent on the **Run Number** of the experiment.

| Event Code | General Category | Semantic Value (Unilateral Runs) | Semantic Value (Bilateral Runs) |
| :--- | :--- | :--- | :--- |
| **-** | **Target Run Numbers** | **3, 4, 7, 8, 11, 12** | **5, 6, 9, 10, 13, 14** |
| **T0** | **Rest** | Rest / Relaxation | Rest / Relaxation |
| **T1** | **Task 1** | Onset of **Left Fist** motion or imagery | Onset of **Both Fists** motion or imagery |
| **T2** | **Task 2** | Onset of **Right Fist** motion or imagery | Onset of **Both Feet** motion or imagery |

---

## 3. Technical Specifications
*   **Format:** Files are in **EDF+** (European Data Format).
*   **Sampling Rate:** The standard speed is **160 Hz** (160 snapshots of brain activity per second).
*   **Channels:** Signals come from **64 electrodes** (channels) placed on the scalp according to the **International 10-10 system**. 
*   **Montage Discrepancy:** Note that digital records index channels from **0 to 63**, while diagrams often number them **1 to 64**.

---

## 4. Critical Warnings for Developers

### ⚠️ The "Subject 88" Bug
A well-known anomaly exists for **Subject 88**: their data was recorded at **128 Hz** instead of the standard 160 Hz. If you load this subject with others, your processing pipeline will crash or produce errors. **Always exclude Subject 88** during the initial loading phase.

### Data Integrity
Due to inconsistent annotations or anomalies, many professional studies also exclude subjects **38, 89, 92, 100, 104, and 106** to maintain a clean cohort of 103 subjects.

### Repository Management
**Do not include the dataset in your Git repository.** The dataset is large (uncompressed ~3.4 GB). Your submission should only contain the Python scripts to download, parse, and analyze it.