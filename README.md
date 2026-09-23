
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