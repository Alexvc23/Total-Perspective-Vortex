## Total Perspective Vortex: Copilot Instructions

### Project Context & Purpose

* **Core Objective**: Build a brain-computer interface (BCI) using machine learning algorithms to analyze electroencephalographic (EEG) data.


* **Primary Task**: Infer subject thoughts or actions by distinguishing between two specific types of motion within a given timeframe.


* **Workflow Strategy**: Follow an incremental development approach by first testing architecture with standard algorithms before introducing custom implementations.



---

### Technology Stack & Architecture

* **Mandatory Tools**: Code must be written in Python, utilizing MNE for EEG data processing and scikit-learn for machine learning.


* **Pipeline Structure**: The architecture must rely on the `scikit-learn` pipeline object.


* **Custom Classes**: Any custom steps within the pipeline must be implemented using `BaseEstimator` and `TransformerMixin`.


* **Dimensionality Reduction**: Do not rely entirely on pre-built libraries for this step; you must manually implement a dimensionality reduction algorithm (such as CSP or PCA) and develop the projection matrix.


* **Mathematical Operations**: Use permitted Numpy and Scipy functions specifically for eigenvalue calculation and covariance matrix estimation.



---

### Hard Constraints & Red Lines

* **Library Restrictions**: You are explicitly forbidden from using the `mne-realtime` library for data stream simulation.


* **Repository Rules**: The dataset (e.g., PhysioNet data) must never be included or committed to the Git repository.


* **Overfitting Prevention**: You must strictly separate data into Training, Validation, and Test sets to avoid overfitting at all costs.



---

### Performance Targets & Validation

| Metric | Target / Standard |
| --- | --- |
| **Global Accuracy** | Achieve at least a 60% mean accuracy across all subjects in the test data.

 |
| **Real-Time Latency** | The stream simulation prediction script must output results with a delay of less than 2 seconds per data chunk.

 |
| **Validation Method** | Implement cross-validation (`cross_val_score`) across the entire processing pipeline to ensure reliability.

 |
| **Feature Selection** | Conduct thorough feature selection (e.g., signal power by frequency and channel) to maximize pipeline performance.

 |

---

### Future / Bonus Implementations (If Prompted)

* Integrate wavelet transforms to improve the analysis of the signal specter.


* Develop custom functions for singular value decomposition or noise-resistant covariance estimation.