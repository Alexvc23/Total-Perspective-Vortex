import argparse
import numpy as np
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from srcs.data import load_cohort, get_subject_split
from srcs.pipeline import create_bci_pipeline, save_model

def run_training_workflow(args):
    """
    Execute the full training, cross-validation, and holdout evaluation workflow.

    Summary:
        Orchestrates subject-level train/test partitioning to prevent data leakage,
        loads training and test cohorts, runs Stratified K-Fold cross-validation,
        fits the full pipeline, evaluates performance against the >= 60% accuracy benchmark,
        and serializes the fitted model artifact with metadata.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing data path,
            runs, test size, hyperparameters, and output path.

    Returns:
        None

    Remarks:
        Alexander VALENCIA - 01/10/2026
    """
    print("=== Total Perspective Vortex: Training & Rigorous Validation ===")
    
    # 1. Subject-Level Splitting (Leakage Prevention)
    print(f"\n[Step 1] Partitioning subjects from: {args.data_path}")
    train_subs, test_subs = get_subject_split(
        args.data_path, 
        test_size=args.test_size, 
        random_state=args.seed
    )
    
    print(f"  Training Subjects ({len(train_subs)}): {train_subs}")
    print(f"  Holdout Test Subjects ({len(test_subs)}): {test_subs}")
    
    # Proof of Zero Leakage
    assert len(set(train_subs).intersection(set(test_subs))) == 0, "CRITICAL: Subject leakage detected!"
    
    # 2. Load Training Cohort
    print(f"\n[Step 2] Loading training data (Runs: {args.runs})...")
    X_train, y_train, _ = load_cohort(train_subs, args.runs, args.data_path)
    print(f"  Training Tensor: {X_train.shape} | Classes: {np.unique(y_train)}")
    
    # 3. Pipeline Creation & Cross-Validation
    print(f"\n[Step 3] Running {args.n_splits}-fold Stratified Cross-Validation...")
    pipeline = create_bci_pipeline(n_components=args.n_components, reg=args.reg)
    
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, n_jobs=-1)
    
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    print(f"  CV Accuracies: {cv_scores}")
    print(f"  Mean CV Accuracy: {mean_cv:.4f} (+/- {std_cv:.4f})")
    
    # 4. Final Training & Holdout Evaluation
    print(f"\n[Step 4] Fitting final model and evaluating on unseen test subjects...")
    pipeline.fit(X_train, y_train)
    
    X_test, y_test, _ = load_cohort(test_subs, args.runs, args.data_path)
    test_acc = pipeline.score(X_test, y_test)
    
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    # 5. Benchmark Assertion & Export
    print(f"\n[Step 5] Performance Verification & Model Export...")
    # Assertion check (using test accuracy as the rigorous benchmark)
    benchmark_met = test_acc >= 0.60
    status = "PASSED" if benchmark_met else "FAILED"
    print(f"  Benchmark (>= 60%): {status} ({test_acc:.4f})")
    
    if not benchmark_met:
        print("  Warning: Model did not meet the 60% accuracy threshold.")
    
    # Save the model
    metadata = {
        "train_subjects": train_subs,
        "test_subjects": test_subs,
        "runs": args.runs,
        "cv_mean": mean_cv,
        "test_acc": test_acc,
        "n_components": args.n_components,
        "reg": args.reg
    }
    
    saved_path = save_model(pipeline, args.output, metadata)
    file_size = os.path.getsize(saved_path) / 1024
    print(f"  Model saved to: {saved_path} ({file_size:.2f} KB)")
    
    print("\n=== Training Process Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCI Pipeline Training Script")
    
    #region Data arguments
    parser.add_argument("--data-path", type=str, default="doc/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0",
                        help="Path to PhysioNet dataset")
    parser.add_argument("--runs", type=int, nargs="+", default=[4, 8, 12],
                        help="List of runs to include (default: 4, 8, 12)")
    #! --test-size and --seed are used for subject-level splitting to prevent data leakage
        #? 0.2 means 20% of subjects will be held out for testing and 80% will be used for training.
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion of subjects for testing (default: 0.2)")
    #! --seed is used to ensure reproducibility of the subject-level split.
        #? --seed=42 makes sure the same subjects are selected for training and testing each time the script is run.
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    #region Pipeline arguments
    parser.add_argument("--n-components", type=int, default=6,
                        help="Number of CSP components")
    #! --reg is the shrinkage regularization parameter for covariance estimation in CSP.
        #? this is the psilon value (very close to zero) that prevents singular covariance matrices and improves numerical stability.
    parser.add_argument("--reg", type=float, default=1e-6,
                        help="CSP covariance regularization")
    #! --n-splits is the number of cross-validation folds to use.
        #? The default is 5, which means 5-fold cross-validation will be used.
            #* a fold validation means the training data is split into 5 parts, and each part is used once as a validation set while the other 4 parts are used for training.
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of CV folds")
    
    #region Output arguments
    #! we store the trained model in a joblib file, which is a serialized format for Python objects.
        #? this format allows us to save the trained pipeline and load it later for inference without retraining.
    parser.add_argument("--output", type=str, default="models/bci_pipeline.joblib",
                        help="Path to save the serialized model")

    args = parser.parse_args()
    run_training_workflow(args)
