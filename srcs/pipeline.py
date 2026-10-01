import joblib
import os
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from srcs.csp import CustomCSP

def create_bci_pipeline(n_components: int = 6, reg: float = 1e-6):
    """
    Create a standardized Scikit-Learn BCI pipeline.

    Summary:
        Constructs a pipeline consisting of a CustomCSP spatial filter transformer
        followed by a Linear Discriminant Analysis (LDA) classifier.

    Args:
        n_components (int): Number of spatial components to retain in CSP.
        reg (float): Regularization parameter for CSP covariance estimation.

    Returns:
        sklearn.pipeline.Pipeline: The un-fitted BCI pipeline.

    Remarks:
        Alexander VALENCIA - 01/10/2026
    """
    pipeline = Pipeline([
        ('csp', CustomCSP(n_components=n_components, reg=reg)),
        ('lda', LinearDiscriminantAnalysis())
    ])
    return pipeline

def save_model(pipeline: Pipeline, path: str, metadata: dict = None):
    """
    Serialize the fitted pipeline and metadata to disk.

    Summary:
        Saves the pipeline using joblib. Creates the destination directory if it
        doesn't exist and attaches operational metadata (timestamp, version, etc.).

    Args:
        pipeline (Pipeline): The fitted Scikit-Learn pipeline to save.
        path (str): The file path where the model should be saved (e.g., 'models/bci.joblib').
        metadata (dict): Optional dictionary of training metrics or hyperparameters.

    Returns:
        str: The absolute path to the saved model file.

    Remarks:
        Alexander VALENCIA - 01/10/2026
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    payload = {
        "pipeline": pipeline,
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat(),
    }
    
    joblib.dump(payload, path)
    return os.path.abspath(path)

def load_model(path: str):
    """
    Deserialize a BCI pipeline and its metadata from disk.

    Summary:
        Loads a joblib artifact and extracts the pipeline and associated training
        metadata.

    Args:
        path (str): Path to the .joblib model file.

    Returns:
        tuple: (pipeline, metadata) where pipeline is the Scikit-Learn object.

    Remarks:
        Alexander VALENCIA - 01/10/2026
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
        
    payload = joblib.load(path)
    return payload["pipeline"], payload.get("metadata", {}), payload.get("timestamp")
