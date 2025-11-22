import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib

def split_data(df: pd.DataFrame, target: str = "pass_fail", test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset into train and test sets.
    
    Parameters:
        df (pd.DataFrame): The dataset to split
        target (str): The target variable name
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    
    """
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def resample_data(X_train, y_train):
    """
    Apply SMOTE to balance the training data.
    Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
    Returns:
        X_res, y_res: Resampled training features and labels    
    """
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

def scale_data(X_train, X_test):
    """
    Scale features using StandardScaler.
    Parameters:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
    Returns:
        X_train_scaled, X_test_scaled: Scaled training and test features    
    """
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_models(X_train, y_train):
    """
    Train multiple models and return a dict of trained models.
    
    Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
    Returns:
        dict: Dictionary with model names as keys and trained model instances as values
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights='uniform')
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate models and print metrics.
    
    Parameters:
        models (dict): Dictionary of trained models
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): True labels for test set
    Returns:
        dict: Dictionary with model names as keys and their accuracy and f1_score as values
        
    """
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results[name] = {"accuracy": acc, "f1_score": f1}
        print(f"---{name}---")
        print(classification_report(y_test, y_pred))
    return results


def save_model(model, path: str):
    """
    Save trained model to disk.
    
    Parameters:
        model: Trained model instance
        path (str): File path to save the model 
            
    """
    try:
        joblib.dump(model, path)
        print(f"Model saved at {path}")
    except Exception as e:
        print(f"Error saving model: {e}")
