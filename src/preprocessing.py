
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    """
    Load the dataset from the specified CSV file path.
    
    Parameters:
        path (str): The file path to the CSV file.
    
    Returns: 
        pd.DataFrame: The loaded dataset as a pandas DataFrame. 
    """
    try:
        df = pd.read_csv(path, sep=";") # Dataset is from UCI repository which uses ';' as separator
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    

def preprocess_data(df) -> pd.DataFrame:
    """
    Pre-Process the student dataset:
    - Converts yes/no columns to 1/0
    - Standardizes categorical values
    - Handles missing values (if any)

    Parameters:
        df (pd.DataFrame): Raw dataset

    Returns:
        pd.DataFrame: Cleaned dataset
    """

    # List of yes/no columns
    yes_no_cols = [
        "schoolsup", "famsup", "paid", "activities",
        "nursery", "higher", "internet", "romantic"
    ]

    # Convert yes/no to 1/0
    for col in yes_no_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # Fill missing numeric columns with median (if any)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill missing categorical columns with mode (if any)
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if df[col].nunique() > 1]
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())


    return df

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary target 'pass_fail' from G3 and drop G1, G2, G3 to avoid data leakage.

    Parameters:
        df (pd.DataFrame): Preprocessed dataset

    Returns:
        pd.DataFrame: Dataset with 'pass_fail' target added and G1,G2,G3 removed
    """
    df["pass_fail"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
    df = df.drop(["G1", "G2", "G3"], axis=1)
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering on the student dataset:
    - One-hot encodes categorical columns
    - Scales numeric features
    - Creates new features like total_alcohol, study_fail_ratio, social_activity

    Parameters:
        df (pd.DataFrame): Preprocessed dataset

    Returns:
        pd.DataFrame: Dataset ready for modeling
    """
    
        # --- Create new features ---
    df["total_alcohol"] = df["Dalc"] + df["Walc"]
    df["study_fail_ratio"] = df["studytime"] / (df["failures"] + 1)

    # --- One-hot encode categorical columns ---
    categorical_cols =  df.select_dtypes(include=['object', 'bool']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Convert boolean columns to 0/1
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # --- Scale numeric features ---
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if "pass_fail" in numeric_cols:
        numeric_cols.remove("pass_fail")
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """
    Save the processed dataset to a CSV file.

    Parameters:
        df (pd.DataFrame): The processed dataset.
        path (str): The file path to save the CSV file.
    """
    try:
        df.to_csv(path, sep=';', index=False)
        print(f"Data saved successfully to {path}")
    except Exception as e:
        print(f"Error saving data: {e}")