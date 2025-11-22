import pandas as pd

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

    return df

def save_data(df: pd.DataFrame, path: str) -> None:
    """
    Save the processed dataset to a CSV file.

    Parameters:
        df (pd.DataFrame): The processed dataset.
        path (str): The file path to save the CSV file.
    """
    try:
        df.to_csv(path, index=False)
        print(f"Data saved successfully to {path}")
    except Exception as e:
        print(f"Error saving data: {e}")