"""
Data loading utilities for VQA4Mix project.
"""

import json
import pandas as pd
import os

def load_json_data(file_path):
    """
    Load data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict or list: The loaded JSON data.
    """
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def load_annotation_data(file_path):
    """
    Load annotation data from a JSON file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        pd.DataFrame: DataFrame containing the annotation data.
    """
    return pd.read_json(file_path)

def save_json_data(data, output_file):
    """
    Save data to a JSON file.
    
    Args:
        data: The data to save.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {output_file}")

def convert_df_to_json(df, output_file):
    """
    Convert a DataFrame to a list of dictionaries and save as JSON.
    
    Args:
        df (pd.DataFrame): The DataFrame to convert.
        output_file (str): Path to the output JSON file.
    """
    list_of_dicts = df.to_dict(orient="records")
    save_json_data(list_of_dicts, output_file)
