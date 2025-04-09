"""
Evaluation utilities for VQA4Mix project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_accuracy(predictions, ground_truth):
    """
    Calculate accuracy of predictions.
    
    Args:
        predictions (list): List of predicted answers.
        ground_truth (list): List of ground truth answers.
        
    Returns:
        float: Accuracy score.
    """
    correct = sum(p == gt for p, gt in zip(predictions, ground_truth))
    return correct / len(predictions) if len(predictions) > 0 else 0

def calculate_accuracy_by_difficulty(df, prediction_cols, solution_col):
    """
    Calculate accuracy by difficulty level.
    
    Args:
        df (pd.DataFrame): DataFrame containing predictions and ground truth.
        prediction_cols (dict): Dictionary mapping difficulty levels to prediction column names.
        solution_col (str): Name of the column containing ground truth.
        
    Returns:
        dict: Dictionary mapping difficulty levels to accuracy scores.
    """
    results = {}
    for difficulty, col in prediction_cols.items():
        accuracy = calculate_accuracy(df[col], df[solution_col])
        results[difficulty] = accuracy
    return results

def plot_accuracy_comparison(accuracies, title="Accuracy Comparison", figsize=(10, 6)):
    """
    Plot accuracy comparison between different models or difficulty levels.
    
    Args:
        accuracies (dict): Dictionary mapping model/difficulty names to accuracy scores.
        title (str): Title of the plot.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(accuracies.keys())
    values = list(accuracies.values())
    
    bars = ax.bar(names, values, color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2%}', ha='center', va='bottom')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    return fig

def generate_confusion_matrix(predictions, ground_truth, labels=None):
    """
    Generate a confusion matrix for multiple-choice questions.
    
    Args:
        predictions (list): List of predicted answers.
        ground_truth (list): List of ground truth answers.
        labels (list, optional): List of label names. Defaults to ['A', 'B', 'C', 'D'].
        
    Returns:
        pd.DataFrame: Confusion matrix as a DataFrame.
    """
    if labels is None:
        labels = ['A', 'B', 'C', 'D']
    
    # Initialize confusion matrix
    confusion_matrix = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}
    
    # Fill confusion matrix
    for pred, true in zip(predictions, ground_truth):
        if pred in labels and true in labels:
            confusion_matrix[true][pred] += 1
    
    # Convert to DataFrame
    df_confusion = pd.DataFrame(confusion_matrix)
    
    return df_confusion
