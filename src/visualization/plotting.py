"""
Visualization utilities for VQA4Mix project.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_confusion_matrix(confusion_matrix, title="Confusion Matrix", figsize=(8, 6)):
    """
    Plot a confusion matrix.
    
    Args:
        confusion_matrix (pd.DataFrame): Confusion matrix as a DataFrame.
        title (str): Title of the plot.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig

def plot_accuracy_by_category(accuracies, categories, title="Accuracy by Category", figsize=(12, 6)):
    """
    Plot accuracy by category.
    
    Args:
        accuracies (list): List of accuracy values.
        categories (list): List of category names.
        title (str): Title of the plot.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(categories, accuracies, color='skyblue')
    
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
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_sample_images(images, titles=None, figsize=(15, 10), rows=2, cols=3):
    """
    Plot a grid of sample images.
    
    Args:
        images (list): List of images to plot.
        titles (list, optional): List of titles for each image.
        figsize (tuple): Figure size.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.axis('off')
        
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])
    
    # Hide any unused subplots
    for i in range(len(images), rows * cols):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    return fig

def plot_model_comparison(model_results, metric='accuracy', title="Model Comparison", figsize=(10, 6)):
    """
    Plot a comparison of different models.
    
    Args:
        model_results (dict): Dictionary mapping model names to metric values.
        metric (str): Name of the metric being compared.
        title (str): Title of the plot.
        figsize (tuple): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = list(model_results.keys())
    values = list(model_results.values())
    
    bars = ax.bar(models, values, color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2%}', ha='center', va='bottom')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig
