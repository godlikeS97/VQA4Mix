"""
Multiple choice question generation utilities for VQA4Mix project.
"""

import random
import os
from openai import OpenAI

def generate_random_choice():
    """
    Generate a random choice from A, B, C, D.
    
    Returns:
        str: A random choice (A, B, C, or D).
    """
    return random.choice(['A', 'B', 'C', 'D'])

def generate_multiple_choice_question(reference_caption, correct_choice, level='medium', api_key=None):
    """
    Generate a multiple choice question with distractors based on a reference caption.
    
    Args:
        reference_caption (str): The ground truth caption.
        correct_choice (str): The correct choice (A, B, C, or D).
        level (str): Difficulty level ('easy', 'medium', or 'hard').
        api_key (str, optional): OpenAI API key. If None, uses environment variable.
        
    Returns:
        str: The generated multiple choice question.
    """
    # Define the prompt based on difficulty level
    if level == 'easy':
        level_message = "The distractors are obviously incorrect but still loosely related to the context."
    elif level == 'medium':
        level_message = "The distractors are somewhat related to the context but contain inaccuracies or non-fluent language."
    elif level == 'hard':
        level_message = "The distractors are closely related to the context but may confuse someone without careful observation."

    prompt = f"""
    Given the ground truth caption below:
    "{reference_caption}"
    Generate three plausible but incorrect distractors.
    "{level_message}"
    Format the result as a multiple-choice question. 
    Question title should be "Which of the following captions best describes the painting?".
    The correct choice should be placed at choice "{correct_choice}". 
    Do not generate special symbols such as '*'.
    """

    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=200,
    )

    # Extract the generated multiple-choice question
    question = response.choices[0].message.content
    
    return question
