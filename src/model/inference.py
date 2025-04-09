"""
Model inference utilities for VQA4Mix project.
"""

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import skimage.io as io
from PIL import Image
import numpy as np

def load_llava_model(model_path, device='cuda:0', load_in_4bit=True):
    """
    Load the LLaVA model.
    
    Args:
        model_path (str): Path to the model.
        device (str): Device to load the model on.
        load_in_4bit (bool): Whether to load the model in 4-bit precision.
        
    Returns:
        tuple: (processor, model)
    """
    processor = LlavaNextProcessor.from_pretrained(model_path)
    
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        load_in_4bit=load_in_4bit
    )
    
    model.to(device)
    
    return processor, model

def perform_multiple_choice_task(processor, model, img_path, question, device='cuda:0', max_new_tokens=150):
    """
    Perform a multiple-choice task using the LLaVA model.
    
    Args:
        processor: The LLaVA processor.
        model: The LLaVA model.
        img_path (str): Path to the image.
        question (str): The multiple-choice question.
        device (str): Device to run inference on.
        max_new_tokens (int): Maximum number of tokens to generate.
        
    Returns:
        str: The model's answer.
    """
    image = io.imread(img_path)
    
    conversation = [
        {
          "role": "user",
          "content": [
              {"type": "text", "text": question + "\\nOnly return the correct choice with a single letter."},
              {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    # Autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output = processor.decode(output[0], skip_special_tokens=True)

    # Extract the answer
    mcq_answer = output.split('[/INST]')[1].strip()
    return mcq_answer

def batch_inference(processor, model, image_paths, questions, device='cuda:0', max_new_tokens=150):
    """
    Perform batch inference on multiple images.
    
    Args:
        processor: The LLaVA processor.
        model: The LLaVA model.
        image_paths (list): List of paths to images.
        questions (list): List of questions corresponding to each image.
        device (str): Device to run inference on.
        max_new_tokens (int): Maximum number of tokens to generate.
        
    Returns:
        list: List of model answers.
    """
    answers = []
    
    for img_path, question in zip(image_paths, questions):
        answer = perform_multiple_choice_task(
            processor, model, img_path, question, device, max_new_tokens
        )
        answers.append(answer)
    
    return answers
