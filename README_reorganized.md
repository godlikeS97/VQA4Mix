# VQA4Mix: Evaluating the Limitations of Generative Models in Image Captioning

VQA4Mix is a research project focused on evaluating the limitations of generative models in image captioning through Visual Question Answering (VQA) with multiple-choice questions across different image categories (food, painting, people, and cat).

## Research Focus

This project investigates the limitations of generative AI models in image captioning by:

1. **Evaluating model performance across diverse image categories**: Testing how well models perform on different types of images (food, paintings, people, and cats)
2. **Assessing robustness to image variations**: Analyzing how image augmentations affect model performance
3. **Measuring accuracy across difficulty levels**: Testing models with easy, medium, and hard multiple-choice questions
4. **Comparing performance across different visual domains**: Identifying which categories are more challenging for current models

## Project Structure

The project has been reorganized to combine code from different categories (food, painting, people, cat) into a unified structure:

```
VQA4Mix/
├── data/                  # All datasets
│   ├── food/              # Food images and annotations
│   ├── painting/          # Painting images and annotations
│   ├── people/            # People images and annotations
│   └── cat/               # Cat images and annotations
├── notebooks/             # Jupyter notebooks
│   ├── unified_pipeline.ipynb     # Combined pipeline for all categories
│   ├── data_preprocessing.ipynb   # Data preprocessing utilities
│   └── result_analysis.ipynb      # Analysis of results
├── src/                   # Source code
│   ├── data_processing/   # Data loading and processing utilities
│   ├── model/             # Model-related code
│   ├── utils/             # Utility functions
│   └── visualization/     # Visualization utilities
├── README.md
└── LICENSE
```

## Modules

### Data Processing

The `src/data_processing` module provides utilities for loading and processing data:

- `data_loader.py`: Functions for loading and saving JSON data

### Model

The `src/model` module contains model-related code:

- `question_generator.py`: Functions for generating multiple-choice questions
- `inference.py`: Functions for performing inference with the LLaVA model

### Utils

The `src/utils` module provides utility functions:

- `image_augmentation.py`: Functions for augmenting images
- `evaluation.py`: Functions for evaluating model performance

### Visualization

The `src/visualization` module provides visualization utilities:

- `plotting.py`: Functions for plotting results and visualizations

## Notebooks

### Unified Pipeline

The `notebooks/unified_pipeline.ipynb` notebook combines the functionality from all category-specific pipelines into a single workflow. It includes:

- Loading data from all categories
- Generating multiple-choice questions
- Performing inference with the LLaVA model
- Evaluating results

### Data Preprocessing

The `notebooks/data_preprocessing.ipynb` notebook handles data preprocessing tasks:

- Loading and examining data from different categories
- Standardizing data formats
- Cleaning and preprocessing data
- Saving processed data for further use

### Result Analysis

The `notebooks/result_analysis.ipynb` notebook analyzes the results:

- Loading and examining results from different categories
- Calculating accuracy metrics
- Visualizing results
- Comparing performance across categories and difficulty levels

## Usage

1. **Data Preprocessing**:
   - Run `notebooks/data_preprocessing.ipynb` to preprocess data from all categories

2. **Unified Pipeline**:
   - Run `notebooks/unified_pipeline.ipynb` to generate multiple-choice questions and perform inference

3. **Result Analysis**:
   - Run `notebooks/result_analysis.ipynb` to analyze the results

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-image
- PIL

## Methodology

The project employs a systematic approach to evaluate generative models:

1. **Data Collection**: Gathering diverse images across four categories (food, paintings, people, cats)
2. **Question Generation**: Creating multiple-choice questions at three difficulty levels (easy, medium, hard)
3. **Model Evaluation**: Testing state-of-the-art vision-language models (e.g., LLaVA) on the multiple-choice questions
4. **Image Augmentation**: Applying various transformations to test model robustness
5. **Performance Analysis**: Analyzing accuracy across categories, difficulty levels, and with/without augmentations

## Key Findings

Preliminary results indicate:

- Performance varies significantly across image categories
- Models struggle more with certain visual domains (e.g., paintings) compared to others
- Image augmentations can significantly impact model performance
- The difficulty level of questions strongly correlates with model accuracy

## License

[MIT License](LICENSE)
