# Machine Learning Projects

This repository contains implementations of various machine learning models using TensorFlow/Keras. The projects demonstrate both regression and classification tasks.

## Projects

### 1. Health Insurance Cost Prediction
(`LinearR.ipynb`)

A neural network model that predicts health insurance costs based on various factors including:
- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

**Technical Details:**
- Architecture: Dense Neural Network with 2 hidden layers (64 units each)
- Optimization: Adam optimizer
- Loss: Mean Squared Error (MSE)
- Metrics: Mean Absolute Error (MAE)
- Dataset: Insurance cost dataset from FreeCodeCamp

### 2. Cat vs Dog Image Classification
(`DogOrCat.ipynb`)

A convolutional neural network (CNN) that classifies images as either cats or dogs.

**Technical Details:**
- Architecture: CNN with:
  - 3 Convolutional layers with MaxPooling
  - Dense layers for classification
  - Binary output with sigmoid activation
- Image preprocessing:
  - Rescaling (1/255)
  - Data augmentation (rotation, shift, zoom, flip)
- Image size: 150x150 pixels
- Batch size: 128
- Dataset: Cats and Dogs dataset from FreeCodeCamp

## Setup and Dependencies

Required libraries:
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- TensorFlow Docs (for visualization)

```bash
pip install tensorflow numpy pandas matplotlib
pip install -q git+https://github.com/tensorflow/docs
```

## Usage

Each notebook can be run in Google Colab or locally. The notebooks include:
- Data preprocessing
- Model creation and training
- Evaluation and visualization of results

## Data Sources

- Health Insurance dataset: https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
- Cats and Dogs dataset: https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip

## Performance Metrics

- Insurance Cost Prediction: Model aims for MAE < 3500
- Cat vs Dog Classification: Model aims for accuracy >= 63%

## Future Improvements

Potential areas for enhancement:
1. Implement cross-validation
2. Try different model architectures
3. Add more data augmentation techniques
4. Experiment with transfer learning
5. Implement model deployment examples

## Contributing

Feel free to open issues or submit pull requests for improvements to either project.
