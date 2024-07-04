# Heart Disease Prediction using Neural Networks

This project aims to predict the likelihood of heart disease in patients using a neural network model implemented in PyTorch. The model is trained on a dataset of patient features and can be used to identify high-risk individuals for early intervention.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Heart disease prediction is an essential task in healthcare for early diagnosis and treatment. This project leverages a neural network to predict heart disease based on patient data.

## Dataset

The dataset used in this project contains various features such as age, sex, chest pain type, resting blood pressure, cholesterol levels, and more. The target variable indicates the presence or absence of heart disease.

You can download the dataset from [Kaggle] or use any other suitable dataset.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib (optional, for visualization)

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/heart-disease-prediction.git
    cd heart-disease-prediction
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Preprocess the data and train the model:
    ```bash
    python train.py
    ```

2. Evaluate the model:
    ```bash
    python evaluate.py
    ```

3. Make predictions on new data:
    ```python
    from model import HeartDiseaseNN
    import torch
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # Load the trained model
    model = HeartDiseaseNN(input_size=13)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # Example new data point
    new_data = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
    scaler = StandardScaler()
    new_data_scaled = scaler.transform(new_data)
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

    # Make prediction
    with torch.no_grad():
        prediction = model(new_data_tensor)
        print(f'Predicted probability of heart disease: {prediction.item():.4f}')
    ```

## Model Architecture

The neural network consists of three fully connected layers with ReLU activation functions. The final layer uses a sigmoid activation to output a probability.

## Training

The model is trained using the binary cross-entropy loss function and the Adam optimizer. Training data is scaled using StandardScaler.

## Evaluation

Model performance is evaluated using accuracy, precision, recall, and F1-score metrics.

## Results

Provide a summary of the model's performance on the test set, including accuracy, precision, recall, F1-score, and any other relevant metrics.

## Contributing

Contributions are welcome! Please create an issue or submit a pull request for any changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
