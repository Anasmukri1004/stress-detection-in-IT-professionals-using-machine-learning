# AGENTS.md

This document provides instructions for developers and agents on how to work with the Stress Detection for IT Professionals repository.

## 1. Repository Overview

This project is a machine learning application designed to detect stress levels in IT professionals. It consists of a Flask frontend that serves a web interface and a backend that includes the machine learning model and data.

## 2. Environment Setup

To work with this project, you'll need to set up a Python environment with the necessary libraries.

### 2.1. Backend and Frontend Dependencies

The required libraries are listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes the following libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Flask
- joblib

### 2.2. Jupyter Notebook

The backend model is developed in a Jupyter Notebook. If you don't have Jupyter installed, you can install it via pip:

```bash
pip install jupyter
```

## 3. Running the Application

The Flask application serves the frontend of the project.

### 3.1. Running the Flask Server

To run the Flask application, navigate to the `FRONTEND` directory and execute the following command:

```bash
python app.py
```

This will start a local web server, and you can access the application by opening a web browser and navigating to `http://127.0.0.1:5000`.

## 4. Training the Models

The machine learning model for stress detection is trained using the Jupyter Notebook located in the `BACKEND` directory.

### 4.1. Training the Stress Detection Model

1.  **Navigate to the `BACKEND` directory.**
2.  **Start the Jupyter Notebook server:**
    ```bash
    jupyter notebook
    ```
3.  **Open the `stress.ipynb` notebook.**
4.  **Run all the cells in the notebook.** This will load the data, preprocess it, train the models (RandomForestRegressor, AdaBoostRegressor, ExtraTreeRegressor), and evaluate their performance.
5.  **The trained models are saved as pickled files** in the `FRONTEND` directory, which are then used by the Flask application.
