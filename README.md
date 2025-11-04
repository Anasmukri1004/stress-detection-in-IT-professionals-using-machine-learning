# Stress Detection in IT Professionals Using Machine Learning

## 1. Project Overview

The COVID-19 pandemic has drastically altered people’s perspectives on healthcare constraints and lifestyles. With the widespread impact of the virus, educational institutions worldwide closed to curb the disease’s spread, affecting food availability and medical facilities. Surveys have shown that stress levels have increased due to concerns about job security, family health, and academic performance. Prolonged work hours and tight deadlines further exacerbate stress, leading to heart and muscle-related issues.

This project aims to create a model for predicting stress levels in IT professionals using machine learning algorithms and data science techniques. By monitoring various psychological parameters, the model will categorize individuals based on their stress levels, providing insights for personalized stress management. The goal is to offer a proactive approach to mental health, helping IT professionals manage stress effectively and improve their overall well-being.

## 2. Getting Started

This section will guide you through setting up the project on your local machine.

### 2.1. Prerequisites

Make sure you have Python installed on your system. You will also need `pip` to install the required libraries.

### 2.2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/stress-detection-it-professionals.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd stress-detection-it-professionals
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 3. Usage

This section explains how to run the application and train the models.

### 3.1. Running the Web Application

To run the Flask web application, navigate to the `FRONTEND` directory and execute the following command:

```bash
python app.py
```

Open your web browser and go to `http://127.0.0.1:5000` to use the application. The web interface allows you to input various parameters and get a stress level prediction.

### 3.2. Training the Machine Learning Models

The machine learning models are trained using a Jupyter Notebook.

1.  **Navigate to the `BACKEND` directory.**
2.  **Start the Jupyter Notebook server:**
    ```bash
    jupyter notebook
    ```
3.  **Open and run the `stress.ipynb` notebook.** This will train the models and save them as pickled files in the `FRONTEND` directory.

## 4. Project Structure

-   `BACKEND/`: Contains the Jupyter Notebook (`stress.ipynb`) for model training and the dataset.
-   `FRONTEND/`: Contains the Flask web application (`app.py`), HTML templates, and the trained model files.
-   `Data set/`: Contains the raw dataset used for training the model.
-   `README.md`: This file, providing an overview of the project.
-   `AGENTS.md`: Provides instructions for developers and agents.

## 5. Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.
