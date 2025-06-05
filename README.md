# Customer Churn Prediction Project

This project focuses on predicting customer churn using machine learning techniques, with an emphasis on data preprocessing, model training, hyperparameter tuning, and MLOps practices using MLflow and GitHub Actions.

## Project Structure

The project is organized into several key directories:

- `./preprocessing/`: Contains Jupyter notebooks and Python scripts related to data preprocessing, exploratory data analysis (EDA), and feature engineering. This includes the `EKSPERIMEN_Ikhwananda-siswa.ipynb` notebook for detailed analysis and `automate_Ikhwananda-siswa.py` for automated preprocessing.
- `./membangun_model/`: Houses the core modeling scripts, including `modelling.py` for training various models and `modelling_tuning.py` for hyperparameter tuning using Optuna. It also contains subdirectories for images and preprocessed datasets.
- `./MLProject/`: This directory is set up as an MLflow project, containing `conda.yaml` for environment management, `requirements.txt` for Python dependencies, `modelling.py` (a copy or link to the main modeling script), and MLflow-related artifacts like `MLmodel`.
- `./data/`: Stores the raw datasets (`customer_churn_dataset-training-master.csv` and `customer_churn_dataset-testing-master.csv`).
- `./models/`: Contains saved machine learning models and preprocessors (e.g., `preprocessor.pkl`).
- `./.workflow`: Contains the GitHub Actions workflow definition for CI/CD.

## Setup and Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd submission-akhir
    ```

2.  **Create a Conda environment (recommended):**
    ```bash
    conda env create -f MLProject/conda.yaml
    conda activate mlflow-env
    ```
    Alternatively, you can install dependencies using pip:
    ```bash
    pip install -r MLProject/requirements.txt
    ```

3.  **Install Optuna (if not included in requirements):**
    ```bash
    pip install optuna
    ```

## Running the Project

### Data Preprocessing and EDA

Open and run the `EKSPERIMEN_Ikhwananda-siswa.ipynb` notebook located in the `./preprocessing/` directory. This notebook guides you through data loading, quality assessment, EDA, and the creation of the preprocessing pipeline.

### Model Training and Hyperparameter Tuning

To train models and perform hyperparameter tuning, you can run the Python scripts:

-   **For basic model training:**
    ```bash
    python membangun_model/modelling.py
    ```
-   **For hyperparameter tuning with Optuna:**
    ```bash
    python membangun_model/modelling_tuning.py
    ```

These scripts will log experiments to MLflow.

### MLflow Tracking

To view the MLflow UI and track your experiments, run the following command in your terminal from the project root directory:

```bash
mlflow ui
```

Then, open your web browser and navigate to `http://localhost:5000`.

### GitHub Actions

The project includes a GitHub Actions workflow defined in `./.workflow`. This workflow automates testing, model training, and deployment processes. It is triggered on pushes/pull requests to `main`/`master` branches within the `MLProject` folder or can be manually dispatched.

-   **`test` job**: Runs tests for the MLProject.
-   **`train-model` job**: Trains the model and uploads artifacts.
-   **`deploy` job**: Registers the model to MLflow registry (on `main`/`master` branches).

Ensure your GitHub repository is set up correctly for the workflow to run.