name: ML Model CI/CD Pipeline

on:
  push:
    branches: [ main, master ]
    paths:
      - 'MLProject/**'
  pull_request:
    branches: [ main, master ]
    paths:
      - 'MLProject/**'
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12.3'
          cache: 'pip'

      - name: Install dependencies
        run: |
          conda env create -f MLProject/conda.yaml
          source $(conda info --base)/etc/profile.d/conda.sh
          conda activate mlflow-env

      - name: Run tests
        run: echo "No tests defined yet. Add your test commands here."


  train-model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12.3'
          cache: 'pip'

      - name: Setup Conda
        uses: conda-incubator/setup-miniconda@v2
        with:
          activate-environment: mlflow-env
          environment-file: MLProject/conda.yaml
          auto-activate-base: false

      - name: Install additional dependencies
        shell: bash -l {0}
        run: |
          conda info
          conda list
          pip install mlflow scikit-learn pandas numpy matplotlib seaborn

      - name: Train model and log to MLflow
        shell: bash -l {0}
        run: |
          cd MLProject
          export MLFLOW_TRACKING_URI="file:./mlruns"
          python modelling.py

      - name: Upload MLflow artifacts
        uses: actions/upload-artifact@v3
        with:
          name: mlflow-artifacts
          path: |
            MLProject/mlruns
            MLProject/MLmodel
            MLProject/conda.yaml
            MLProject/images/*.png

            
  deploy:
    needs: train-model
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master'
    steps:
      - uses: actions/checkout@v3

      - name: Download model artifacts
        uses: actions/download-artifact@v3
        with:
          name: model-artifacts
          path: MLProject/

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12.3'
          cache: 'pip'

      - name: Install Conda and dependencies
        run: |
          source $HOME/miniconda/etc/profile.d/conda.sh
          conda init bash
          conda env create -f MLProject/conda.yaml
          conda activate mlflow-env

      - name: Model registration step
        run: |
          source $HOME/miniconda/etc/profile.d/conda.sh
          conda activate mlflow-env
          # This step assumes you have a way to identify the run_id from the previous train-model job
          # and that your MLflow tracking server is accessible.
          # For simplicity, this example assumes the model is already logged and you just need to promote it.
          # You might need to adjust this based on how your MLflow tracking server is set up and how you want to register models.
          echo "Model registration step. Implement your MLflow model registration logic here."
          # Example: mlflow.register_model("runs:/<run_id>/model", "YourModelName")
        env:
          MLFLOW_TRACKING_URI: 'http://localhost:5000' # Replace with your MLflow tracking server URI if remote
          # MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_TRACKING_USERNAME }}
          # MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_TRACKING_PASSWORD }}