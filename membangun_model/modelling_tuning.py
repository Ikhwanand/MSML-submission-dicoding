import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns 
import mlflow 
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib
import optuna


# Load dataset
train_df = pd.read_csv('../data/customer_churn_dataset-training-master.csv')
test_df = pd.read_csv('../data/customer_churn_dataset-testing-master.csv')

preprocessor = joblib.load('../models/preprocessor.pkl')

train_df_preprocessing = preprocessor.fit_transform(train_df.drop(columns=['CustomerID', 'Churn']))
test_df_preprocessing = preprocessor.transform(test_df.drop(columns=['CustomerID', 'Churn']))

# train_df_preprocessing = pd.DataFrame(train_df_preprocessing)
# test_df_preprocessing = pd.DataFrame(test_df_preprocessing)

# train_df_preprocessing.to_csv('./preprocessing_dataset/train_preprocessing.csv', index=False)
# test_df_preprocessing.to_csv('./preprocessing_dataset/test_preprocessing.csv', index=False)

# Set experiment name
experiment_name = "Logistic_Regression_Optuna_Python_Script"
mlflow.set_experiment(experiment_name)

print(f"MLflow experiment '{experiment_name}' is ready!")
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


# Prepare Data for Modeling
X_train = train_df_preprocessing
y_train = train_df['Churn'].fillna(0) # Handle missing values

X_test = test_df_preprocessing
y_test = test_df['Churn'].fillna(0)


print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print("Target distribution in training set:")
print(y_train.value_counts(normalize=True))


def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test, params=None):
    """
    Train model and log everything to MLflow
    """
    with mlflow.start_run(run_name=model_name):
        if params:
            mlflow.log_params(params)
        else:
            mlflow.autolog()
            
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Log metrics
        mlflow.log_metric('train_accuracy', train_accuracy)
        mlflow.log_metric('test_accuracy', test_accuracy)
        
        # Classification report
        train_report = classification_report(y_train, y_train_pred, output_dict=True)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        # Log additional metrics
        mlflow.log_metric("train_precision", train_report['weighted avg']['precision'])
        mlflow.log_metric('train_recall', train_report['weighted avg']['recall'])
        mlflow.log_metric('train_f1', train_report['weighted avg']['f1-score'])
        
        mlflow.log_metric('test_precision', test_report['weighted avg']['precision'])
        mlflow.log_metric('test_recall', test_report['weighted avg']['recall'])
        mlflow.log_metric('test_f1', test_report['weighted avg']['f1-score'])
        
        # Create and log confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'./images/confusion_matrix_{model_name}.png')
        mlflow.log_artifact(f'./images/confusion_matrix_{model_name}.png')
        plt.close()
        
        # Log model
        signature = infer_signature(X_train, y_train_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=f"churn_predicted_{model_name}"
        )
        
        print(f"\n{'='*50}")
        print(f"MODEL: {model_name}")
        print(f"{'='*50}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_report['weighted avg']['precision']:.4f}")
        print(f"Test Recall: {test_report['weighted avg']['recall']:.4f}")
        print(f"Test F1-Score: {test_report['weighted avg']['f1-score']:.4f}")
        
        return model, test_accuracy



def objective_logistic_regression(trial):
    C = trial.suggest_float('C', 0.001, 10.0, log=True)
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    max_iter = trial.suggest_int('max_iter', 100, 1000)
    
    model = LogisticRegression(C=C, solver=solver, penalty=penalty, max_iter=max_iter)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    with mlflow.start_run(run_name=f"Logistic_Regression_Optuna_Trial_{trial.number}"):
        mlflow.log_params({
            'C':C,
            'solver': solver,
            'penalty': penalty,
            'max_iter': max_iter
        })
        mlflow.log_metric('test_accuracy', accuracy)
        mlflow.log_metric('trial_number', trial.number)
    return accuracy


print("Starting hyperparameter optimization with Optuna...")
study_lr = optuna.create_study(direction='maximize')

print("Optimizing Logistic Regression Hyperparameters...")
study_lr.optimize(objective_logistic_regression, n_trials=50)

print("\nBest Logistic Regression parameters:")
print(study_lr.best_params)
print(f"Best accuracy: {study_lr.best_value:.4f}")


best_params = study_lr.best_params
final_model = LogisticRegression(**best_params, random_state=42)

best_train, best_score = train_and_log_model(
    model=final_model,
    model_name='Logistic_Regression_Optimized',
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    params=best_params
)

print("\n🏆 BEST MODEL PERFORMANCE:")
print(f"Optimized Random Forest Accuracy: {best_score:.4f}")