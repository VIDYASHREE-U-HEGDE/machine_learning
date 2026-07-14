import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------
# MLflow Configuration
# --------------------------------

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment(
    "Customer Churn Prediction"
)


# --------------------------------
# Dataset Path
# --------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "customer_churn.csv"
)


# --------------------------------
# Load Dataset
# --------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded")
print(df.head())


# --------------------------------
# Data Cleaning
# --------------------------------

if "CustomerID" in df.columns:
    df.drop(
        "CustomerID",
        axis=1,
        inplace=True
    )


if "customerID" in df.columns:
    df.drop(
        "customerID",
        axis=1,
        inplace=True
    )


if "TotalCharges" in df.columns:

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"].fillna(
        df["TotalCharges"].median(),
        inplace=True
    )


# --------------------------------
# Feature Target Split
# --------------------------------

X = df.drop(
    "Churn",
    axis=1
)

y = df["Churn"]



# --------------------------------
# Identify Columns
# --------------------------------

categorical_columns = X.select_dtypes(
    include="object"
).columns



# --------------------------------
# Preprocessing
# --------------------------------

preprocessor = ColumnTransformer(

    transformers=[

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )

    ],

    remainder="passthrough"

)



# --------------------------------
# Model
# --------------------------------

model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),


        (
            "classifier",
            RandomForestClassifier(

                n_estimators=200,

                random_state=42

            )

        )

    ]

)



# --------------------------------
# Train Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# --------------------------------
# MLflow Run
# --------------------------------

with mlflow.start_run():


    print("Training started...")


    # Train model

    model.fit(

        X_train,

        y_train

    )


    # Prediction

    prediction = model.predict(

        X_test

    )


    probability = model.predict_proba(

        X_test

    )[:,1]



    # Metrics

    accuracy = accuracy_score(

        y_test,

        prediction

    )


    precision = precision_score(

        y_test,

        prediction

    )


    recall = recall_score(

        y_test,

        prediction

    )


    f1 = f1_score(

        y_test,

        prediction

    )


    roc_auc = roc_auc_score(

        y_test,

        probability

    )



    print(
        classification_report(
            y_test,
            prediction
        )
    )


    print(
        "Accuracy:",
        accuracy
    )


    print(
        "ROC-AUC:",
        roc_auc
    )



    # --------------------------------
    # Confusion Matrix
    # --------------------------------

    cm = confusion_matrix(

        y_test,

        prediction

    )


    plt.figure(
        figsize=(6,4)
    )


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues"

    )


    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )


    plt.title(
        "Customer Churn Confusion Matrix"
    )


    plt.tight_layout()



    image_path = os.path.join(

        BASE_DIR,

        "confusion_matrix.png"

    )


    plt.savefig(

        image_path

    )


    plt.close()



    # --------------------------------
    # MLflow Logging
    # --------------------------------


    mlflow.log_param(

        "algorithm",

        "Random Forest"

    )


    mlflow.log_param(

        "n_estimators",

        200

    )


    mlflow.log_metric(

        "accuracy",

        accuracy

    )


    mlflow.log_metric(

        "precision",

        precision

    )


    mlflow.log_metric(

        "recall",

        recall

    )


    mlflow.log_metric(

        "f1_score",

        f1

    )


    mlflow.log_metric(

        "roc_auc",

        roc_auc

    )


    mlflow.log_artifact(

        image_path

    )



    # --------------------------------
    # Save Model
    # --------------------------------


    MODEL_PATH = os.path.join(

        BASE_DIR,

        "models",

        "model.pkl"

    )


    os.makedirs(

        os.path.dirname(MODEL_PATH),

        exist_ok=True

    )


    joblib.dump(

        model,

        MODEL_PATH

    )


    print(
        "Model saved:",
        MODEL_PATH
    )



    # --------------------------------
    # MLflow Model Save
    # --------------------------------


    mlflow.sklearn.log_model(

        sk_model=model,

        name="customer_churn_model",

        serialization_format="cloudpickle"

    )



print(
    "MLflow tracking completed"
)

print(
    "Training completed successfully"
)