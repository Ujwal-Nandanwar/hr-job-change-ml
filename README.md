# HR Job Change Prediction

An end-to-end machine learning project that explores whether a data
science candidate is likely to seek a job change. It combines data
preprocessing, exploratory analysis, individual classification models,
ensemble learning, model evaluation, and an interactive Streamlit
dashboard.

-   **Live application:** [Open the Streamlit
    dashboard](https://hr-job-change-ml.streamlit.app)
-   **Source code:** [View the GitHub
    repository](https://github.com/Ujwal-Nandanwar/hr-job-change-ml)

## Table of Contents

-   [Project Overview](#project-overview)
-   [Objectives](#objectives)
-   [Dataset](#dataset)
-   [Machine Learning Workflow](#machine-learning-workflow)
-   [Models Implemented](#models-implemented)
-   [Final Test-Set Results](#final-test-set-results)
-   [Interactive Dashboard](#interactive-dashboard)
-   [Project Structure](#project-structure)
-   [Run Locally](#run-locally)
-   [Deployment](#deployment)
-   [Understanding the Metrics](#understanding-the-metrics)
-   [Limitations and Responsible Use](#limitations-and-responsible-use)
-   [Future Improvements](#future-improvements)

## Project Overview

Organizations may want to understand which candidates are more likely to
seek a job change. This project uses the Kaggle **HR Analytics: Job
Change of Data Scientists** dataset to build and compare classification
models for its binary target.

The application accepts candidate characteristics and displays
predictions from four individual models and two ensemble models. It also
presents saved evaluation results, tuning experiments, decision
thresholds, and feature insights.

This is an educational and experimental machine learning project.
Predictions are estimates based on the dataset, not facts about an
individual.

## Objectives

-   Prepare and inspect a real-world tabular dataset.
-   Explore target distribution, feature characteristics, and data
    quality.
-   Build a consistent preprocessing and modelling workflow.
-   Train and evaluate multiple classification algorithms.
-   Experiment with hyperparameter tuning and decision thresholds.
-   Compare individual models with Soft Voting and Stacking ensembles.
-   Present saved results through a multi-section Streamlit dashboard.
-   Make the project accessible through GitHub and Streamlit Community
    Cloud.

## Dataset

-   **Dataset:** HR Analytics: Job Change of Data Scientists
-   **Source:** Kaggle

The target indicates whether a candidate belongs to the
job-change-seeking class. In the cleaned dataset, the target
distribution is:

| Target Class | Rows | Meaning |
|:--|--:|:--|
| `0` | 14,381 | Not in the job-change-seeking class |
| `1` | 4,777 | In the job-change-seeking class |
| **Total** | **19,158** | **Total number of records** |

The identifier `enrollee_id` is excluded from model features. The
cleaned dataset contains 15 columns: 14 input features and the target.

The modelling workflow uses a stratified split with 85% for development
and 15% for held-out testing, using `random_state=42`.

-   **Development split:** 16,284 rows
-   **Held-out test split:** 2,874 rows

The reported final metrics are calculated on the held-out test set.

## Machine Learning Workflow

1.  **Data inspection and exploratory data analysis:** Inspect feature
    types, missing values, target distribution, and relationships.
2.  **Preprocessing:** Prepare features for modelling. Saved individual
    estimators are scikit-learn Pipelines, so their preprocessing is
    applied during prediction.
3.  **Train/test split:** Use a stratified development split while
    retaining a separate test set.
4.  **Individual model training:** Train Logistic Regression, Decision
    Tree, K-Nearest Neighbors, and Random Forest.
5.  **Threshold selection:** Use saved model-specific probability
    thresholds to convert probabilities into class predictions. The
    threshold is not necessarily 0.50.
6.  **Ensemble construction:** Build Soft Voting and Stacking using
    out-of-fold base-model probabilities from a stratified five-fold
    process. Stacking uses Logistic Regression as its meta-model.
7.  **Evaluation:** Compare accuracy, precision, recall, F1-score,
    balanced accuracy, ROC-AUC, PR-AUC, log loss, and Brier score where
    available.
8.  **Dashboard integration:** Load saved models, metadata, thresholds,
    metrics, and the ensemble bundle into Streamlit. The app displays
    saved results; it does not retrain models when a visitor opens it.

## Models Implemented

### Individual models

-   **Logistic Regression:** Linear classification baseline.
-   **Decision Tree:** Learns feature-based decision rules.
-   **K-Nearest Neighbors (KNN):** Predicts based on nearby training
    examples.
-   **Random Forest:** Combines predictions from multiple decision
    trees.

### Ensemble models

-   **Soft Voting:** Averages predicted probabilities from the four
    individual models and applies its selected threshold.
-   **Stacking:** Uses base-model out-of-fold probability predictions as
    inputs to a Logistic Regression meta-model.

The saved ensemble bundle contains the information required by the app
to reproduce ensemble predictions, including the base models,
meta-model, probability-column configuration, and selected thresholds.

## Final Test-Set Results

The following results were obtained on the held-out test set. Values are rounded for readability.

| Model | Threshold | Accuracy | Precision | Recall | F1-score | Balanced Accuracy | ROC-AUC | PR-AUC |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| **Stacking** | 0.3017 | 0.7971 | 0.5714 | 0.7476 | **0.6477** | **0.7806** | 0.8109 | 0.5464 |
| Random Forest | 0.56 | 0.7954 | 0.5687 | 0.7448 | 0.6449 | 0.7785 | 0.8094 | 0.5443 |
| Soft Voting | 0.3886 | 0.7947 | 0.5678 | 0.7420 | 0.6433 | 0.7771 | **0.8120** | 0.5435 |
| Logistic Regression | 0.30 | 0.7888 | 0.5579 | 0.7392 | 0.6359 | 0.7722 | 0.8060 | 0.5450 |
| Decision Tree | 0.31 | 0.7867 | 0.5545 | 0.7378 | 0.6332 | 0.7704 | 0.8030 | 0.5343 |
| KNN | 0.34 | 0.7773 | 0.5390 | 0.7420 | 0.6244 | 0.7655 | 0.7986 | 0.5117 |

### Results summary

-   **Stacking has the highest F1-score** in this reported test
    comparison, at approximately **0.6477**.
-   **Soft Voting has the highest ROC-AUC**, at approximately
    **0.8120**.
-   Stacking's precision is approximately 57.1% and recall approximately
    74.8% at its selected threshold.
-   Differences between models are modest. These results do not
    establish that one model will be best for every future population.

The reported thresholds are model-specific. Accuracy, precision, recall,
F1-score, and balanced accuracy depend on the selected threshold;
ROC-AUC and PR-AUC evaluate ranking across thresholds.

## Interactive Dashboard

The Streamlit application contains four sections accessed through custom
horizontal navigation.

### 1. Prediction

-   Enter candidate characteristics.
-   Obtain predictions from four individual models and two ensemble
    models.
-   View estimated probabilities and selected thresholds.
-   Compare model predictions and their agreement.

### 2. Model Performance

-   Compare all six models on the held-out test set.
-   Review accuracy, precision, recall, F1-score, balanced accuracy,
    ROC-AUC, PR-AUC, log loss, and Brier score where available.
-   Explore saved evaluation plots and threshold information.

### 3. Hyperparameter Tuning

-   Review saved tuning results for individual models.
-   Inspect saved best hyperparameters.
-   Compare baseline and tuned PR-AUC results where available.
-   View ensemble configuration and compare ensemble results with
    individual models.
-   Review selected decision thresholds.

The ensemble section presents the saved configuration and final
evaluation results; it should not be interpreted as a separate
exhaustive ensemble hyperparameter search.

### 4. Feature Insights

-   Explore the feature insights and plots included in the project.
-   Treat these as exploratory model-analysis aids, not proof of
    causation.

### Interface features

-   Custom horizontal navigation.
-   Light and dark themes.
-   Interactive tables and charts.
-   Saved artifacts loaded by the app rather than retraining on every
    visit.

## Project Structure

The principal project files are organized as follows. The exact artifact
and plot filenames may evolve as experiments are updated.

``` text
hr-job-change-ml/
├── data/
│   ├── raw/
│   │   ├── aug_train.csv
│   │   └── aug_test.csv
│   └── processed/
│       └── clean_hr_job_change.csv
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_model_training_and_evaluation.ipynb
│   ├── 03_logistic_accuracy_experiment.ipynb
│   ├── 04_decision_tree_accuracy_experiment.ipynb
│   └── 05_ensemble_model.ipynb
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── metrics.py
│   └── train_models.py
├── app/
│   ├── streamlit_app.py
│   ├── views/
│   │   ├── prediction.py
│   │   ├── performance.py
│   │   ├── tuning.py
│   │   └── insights.py
│   ├── components/
│   │   ├── header.py
│   │   └── navigation.py
│   ├── utils/
│   │   ├── model_loader.py
│   │   └── prediction_utils.py
│   └── styles/
│       └── theme.css
├── artifacts/
│   ├── models/
│   ├── metrics/
│   ├── metadata/
│   ├── plots/
│   ├── tuning/
│   └── ensemble/
├── .gitignore
├── requirements.txt
└── README.md
```

The `artifacts/` directory contains trained models and saved outputs
required by the app. Keep all required files in the repository or adapt
the loading workflow to retrieve them from another storage location.

## Run Locally

### Prerequisites

-   A Python version compatible with the pinned dependencies and saved
    scikit-learn models.
-   The required saved model and metadata artifacts.

### 1. Clone the repository

``` bash
git clone https://github.com/Ujwal-Nandanwar/hr-job-change-ml
cd hr-job-change-ml
```

Alternatively, download the repository ZIP and extract it.

### 2. Create a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

``` bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use the committed `requirements.txt` as the source of dependency
versions. If model loading fails due to a scikit-learn or Python version
mismatch, use versions compatible with the saved artifacts.

### 4. Launch Streamlit

From the repository root:

``` bash
streamlit run app/streamlit_app.py
```

Streamlit prints a local URL in the terminal, commonly
`http://localhost:8501`.

## Deployment

The application is deployed using **Streamlit Community Cloud**, with
source code stored in GitHub.

## Understanding the Metrics

-   **Accuracy:** Fraction of all predictions that are correct.
-   **Precision:** Of candidates predicted as class `1`, the fraction
    that truly belong to class `1`.
-   **Recall:** Of all true class `1` candidates, the fraction
    identified by the model.
-   **F1-score:** Harmonic mean of precision and recall.
-   **Balanced accuracy:** Average recall across the two classes.
-   **ROC-AUC:** Measures how well the model ranks positive examples
    above negative examples across thresholds.
-   **PR-AUC:** Summarizes the precision-recall trade-off across
    thresholds; useful when class frequencies are uneven.
-   **Log loss:** Evaluates predicted probabilities, penalizing
    confident incorrect predictions.
-   **Brier score:** Mean squared error between predicted probabilities
    and binary outcomes.

A positive prediction means the model classifies the candidate as
belonging to the dataset's job-change-seeking class. It does **not**
prove the candidate will accept an offer or leave a job.

## Limitations and Responsible Use

-   Performance depends on the dataset, its labels, and the population
    represented.
-   A held-out test result is not a guarantee of real-world performance.
-   A probability is a model estimate, not certainty.
-   A false positive can incorrectly flag a candidate; a false negative
    can miss someone in the positive class.
-   The dataset may contain sampling, measurement, or historical biases.
-   This project is intended for education and experimentation, not as
    the sole basis for hiring, rejection, or other consequential
    employment decisions.
-   Any real-world use would require further validation, fairness
    assessment, privacy review, and domain oversight.

## Future Improvements

-   Add clearer explanations of individual predictions and feature
    contributions.
-   Expand validation with repeated or nested cross-validation where
    appropriate.
-   Evaluate calibration and threshold choices against explicit
    operational goals.
-   Add automated tests for preprocessing, artifact loading, and
    prediction consistency.
-   Track experiment configurations and model versions.
-   Improve deployment monitoring and error reporting.

## Project Links

-   **Live demo:** [https://hr-job-change-ml.streamlit.app](https://hr-job-change-ml.streamlit.app)
-   **GitHub repository:**
    [https://github.com/Ujwal-Nandanwar/hr-job-change-ml](https://github.com/Ujwal-Nandanwar/hr-job-change-ml)
