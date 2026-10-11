
# HR Job Change Prediction

An interactive machine learning dashboard that predicts whether a
data science candidate is likely to seek a job change.

## Features

- Interactive candidate prediction
- Individual model and ensemble predictions
- Model performance comparison
- Hyperparameter tuning results
- Feature insights
- Light and dark themes

## Models

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest
- Soft Voting Ensemble
- Stacking Ensemble

## Performance Highlights

The evaluated Stacking model achieved an F1-score of approximately
0.6477 on the held-out test set.

The Soft Voting ensemble achieved a ROC-AUC of approximately 0.8120
on the held-out test set.

These are experimental results on the selected dataset, not
guarantees of performance on new candidates.

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/streamlit_app.py
```

## Dataset

HR Analytics: Job Change of Data Scientists, sourced from Kaggle.

## Disclaimer

This project is for educational and research purposes. Predictions
should not be used as the sole basis for employment decisions.
