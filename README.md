
# Customer Churn Prediction

This is a small machine learning project that predicts whether a telecom customer is likely to leave. It uses the Telco Customer Churn dataset, a Decision Tree model, and a FastAPI endpoint for making predictions.

## GitHub Repository:
https://github.com/buildwithhanish-byte/customer-churn-prediction

## Folder structure

```bash
my_churn_project/
├── data/TelcoCustomerChurn.csv
├── notebook/churn_analysis.ipynb
├── model/churn_model.pkl, model_columns.pkl
├── app.py
├── requirements.txt
├── sample_request.json
└── README.md
```

## Setup

Install the required packages from the main project folder:

```bash
pip install -r requirements.txt
```

## Run the notebook

Open `notebook/churn_analysis.ipynb` in Jupyter or VS Code, then run all the cells. This will reproduce the analysis and create the model files again.

## Run the API

From the main project folder, run:

```bash
uvicorn app:app --reload
```

The API can be tested in the browser at `http://127.0.0.1:8000/docs`. A request can also be sent directly to `http://127.0.0.1:8000/predict` using the data in `sample_request.json`.

## Model

The model is a Decision Tree Classifier with `class_weight='balanced'` and `max_depth=5`. An unrestricted decision tree performed very well on the training data but did not generalize as well to new data. Setting a maximum depth of 5 reduced that overfitting. The balanced class weights were used because the dataset contains fewer churn cases than non-churn cases.

The final model reached about **71.6% accuracy** on the test data. Recall for customers who churned was about **78.3%**. Recall was given more importance than precision because missing a customer who is likely to leave can cost more than making an unnecessary retention offer.

## Main finding

Contract type had the strongest relationship with churn. Customers with month-to-month contracts left much more often than customers on one-year or two-year contracts. Fiber-optic service and low tenure were also useful churn indicators. Based on these results, longer-contract incentives and extra support for newer customers could be useful retention strategies.

### Sample request
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

### Sample response
```json
{
  "prediction": "Yes",
  "churn_probability": 0.82
}
```
