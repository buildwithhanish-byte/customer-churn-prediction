from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("model/churn_model.pkl")
model_columns = joblib.load("model/model_columns.pkl")

from typing import Literal
from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]

class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(..., ge=0, le=100)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
   
def preprocess_input(data: CustomerData):
    df = pd.DataFrame([data.model_dump()])

    # Consolidate "No phone/internet service" into "No"
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        df[col] = df[col].replace({'No phone service': 'No', 'No internet service': 'No'})

    # Binary encode: same 0/1 mapping the notebook derived from sorted(unique())
    binary_map = {'No': 0, 'Yes': 1}
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                   'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

    # One-hot encode manually — pd.get_dummies is unsafe on a single-row
    # input because it only creates columns for categories present in
    # that one row, not all categories the model was trained on
    df['InternetService_Fiber optic'] = int(df['InternetService'].iloc[0] == 'Fiber optic')
    df['InternetService_No'] = int(df['InternetService'].iloc[0] == 'No')

    df['Contract_One year'] = int(df['Contract'].iloc[0] == 'One year')
    df['Contract_Two year'] = int(df['Contract'].iloc[0] == 'Two year')

    df['PaymentMethod_Credit card (automatic)'] = int(df['PaymentMethod'].iloc[0] == 'Credit card (automatic)')
    df['PaymentMethod_Electronic check'] = int(df['PaymentMethod'].iloc[0] == 'Electronic check')
    df['PaymentMethod_Mailed check'] = int(df['PaymentMethod'].iloc[0] == 'Mailed check')

    df = df.drop(columns=['InternetService', 'Contract', 'PaymentMethod'])

        # Feature 1: TotalServices (sum of 6 add-on service flags)
    service_sum_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                         'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['TotalServices'] = df[service_sum_cols].sum(axis=1)

    # Feature 2: TenureGroup (bucketed tenure, one-hot encoded)
    bins = [-1, 12, 24, 48, 72]
    labels = ['0-1yr', '1-2yr', '2-4yr', '4-6yr']
    tenure_group = pd.cut(df['tenure'], bins=bins, labels=labels).iloc[0]

    df['TenureGroup_1-2yr'] = int(tenure_group == '1-2yr')
    df['TenureGroup_2-4yr'] = int(tenure_group == '2-4yr')
    df['TenureGroup_4-6yr'] = int(tenure_group == '4-6yr')

    # Guarantee identical column set/order to what the model was trained on
    df = df.reindex(columns=model_columns, fill_value=0)

    return df

@app.post("/predict")
def predict(data: CustomerData):
    try:
        processed = preprocess_input(data)
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]

        return {
            "prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(float(probability), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))