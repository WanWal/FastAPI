from fastapi import FastAPI 
from pydantic import BaseModel
import pickle
import pandas as pd

# Load model
with open("obesity_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class ObesityInput(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

@app.post("/predict")
def predict(data: ObesityInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    le = ['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
       'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I',
       'Overweight_Level_II']
    prediction = le[int(prediction[0])]
    return {"prediction": prediction}
