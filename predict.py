import pandas as pd
import joblib

model = joblib.load("../saved_model/model.pkl")

sample = {
    "Experience (Years)": 3,
    "Salary Expectation ($)": 50000,
    "Projects Count": 4,
    "AI Score (0-100)": 85,
    "Skills": "Python",
    "Education": "B.Tech",
    "Certifications": "AWS",
    "Job Role": "Data Scientist"
}

df = pd.DataFrame([sample])

prediction = model.predict(df)

print(prediction)