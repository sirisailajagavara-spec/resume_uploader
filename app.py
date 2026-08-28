from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import traceback
import os
import sys

# Safe print to prevent OSError on Windows background redirection
def print_safe(*args, **kwargs):
    try:
        import builtins
        builtins.print(*args, **kwargs)
    except Exception:
        pass

print = print_safe


app = Flask(__name__)
CORS(app)

# Load model
try:
    import sklearn
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    print("--- SERVER STARTUP ---")
    print("Flask app sklearn version:", sklearn.__version__)
    print("Loading model from absolute path:", model_path)
    model = joblib.load(model_path)
    print("Model loaded successfully. Type:", type(model))
    print("----------------------")
except Exception as e:
    print("Model loading failed:", e)
    model = None


@app.route("/")
def home():
    # Serve index.html from the current directory
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        print("INPUT RECEIVED:", data)

        # Support both original dataset headers and friendly frontend keys
        experience = data.get("Experience (Years)") if "Experience (Years)" in data else data.get("experience")
        salary = data.get("Salary Expectation ($)") if "Salary Expectation ($)" in data else data.get("salary")
        projects = data.get("Projects Count") if "Projects Count" in data else data.get("projects")
        score = data.get("AI Score (0-100)") if "AI Score (0-100)" in data else data.get("score")
        skills = data.get("Skills") if "Skills" in data else data.get("skills")
        education = data.get("Education") if "Education" in data else data.get("education")
        certifications = data.get("Certifications") if "Certifications" in data else data.get("certifications")
        role = data.get("Job Role") if "Job Role" in data else data.get("role")

        # Detect auto-calculate flag from frontend
        auto_calculate = data.get("auto_calculate", False)

        # Cast/Clean inputs to avoid pipeline failures
        try:
            experience = float(experience) if experience is not None else 0.0
            salary = float(salary) if salary is not None else 0.0
        except ValueError as val_err:
            return jsonify({"error": f"Invalid numerical inputs: {str(val_err)}"}), 400

        skills = str(skills) if skills is not None else ""
        education = str(education) if education is not None else ""
        certifications = str(certifications) if certifications is not None else ""
        role = str(role) if role is not None else ""

        # Heuristic Imputation if auto_calculate is enabled or fields are missing
        if auto_calculate or projects is None:
            # Estimate Projects Count: roughly 1 project per year of experience, capped
            projects = max(1, min(10, int(experience))) if experience > 0 else 3
        else:
            try:
                projects = int(projects)
            except (ValueError, TypeError):
                projects = 3

        if auto_calculate or score is None:
            # Estimate AI Score (0-100) based on qualifications (baseline: 35)
            calculated_score = 35.0
            
            # Experience points
            calculated_score += min(20.0, experience * 2.0)
            
            # Education bonuses
            edu_lower = education.lower()
            if "phd" in edu_lower or "doctor" in edu_lower:
                calculated_score += 15.0
            elif "m.tech" in edu_lower or "mba" in edu_lower or "master" in edu_lower:
                calculated_score += 10.0
            elif "b.tech" in edu_lower or "b.sc" in edu_lower or "bachelor" in edu_lower:
                calculated_score += 5.0

            # Certifications bonuses
            cert_lower = certifications.lower()
            if "google ml" in cert_lower or "google machine learning" in cert_lower:
                calculated_score += 10.0
            if "deep learning" in cert_lower:
                calculated_score += 10.0
            if "aws" in cert_lower:
                calculated_score += 10.0
            
            # Skills bonuses (matching key keywords in CSV)
            skills_lower = skills.lower()
            key_skills = ["python", "tensorflow", "pytorch", "nlp", "machine learning", "deep learning", "sql", "wireshark", "ethical hacking"]
            skills_matched = sum(1 for s in key_skills if s in skills_lower)
            calculated_score += min(15.0, skills_matched * 5.0)

            # Cap AI Score between 10 and 100
            score = max(10.0, min(100.0, calculated_score))
        else:
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = 80.0

        # Map frontend -> model training columns
        input_data = {
            "Experience (Years)": experience,
            "Salary Expectation ($)": salary,
            "Projects Count": projects,
            "AI Score (0-100)": score,
            "Skills": skills,
            "Education": education,
            "Certifications": certifications,
            "Job Role": role
        }


        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        print("DATAFRAME COLUMNS:", df.columns)
        print("DATAFRAME VALUE:\n", df)

        # Predict
        prediction = model.predict(df)[0]

        return jsonify({
            "prediction": str(prediction),
            "projects": int(projects),
            "score": float(score)
        })

    except Exception as e:
        print("ERROR IN PREDICT ENDPOINT:", str(e))
        try:
            traceback.print_exc()
        except Exception:
            pass
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=False, port=5001)