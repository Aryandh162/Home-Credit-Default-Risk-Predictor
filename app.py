from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
import os
import json
from src.utils.main_utils import MainUtils
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

# ===============================
# Load top features and models
# ===============================
with open("artifacts/top_features.json", "r") as f:
    feature_info = json.load(f)

top_features = feature_info["top_features"]
feature_means = feature_info["feature_means"]

preprocessor = MainUtils().load_object("artifacts/preprocessor.pkl")
model = MainUtils().load_object("artifacts/model.pkl")
all_features = preprocessor['numeric_cols'] + preprocessor['categorical_columns']


# ===============================
# Home Route - Single Prediction
# ===============================
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error_msg = None  # ✅ FIX: Add variable for handling input errors

    if request.method == 'POST':
        try:
            user_input = {}
            for f in top_features:
                val = request.form.get(f)
                if val and val.strip() != "":
                    try:
                        user_input[f] = float(val)
                    except ValueError:
                        user_input[f] = feature_means[f]  # fallback to mean if invalid
                else:
                    user_input[f] = feature_means[f]

            full_input = {f: user_input.get(f, feature_means[f]) for f in all_features}
            input_df = pd.DataFrame([full_input])

            numeric_cols = preprocessor['numeric_cols']
            categorical_cols = preprocessor['categorical_columns']

            X_num = preprocessor['numeric_pipeline'].transform(input_df[numeric_cols])
            X_cat = pd.get_dummies(input_df[categorical_cols], drop_first=True)
            X_cat = X_cat.reindex(columns=preprocessor['categorical_columns'], fill_value=0)

            X_processed = np.hstack([X_num, X_cat.values])
            prediction = model.predict(X_processed)[0]

        except Exception as e:
            error_msg = f"Error during prediction: {e}"

    return render_template('form.html', top_features=top_features, prediction=prediction, error_msg=error_msg)


# ===============================
# Batch Prediction Route
# ===============================
@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    output_filename = None
    error_msg = None

    if request.method == 'POST':
        try:
            file = request.files['file']
            if not file:
                error_msg = "⚠️ No file selected."
            else:
                upload_dir = os.path.join("artifacts", "prediction_artifacts")
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, file.filename)
                file.save(file_path)

                pipeline = PredictPipeline()
                output_path = pipeline.predict_from_csv(file_path)

                # ✅ FIX: Extract only filename for proper download route
                output_filename = os.path.basename(output_path)

        except Exception as e:
            error_msg = f"Error: {str(e)}"

    return render_template('upload.html', output_filename=output_filename, error_msg=error_msg)


# ===============================
# File Download Route
# ===============================
@app.route('/download/<filename>')
def download_file(filename):
    """
    Download generated prediction CSV
    """
    try:
        directory = os.path.join("artifacts", "predictions")
        file_path = os.path.join(directory, filename)

        if os.path.exists(file_path):
            print(f"✅ File ready for download: {file_path}")
            return send_from_directory(directory, filename, as_attachment=True)
        else:
            print(f"❌ File not found: {file_path}")
            return f"Error: File '{filename}' not found.", 404

    except Exception as e:
        print("❌ Error in download route:", e)
        return "Internal Server Error", 404


# ===============================
# Run the Flask App
# ===============================
if __name__ == '__main__':
    print("🚀 Flask App is running at: http://127.0.0.1:5000")
    app.run(debug=True)
