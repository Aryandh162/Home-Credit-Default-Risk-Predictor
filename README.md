🏦 Home Credit Default Risk Predictor
💡 A Machine Learning Web App for Credit Risk Analysis

This project predicts whether a loan applicant is likely to default or repay based on their financial, demographic, and credit history data.
Built with Flask, scikit-learn, and pandas, it provides both real-time predictions and batch CSV uploads for large-scale credit risk assessment — wrapped in a modern interface.

🚀 Features
🧠 ML-powered predictions using trained classification models
🎯 Single Prediction Mode: Enter applicant features manually and get instant results
📁 Batch Prediction Mode: Upload a CSV file for bulk predictions
📊 Automated preprocessing using a saved pipeline (preprocessor.pkl)
💾 Downloadable prediction results in CSV format

🧩 Tech Stack
Python, Flask (backend)
scikit-learn, pandas, NumPy (ML & data processing)
HTML5, Bootstrap 5, custom CSS (frontend)
JSON, Pickle, Jinja2 templates (for model + UI integration)

⚙️ How It Works
User inputs or uploads customer financial data.
Data is preprocessed using the trained pipeline (preprocessor.pkl).
Model (model.pkl) predicts credit risk (✅ Low Risk / 🚨 High Risk).
Results are displayed on the web interface or saved as a downloadable file.
