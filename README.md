# credit-card-fraud-predictor

# Machine Learning Full-Stack Web Application

A full-stack web application consisting of a FastAPI backend utilizing pre-trained scikit-learn models and a local HTML/JavaScript frontend to deliver real-time predictions.

## Project Structure
- `main.py`: FastAPI backend application with endpoints for regression and classification predictions.
- `index.html`: Frontend user interface with interactive input forms.
- `regression_model.pkl`: Trained linear regression model saved via joblib.
- `classification_model.pkl`: Trained classification model saved via joblib.

## How to Run the Application

1. Start the Backend Server (FastAPI)
Open a terminal in your project directory.

Install required packages (if not already installed):

Bash
pip install fastapi uvicorn scikit-learn joblib pydantic
Start the FastAPI backend:

Bash
uvicorn main:app --reload --port 5000 --host 127.0.0.1
(You can verify the backend and view the API documentation at http://127.0.0.1:5000/docs

 2: Start the Frontend Server (HTML)
Open a second terminal window in the project directory.

Start Python's HTTP server:

Bash
python -m http.server 3000
Open your web browser and go to:

Plaintext
http://localhost:3000
Enter input values into the forms and click the prediction buttons to receive real-time model outputs!
