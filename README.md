# Emotion Detection App

A web application for detecting emotions in text using a machine learning model trained on a labeled dataset of emotional expressions. Built with Streamlit, this app allows users to input text and receive real-time emotion predictions, along with prediction probabilities and visualizations.

## Features
- **Emotion Classification**: Predicts emotions such as joy, sadness, anger, fear, surprise, disgust, shame, and neutral from user-input text.
- **Probability Visualization**: Displays prediction confidence and probability distribution for all emotion classes.
- **User Interaction Tracking**: Logs page visits and prediction details in a local SQLite database.
- **Interactive UI**: Built with Streamlit for a fast, interactive web experience.



## Project Structure
```
Emotion-Detection/
├── app.py                      # Main Streamlit app
├── track_utils.py              # Utilities for tracking user activity and predictions
├── models/
│   └── emotion_detection_pipeline.pkl  # Trained ML pipeline
├── data/
│   ├── emotion_dataset_raw.csv # Raw emotion dataset
│   ├── emotion_dataset.csv     # Cleaned/processed dataset
│   └── data.db                 # SQLite database for logs
├── notebooks/
│   └── Emotion-Detection.ipynb # Jupyter notebook for model training
├── examples.txt                # Example texts for testing
└── README.md                   # Project documentation
```

## How It Works
1. **User Input**: Enter any text in the app's text area.
2. **Prediction**: The app uses a pre-trained machine learning pipeline to predict the emotion.
3. **Visualization**: Results are shown with emoji, confidence score, and a probability bar chart.
4. **Tracking**: All predictions and page visits are logged for analysis.

## Model Training
- **Dataset**: The model is trained on a labeled dataset (`emotion_dataset_raw.csv`) containing text samples and their corresponding emotions.
- **Preprocessing**: Text is cleaned (removal of user handles, stopwords, etc.) using `neattext`.
- **Model**: A scikit-learn pipeline with `CountVectorizer` and `LogisticRegression` is used.
- **Training**: See `notebooks/Emotion-Detection.ipynb` for full training and evaluation code.
- **Export**: The trained pipeline is saved as `models/emotion_detection_pipeline.pkl`.

## Dependencies
Install the following Python packages (create a `requirements.txt` as needed):
- streamlit
- altair
- pandas
- numpy
- scikit-learn
- joblib
- neattext
- pytz

You can install them with:
```bash
pip install streamlit altair pandas numpy scikit-learn joblib neattext pytz
```


## Logging & Tracking
- All page visits and predictions are logged in `data/data.db` (SQLite).
- See `track_utils.py` for database schema and logging functions.

## Customization
- **Model**: Retrain or replace the model by editing the notebook and exporting a new `.pkl` file.
- **Dataset**: Add or modify data in `data/emotion_dataset_raw.csv`.
- **UI**: Customize the Streamlit interface in `app.py`.


## Acknowledgements
- Inspired by open-source emotion detection datasets and Streamlit community examples.
- Built with Python, scikit-learn, and Streamlit.