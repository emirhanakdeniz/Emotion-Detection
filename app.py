import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table, IST  # Import IST from track_utils

# Load Model
pipe_lr = joblib.load(open("./models/emotion_detection_pipeline.pkl", "rb"))

# Function
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}

# Main Application
def main():
    st.title("Emotion Detection App")
    create_page_visited_table()
    create_emotionclf_table()
    add_page_visited_details("Home", datetime.now(IST))

    st.subheader("Emotion Detection in Text")

    with st.form(key='emotion_clf_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(IST))

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
