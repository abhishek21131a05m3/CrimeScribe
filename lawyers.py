import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
import speech_recognition as sr
import datetime

# Load environment variables
load_dotenv()

# Load BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to find the most relevant lawyer based on location
def find_lawyer_by_location(df, location):
    location = location.lower()
    matching_rows = df[df['Address'].str.lower().str.contains(location)]
    
    if not matching_rows.empty:
        return matching_rows.iloc[0]
    else:
        return pd.Series()

# Path to the CSV file
lawyers_csv_path = 'lawyers.csv'

if not os.path.exists(lawyers_csv_path):
    st.error("The lawyers CSV file path specified does not exist. Please check the path.")
else:
    lawyers_data = pd.read_csv(lawyers_csv_path)

    # Streamlit app layout
    st.title("Lawyer Finder")

    # Language selection mapping
    lang_mapping = {
        'en': 'English',
        'hi': 'हिन्दी',
        'es': 'Español',
        'fr': 'Français',
        'de': 'Deutsch',
        'zh-CN': '中文 (简体)',
        'ar': 'العربية',
        'te': 'తెలుగు' 
    }
    lang_options = list(lang_mapping.keys())
    lang_labels = list(lang_mapping.values())

    selected_lang = st.sidebar.selectbox("Select Language", lang_options, format_func=lambda x: lang_mapping[x])

    st.write("You can either type your location or use voice input.")
    
    recognizer = sr.Recognizer()

    def recognize_speech_from_mic(recognizer, lang):
        with sr.Microphone() as source:
            st.write("Listening...")
            audio = recognizer.listen(source)
            st.write("Processing...")
            try:
                recognized_text = recognizer.recognize_google(audio, language=lang)
                st.write(f"Recognized Text: {recognized_text}")
                return recognized_text
            except sr.UnknownValueError:
                st.write("Google Speech Recognition could not understand the audio.")
                return ""
            except sr.RequestError as e:
                st.write(f"Could not request results from Google Speech Recognition service; {e}")
                return ""

    # Use session state to store the input to retain it between interactions
    if 'location_input' not in st.session_state:
        st.session_state.location_input = ""

    if st.button("Use Voice Input"):
        recognized_text = recognize_speech_from_mic(recognizer, selected_lang)
        if recognized_text:
            # Translate the recognized text to English
            st.session_state.location_input = GoogleTranslator(source=selected_lang, target='en').translate(recognized_text)

    # Display the location input area with the updated input
    location_input = st.text_area("Enter Your Location:", st.session_state.location_input)

    if st.button("Find Lawyer"):
        if location_input:
            # Translate location input to English (if it's not already)
            translated_location = GoogleTranslator(source=selected_lang, target='en').translate(location_input)
            
            matching_lawyer_row = find_lawyer_by_location(lawyers_data, translated_location)
            
            # Display the lawyer details
            if not matching_lawyer_row.empty:
                st.write("### Lawyer Details")
                lawyer_name = matching_lawyer_row['Name']
                lawyer_address = matching_lawyer_row['Address']
                lawyer_contact = matching_lawyer_row['Phone No']
                
                st.text_area("Name", value=lawyer_name, height=50)
                st.text_area("Address", value=lawyer_address, height=100)
                st.text_area("Contact", value=lawyer_contact, height=50)
                
                # Appointment scheduling
                st.markdown("### Schedule an Appointment")
                date = st.date_input("Select Date", datetime.date.today())
                time = st.time_input("Select Time", datetime.datetime.now().time())
                
                # Ensure the selected time is within business hours (9 AM to 6 PM)
                if time < datetime.time(9, 0) or time > datetime.time(18, 0):
                    st.warning("Please select a time between 9 AM and 6 PM.")
                else:
                    if st.button("Schedule Appointment"):
                        st.success(f"Appointment scheduled with {lawyer_name} on {date} at {time}.")
                        st.session_state.chat_history.append({
                            "query": location_input,
                            "response": f"Appointment scheduled with {lawyer_name} on {date} at {time}."
                        })
            else:
                st.write("No matching lawyer found.")
        else:
            st.write("Please provide a location to find a relevant lawyer.")

    # Display chat history in the sidebar
    st.sidebar.header("Chat History")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        st.sidebar.write(f"**Query:** {chat['query']}")
        st.sidebar.write(f"**Response:** {chat['response']}")
        st.sidebar.write("---")
