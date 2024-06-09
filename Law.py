import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import torch
from deep_translator import GoogleTranslator
import speech_recognition as sr
import datetime

# Load environment variables
load_dotenv()

# Load BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to find the most relevant row based on user input using BERT embeddings
def find_most_relevant_row(df, user_input):
    user_input_embedding = model.encode(user_input, convert_to_tensor=True)
    offense_embeddings = model.encode(df['Offense'].tolist(), convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(user_input_embedding, offense_embeddings)
    
    # Find the index of the highest similarity score
    best_match_idx = torch.argmax(cosine_scores).item()
    best_score = cosine_scores[0, best_match_idx].item()
    
    if best_score > 0:
        best_match_row = df.iloc[best_match_idx]
        return best_match_row
    else:
        return pd.Series()

# Path to the CSV file
csv_path = 'IPC_dataset.csv'
lawyers_csv_path = 'lawyers.csv'

# Initialize session state for page navigation and chat history
if 'page' not in st.session_state:
    st.session_state.page = "main"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Main CrimeScribe page
if st.session_state.page == "main":
    if not os.path.exists(csv_path):
        st.error("The CSV file path specified does not exist. Please check the path.")
    else:
        data = pd.read_csv(csv_path)
        
        # Streamlit app layout
        st.title("CrimeScribe")

        # Language selection mapping
        lang_mapping = {
            'en': 'English (English)',
            'hi': 'हिन्दी (Hindi)',
            'es': 'Español (Spanish)',
            'fr': 'Français (French)',
            'de': 'Deutsch (Dutch)',
            'zh-CN': '中文 (简体) (Chinese Simplified)',
            'ar': 'العربية (Arabic)',
            'te': 'తెలుగు (Telugu)' 
        }
        lang_options = list(lang_mapping.keys())
        lang_labels = list(lang_mapping.values())

        selected_lang = st.sidebar.selectbox("Select Language", lang_options, format_func=lambda x: lang_mapping[x])
        
        st.write("You can either type your query or use voice input.")
        
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

        if st.button("Use Voice Input"):
            recognized_text = recognize_speech_from_mic(recognizer, selected_lang)
            if recognized_text:
                # Translate the recognized text to English
                st.session_state.user_input = GoogleTranslator(source=selected_lang, target='en').translate(recognized_text)

        # Display the input area with the updated input
        user_input = st.text_area("Ask Your Legal Doubt:", st.session_state.user_input)

        if st.button("Ask"):
            if user_input:
                # Translate user input to English (if it's not already)
                translated_input = GoogleTranslator(source=selected_lang, target='en').translate(user_input)
                
                matching_row = find_most_relevant_row(data, translated_input)
                
                # Display the result
                if not matching_row.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        ipc_section = matching_row['IPC Section']
                        st.text_area("IPC Section", value=ipc_section, height=50)
                    with col2:
                        offense = matching_row['Offense']
                        translated_offense = GoogleTranslator(source='en', target=selected_lang).translate(offense)
                        st.text_area("Offense", value=translated_offense, height=100)
                    
                    with col1:
                        punishment = matching_row['Punishment']
                        translated_punishment = GoogleTranslator(source='en', target=selected_lang).translate(punishment)
                        st.text_area("Punishment", value=translated_punishment, height=100)
                    with col2:
                        cognizable = matching_row['Cognizable']
                        translated_cognizable = GoogleTranslator(source='en', target=selected_lang).translate(cognizable)
                        st.text_area("Cognizable", value=translated_cognizable, height=50)
                    
                    with col1:
                        bailable = matching_row['Bailable']
                        translated_bailable = GoogleTranslator(source='en', target=selected_lang).translate(bailable)
                        st.text_area("Bailable", value=translated_bailable, height=50)
                    with col2:
                        court = matching_row['Court']
                        translated_court = GoogleTranslator(source='en', target=selected_lang).translate(court)
                        st.text_area("Court", value=translated_court, height=50)
                    
                    # Update chat history
                    st.session_state.chat_history.append({
                        "query": user_input,
                        "response": {
                            "IPC Section": ipc_section,
                            "Offense": translated_offense,
                            "Punishment": translated_punishment,
                            "Cognizable": translated_cognizable,
                            "Bailable": translated_bailable,
                            "Court": translated_court
                        }
                    })
                else:
                    st.write("No matching row found.")
                    st.session_state.chat_history.append({
                        "query": user_input,
                        "response": "No matching row found."
                    })
            else:
                st.write("Please describe the situation to find the most relevant row.")

        # Display chat history in the sidebar
        st.sidebar.header("Chat History")
        for chat in st.session_state.chat_history:
            st.sidebar.write(f"**Query:** {chat['query']}")
            if isinstance(chat['response'], dict):
                for key, value in chat['response'].items():
                    st.sidebar.write(f"**{key}:** {value}")
            else:
                st.sidebar.write(f"**Response:** {chat['response']}")
            st.sidebar.write("---")

        if st.button("Find Lawyers"):
            st.session_state.page = "lawyers"

# Lawyers page
elif st.session_state.page == "lawyers":
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
            'te': 'తెలుగు'  # Added Telugu
        }
        lang_options = list(lang_mapping.keys())
        lang_labels = list(lang_mapping.values())

        selected_lang = st.sidebar.selectbox("Select Language", lang_options, format_func=lambda x: lang_mapping[x], key="select_language")
        
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

        if st.button("Use Voice Input"):
            recognized_text = recognize_speech_from_mic(recognizer, selected_lang)
            if recognized_text:
                # Translate the recognized text to English
                location_input = GoogleTranslator(source=selected_lang, target='en').translate(recognized_text)
                st.session_state.location_input = location_input

        # Display the input area with the updated input
        location_input = st.text_area("Enter Your Location:", st.session_state.location_input if 'location_input' in st.session_state else "")

        def find_lawyer_by_location(df, location):
            location = location.lower()
            matching_rows = df[df['Address'].str.lower().str.contains(location)]
            
            if not matching_rows.empty:
                return matching_rows.iloc[0]
            else:
                return pd.Series()

        if st.button("Find Lawyer"):
            if location_input:
                matching_lawyer = find_lawyer_by_location(lawyers_data, location_input)
                
                # Display the result
                if not matching_lawyer.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        name = matching_lawyer['Name']
                        st.text_area("Name", value=name, height=50)
                    with col2:
                        address = matching_lawyer['Address']
                        translated_address = GoogleTranslator(source='en', target=selected_lang).translate(address)
                        st.text_area("Address", value=translated_address, height=100)
                    
                    with col1:
                        phone = matching_lawyer['Phone No']
                        st.text_area("Phone", value=phone, height=50)
                    
                    # Appointment scheduling
                    st.markdown("### Schedule an Appointment")
                    date = st.date_input("Select Date", datetime.date.today())
                    time = st.time_input("Select Time", datetime.datetime.now().time())

                    if st.button("Schedule Appointment"):
                        st.write(f"Appointment scheduled with {name} on {date} at {time}.")
                        st.session_state.chat_history.append({
                            "query": location_input,
                            "response": f"Appointment scheduled with {name} on {date} at {time}."
                        })
                else:
                    st.write("No lawyer found at the specified location.")
            else:
                st.write("Please enter your location to find a lawyer.")

        if st.button("Go back to CrimeScribe"):
            st.session_state.page = "main"
