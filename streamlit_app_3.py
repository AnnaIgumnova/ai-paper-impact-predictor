import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
import plotly.express as px


# Remove default top padding
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title = "AI Paper Impact Predictor",
    page_icon = "📈",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open('models/gbc_tuned_streamlit.pkl', 'rb') as f:
        model = pkl.load(f)
    return model

model = load_model()

st.title("AI Paper Impact Predictor")
st.markdown("Predict whether your AI research paper will reach the top 10% of citations — using only metadata available at publication time.")

# Form wrapper — groups all inputs and the submit button together
# Nothing is sent to the model until the user clicks Predict
with st.form("prediction_form"):
    st.subheader("Enter Paper Metadata")

    # Four equal columns for compact layout
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1], gap="small")
    
    with col1:
        st.markdown("**Paper details**")
        topic_name = st.selectbox("Topic Name", options=["Anomaly Detection Techniques and Applications", "Evolutionary Algorithms and Applications", "Metaheuristic Optimization Algorithms Research", "Natural Language Processing Techniques", "Neural Networks and Applications", "Privacy-Preserving Technologies in Data", "Quantum Computing Algorithms and Architecture", "Sentiment Analysis and Opinion Mining", "Speech Recognition and Synthesis", "Topic Modeling"])
        publication_year = st.number_input("Publication Year", min_value=2000, max_value=2030, value=2023)
        language = st.selectbox("Language", options=["en", "other"])
        
    
    with col2:
        st.markdown("**Publication**")
        publication_type = st.selectbox("Publication Type", options=["journal-article", "proceedings-article", "thesis", "other", "unknown"])
        oa_status = st.selectbox("Open Access Status", options=["gold", "diamond", "green", "bronze", "hybrid", "closed"])
        st.markdown("<span style='font-size: 0.875rem; color: rgb(49, 51, 63);'>Sustainable Development Goals</span>", unsafe_allow_html=True)
        sdg_4 = st.checkbox("Quality Education")

    with col3:
        st.markdown("**Collaboration**")
        unique_authors_count = st.number_input("Authors", min_value=1, max_value=100, value=3, step=1)
        countries_distinct_count = st.number_input("Countries", min_value=0, max_value=50, value=1, step=1)
        unique_institutions_count = st.number_input("Institutions", min_value=0, max_value=100, value=1, step=1)


    with col4:
        st.markdown("**Impact signals**")
        referenced_works_count = st.number_input("References", min_value=0, max_value=1000, value=30, step=1)
        funder_count = st.number_input("Funders", min_value=0, max_value=50, value=0, step=1)
        sdg_count = st.number_input("SDGs tagged", min_value=0, max_value=17, value=0, step=1)
    
     # Submit button — triggers prediction when clicked
    submitted = st.form_submit_button("Predict Impact", type="primary", use_container_width=True)

if submitted:

    # Auto-calculate missingness flags from user inputs
    references_missing = 1 if referenced_works_count == 0 else 0
    countries_missing = 1 if countries_distinct_count == 0 else 0
    institutions_missing = 1 if unique_institutions_count == 0 else 0

    # Build a single-row DataFrame in the exact column order the model was trained on
    input_data = pd.DataFrame([{
        'publication_year': publication_year,
        'countries_distinct_count': countries_distinct_count,
        'referenced_works_count': referenced_works_count,
        'unique_authors_count': unique_authors_count,
        'unique_institutions_count': unique_institutions_count,
        'funder_count': funder_count,
        'sdg_count': sdg_count,
        'sdg_4': int(sdg_4),

        # One-hot encode publication type
        'publication_type_journal-article': 1 if publication_type == 'journal-article' else 0,
        'publication_type_other': 1 if publication_type == 'other' else 0,
        'publication_type_proceedings-article': 1 if publication_type == 'proceedings-article' else 0,
        'publication_type_thesis': 1 if publication_type == 'thesis' else 0,
        'publication_type_unknown': 1 if publication_type == 'unknown' else 0,

        # One-hot encode open access status
        'oa_status_bronze': 1 if oa_status == 'bronze' else 0,
        'oa_status_closed': 1 if oa_status == 'closed' else 0,
        'oa_status_diamond': 1 if oa_status == 'diamond' else 0,
        'oa_status_gold': 1 if oa_status == 'gold' else 0,
        'oa_status_green': 1 if oa_status == 'green' else 0,
        'oa_status_hybrid': 1 if oa_status == 'hybrid' else 0,

        # One-hot encode AI topic
        'topic_name_Anomaly Detection Techniques and Applications': 1 if topic_name == 'Anomaly Detection Techniques and Applications' else 0,
        'topic_name_Evolutionary Algorithms and Applications': 1 if topic_name == 'Evolutionary Algorithms and Applications' else 0,
        'topic_name_Metaheuristic Optimization Algorithms Research': 1 if topic_name == 'Metaheuristic Optimization Algorithms Research' else 0,
        'topic_name_Natural Language Processing Techniques': 1 if topic_name == 'Natural Language Processing Techniques' else 0,
        'topic_name_Neural Networks and Applications': 1 if topic_name == 'Neural Networks and Applications' else 0,
        'topic_name_Privacy-Preserving Technologies in Data': 1 if topic_name == 'Privacy-Preserving Technologies in Data' else 0,
        'topic_name_Quantum Computing Algorithms and Architecture': 1 if topic_name == 'Quantum Computing Algorithms and Architecture' else 0,
        'topic_name_Sentiment Analysis and Opinion Mining': 1 if topic_name == 'Sentiment Analysis and Opinion Mining' else 0,
        'topic_name_Speech Recognition and Synthesis': 1 if topic_name == 'Speech Recognition and Synthesis' else 0,
        'topic_name_Topic Modeling': 1 if topic_name == 'Topic Modeling' else 0,

        # One-hot encode language
        'language_en': 1 if language == 'en' else 0,
        'language_other': 1 if language == 'other' else 0,

        # Auto-calculated missingness flags
        'references_missing': references_missing,
        'countries_missing': countries_missing,
        'institutions_missing': institutions_missing,
    }])

    prob = model.predict_proba(input_data)[0][1]  # Probability of being in the top 10% most cited
    prob_pct = round(prob * 100,1)

    # Center the prediction score using empty flanking columns
    col_left, col_score, col_right = st.columns([1, 1, 1])

    with col_score:
        st.subheader("Prediction")
        st.metric(label="High-Impact Probability", value=f"{prob_pct}%")
        st.progress(prob)