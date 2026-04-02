import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
import plotly.express as px
import shap

st.set_page_config(
    page_title = "AI Paper Impact Predictor",
    page_icon = "📈",
    layout="wide"
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
        }
        .stFormSubmitButton > button {
            background-color: #1f77b4;
            color: white;
            border: none;
        }
        .stFormSubmitButton > button:hover {
            background-color: #1a6699;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with model metrics and interpretation tips
with st.sidebar:
    st.subheader("Model Performance")
    st.markdown("Trained on 293,000 AI research papers from OpenAlex.")
    
    st.info("This model is a **screening tool**. It helps prioritise which papers deserve closer attention — based on metadata available at publication time.")    
    
    st.divider()
    
    st.metric("Accuracy", "81.4%")
    st.caption("The model makes the correct prediction 81% of the time.")
    
    st.metric("AUC", "84.6%")
    st.caption("The model is 85% better at ranking high-impact papers than random chance.")
    
    st.metric("Recall", "63.8%")
    st.caption("The model correctly identifies 64% of all high-impact papers.")
    
    st.metric("Precision", "48.7%")
    st.caption("When the model predicts high impact, it is correct about half the time.")
    
    st.divider()
    st.caption("Model: Gradient Boosting Classifier | Dataset: OpenAlex API")

# Load the trained model once and cache it for future use

@st.cache_resource
def load_model():
    with open('models/gbc_tuned_streamlit.pkl', 'rb') as f:
        model = pkl.load(f)
    return model

model = load_model()

st.title("AI Paper Impact Predictor")
st.markdown("Predict whether your AI research paper will reach the top 10% of citations — using only metadata available at publication time.")

# Divide screen — inputs on left, prediction on right
col_form, col_result = st.columns([3, 1])
# Form wrapper — groups all inputs and the submit button together
# Nothing is sent to the model until the user clicks Predict
with col_form:
    with st.form("prediction_form"):
        st.subheader("Enter Paper Metadata")

        # Four equal columns for compact layout
        col1, col2, col3, col4 = st.columns([2, 1, 0.9, 0.9], gap="small")
        
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


with col_result:
    st.markdown("<div style='padding-top: 1rem; text-align: center;'>", unsafe_allow_html=True)
    st.subheader("**High-Impact Probability**")
    st.markdown("</div>", unsafe_allow_html=True)

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


        st.markdown(f"<h1 style='font-size: 5rem; color: #1f77b4; text-align: center; padding-top: 3rem;'>{prob_pct}%</h1>", unsafe_allow_html=True)
        st.progress(prob)

        # Interpretation message based on probability score
        if prob_pct >= 55:
            st.success("Strong signal — this paper is likely to reach the top 10% of citations.")
        elif prob_pct >= 40:
            st.warning("Moderate signal — this paper may reach the top 10% of citations.")
        else:
            st.info("Weak signal — this paper is unlikely to reach the top 10% of citations.")

# Adding SHAP analysis for the features

if submitted:
    st.subheader("Feature Contributions")

    X_train = pd.read_csv("data/features/X_train_trimmed.csv")

        # Create SHAP explainer using the training data and the model
    explainer = shap.TreeExplainer(model, X_train)

        # Calculate SHAP values for this prediction
    shap_values = explainer(input_data, check_additivity=False)

        # Clean labels for SHAP chart
# Clean labels for SHAP chart
    feature_labels = {
        'publication_year': 'Publication Year',
        'referenced_works_count': 'Number of References',
        'unique_authors_count': 'Number of Authors',
        'countries_distinct_count': 'Number of Countries',
        'unique_institutions_count': 'Number of Institutions',
        'funder_count': 'Number of Funders',
        'sdg_count': 'Number of SDGs',
        'sdg_4': 'SDG: Quality Education',
        'references_missing': 'References Data Missing',
        'countries_missing': 'Countries Data Missing',
        'institutions_missing': 'Institutions Data Missing',
        'publication_type_journal-article': 'Journal Article',
        'publication_type_proceedings-article': 'Proceedings Article',
        'publication_type_thesis': 'Thesis',
        'publication_type_other': 'Other Publication Type',
        'publication_type_unknown': 'Unknown Publication Type',
        'oa_status_gold': 'Open Access: Gold',
        'oa_status_diamond': 'Open Access: Diamond',
        'oa_status_green': 'Open Access: Green',
        'oa_status_bronze': 'Open Access: Bronze',
        'oa_status_hybrid': 'Open Access: Hybrid',
        'oa_status_closed': 'Open Access: Closed',
        'topic_name_Anomaly Detection Techniques and Applications': 'Topic: Anomaly Detection',
        'topic_name_Evolutionary Algorithms and Applications': 'Topic: Evolutionary Algorithms',
        'topic_name_Metaheuristic Optimization Algorithms Research': 'Topic: Metaheuristic Optimization',
        'topic_name_Natural Language Processing Techniques': 'Topic: NLP',
        'topic_name_Neural Networks and Applications': 'Topic: Neural Networks',
        'topic_name_Privacy-Preserving Technologies in Data': 'Topic: Privacy Technologies',
        'topic_name_Quantum Computing Algorithms and Architecture': 'Topic: Quantum Computing',
        'topic_name_Sentiment Analysis and Opinion Mining': 'Topic: Sentiment Analysis',
        'topic_name_Speech Recognition and Synthesis': 'Topic: Speech Recognition',
        'topic_name_Topic Modeling': 'Topic: Topic Modeling',
        'language_en': 'Language: English',
        'language_other': 'Language: Other',
    }

    # Build SHAP DataFrame
    shap_df = pd.DataFrame({
        'feature': X_train.columns.tolist(),
        'shap_value': shap_values.values[0]
    })

    # Sort by absolute SHAP value and keep top 10
    shap_df['abs_shap'] = shap_df['shap_value'].abs()
    shap_df = shap_df.sort_values('abs_shap', ascending=False).head(10)
    shap_df = shap_df.sort_values('shap_value', ascending=True)

    # Apply clean labels
    shap_df['feature'] = shap_df['feature'].map(feature_labels)

    # Colour — red for positive, blue for negative
    shap_df['color'] = shap_df['shap_value'].apply(lambda x: 'positive' if x > 0 else 'negative')

    # Plot vertical bar chart
    fig = px.bar(
        shap_df,
        x='feature',
        y='shap_value',
        orientation='v',
        color='color',
        color_discrete_map={'positive': '#2ec4b6', 'negative': '#ff6b6b'},
        title='Top 10 Feature Contributions',
        labels={'shap_value': 'Impact on Prediction', 'feature': ''}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)