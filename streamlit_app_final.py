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
        /* Progress bar colour */
        .stProgress > div > div > div > div {
            background-color: #4C9BE8;
        }
        /* Remove anchor link icon from headings */
        .stMarkdown h3 a, .stMarkdown h2 a, h3 a, h2 a {
            display: none !important;
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
            # User inputs years_since in the form instead of publication_year — we convert to publication_year before sending to model
            years_since = st.number_input("Years since publication", min_value=0, max_value=9, value=3, step=1)
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

    if not submitted:
        st.caption("Enter paper metadata and click Predict to see the impact probability.")

    if submitted:

        # Auto-calculate missingness flags from user inputs
        references_missing = 1 if referenced_works_count == 0 else 0
        countries_missing = 1 if countries_distinct_count == 0 else 0
        institutions_missing = 1 if unique_institutions_count == 0 else 0

        # Convert years_since to publication_year BEFORE building input_data
        publication_year = 2024 - years_since

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


        # Set colour based on probability score
        if prob_pct >= 55:
            color = '#2ca02c'  # green
        elif prob_pct >= 40:
            color = '#4C9BE8'  # blue
        else:
            color = '#f4a261'  # yellow/amber

        st.markdown(f"<h1 style='font-size: 5rem; color: {color}; text-align: center; padding-top: 1.5rem;'>{prob_pct}%</h1>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <style>
                .stProgress > div > div > div > div {{
                    background-color: {color};
                }}
            </style>
        """, unsafe_allow_html=True)
        st.progress(prob)

        st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
        # Interpretation message based on probability score
        if prob_pct >= 55:
            st.success("Strong signal — this paper is likely to reach the top 10% of citations.")
        elif prob_pct >= 40:
            st.info("Moderate signal — this paper may reach the top 10% of citations.")
        else:
            st.warning("Weak signal — this paper is unlikely to reach the top 10% of citations.")


# Adding SHAP analysis for the features

if submitted:

        X_train = pd.read_csv("data/features/X_train_trimmed.csv")

            # Create SHAP explainer using the training data and the model
        explainer = shap.TreeExplainer(model, X_train)

            # Calculate SHAP values for this prediction
        shap_values = explainer(input_data, check_additivity=False)

    
    # Clean labels for SHAP chart
        feature_labels = {
            'publication_year': 'Years since publication',
            'referenced_works_count': 'Number of References',
            'unique_authors_count': 'Number of Authors',
            'countries_distinct_count': 'Number of Countries',
            'unique_institutions_count': 'Number of Institutions',
            'funder_count': 'Number of Funders',
            'sdg_count': 'Number of SDGs',
            'sdg_4': 'SDG:<br>Quality Education',
            'references_missing': 'References:<br>Data Missing',
            'countries_missing': 'Countries:<br>Data Missing',
            'institutions_missing': 'Institutions:<br>Data Missing',
            'publication_type_journal-article': 'Journal Article',
            'publication_type_proceedings-article': 'Proceedings Article',
            'publication_type_thesis': 'Thesis',
            'publication_type_other': 'Other Publication Type',
            'publication_type_unknown': 'Unknown Publication Type',
            'oa_status_gold': 'Open Access:<br>Gold',
            'oa_status_diamond': 'Open Access:<br>Diamond',
            'oa_status_green': 'Open Access:<br>Green',
            'oa_status_bronze': 'Open Access:<br>Bronze',
            'oa_status_hybrid': 'Open Access:<br>Hybrid',
            'oa_status_closed': 'Open Access:<br>Closed',
            'topic_name_Anomaly Detection Techniques and Applications': 'Topic:<br>Anomaly Detection',
            'topic_name_Evolutionary Algorithms and Applications': 'Topic:<br>Evolutionary Algorithms',
            'topic_name_Metaheuristic Optimization Algorithms Research': 'Topic:<br>Metaheuristic Optimization',
            'topic_name_Natural Language Processing Techniques': 'Topic:<br>NLP Techniques',
            'topic_name_Neural Networks and Applications': 'Topic:<br>Neural Networks',
            'topic_name_Privacy-Preserving Technologies in Data': 'Topic:<br>Privacy Technologies',
            'topic_name_Quantum Computing Algorithms and Architecture': 'Topic:<br>Quantum Computing',
            'topic_name_Sentiment Analysis and Opinion Mining': 'Topic:<br>Sentiment Analysis',
            'topic_name_Speech Recognition and Synthesis': 'Topic:<br>Speech Recognition',
            'topic_name_Topic Modeling': 'Topic:<br>Topic Modeling',
            'language_en': 'Language:<br>English',
            'language_other': 'Language:<br>Other',
        }

        # Build SHAP DataFrame with all 34 features
        shap_df = pd.DataFrame({
            'feature': X_train.columns.tolist(),
            'shap_value': shap_values.values[0]
        })

        # Calculate abs and percentages from ALL 34 features first
        shap_df['abs_shap'] = shap_df['shap_value'].abs()

        # Save total before any filtering — used to calculate top 10 coverage
        total_abs_shap = shap_df['abs_shap'].sum()

        shap_df['shap_pct'] = (shap_df['abs_shap'] / total_abs_shap) * 100

        # Apply direction to percentage
        shap_df['shap_pct'] = shap_df.apply(
            lambda row: row['shap_pct'] if row['shap_value'] > 0 else -row['shap_pct'], axis=1
        )

        # Filter — keep numerical always, keep OHE only if selected
        numerical_cols = ['publication_year', 'referenced_works_count', 'unique_authors_count',
                          'countries_distinct_count', 'unique_institutions_count',
                          'funder_count', 'sdg_count', 'sdg_4']

        mask = (input_data.iloc[0].values != 0) | (shap_df['feature'].isin(numerical_cols))
        shap_df = shap_df[mask.values]

        # Filter top 10 by absolute value
        shap_df = shap_df.sort_values('abs_shap', ascending=False).head(10)
        shap_df = shap_df.sort_values('shap_pct', ascending=True)

        # Calculate what percentage of total influence top 10 covers
        top10_pct = round(shap_df['abs_shap'].sum() / total_abs_shap * 100, 1)

        # Reset index after all transformations
        shap_df = shap_df.reset_index(drop=True)

        # Apply clean labels
        shap_df['feature'] = shap_df['feature'].map(feature_labels)

        # Colour — green for positive, red for negative
        shap_df['color'] = shap_df['shap_value'].apply(lambda x: 'positive' if x > 0 else 'negative')

        shap_df['text_label'] = shap_df['shap_pct'].abs().round(1).astype(str) + '%'

        # Plot vertical bar chart
        fig = px.bar(
            shap_df,
            x='feature',
            y='shap_pct',
            orientation='v',
            color='color',
            color_discrete_map={'positive': '#2ca02c', 'negative': '#ef233c'},
            labels={'shap_pct': 'Impact Rate (%)', 'feature': ''},
            text='text_label'
        )
        fig.update_traces(
            textposition='outside',
            textfont=dict(color='black')
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Arial', size=13),
            xaxis=dict(showgrid=False, zeroline=True, zerolinecolor='#cccccc', zerolinewidth=1, showticklabels=True, title='', tickfont=dict(color='black')),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            margin=dict(l=20, r=20, t=40, b=100),
            bargap=0.3
        )
        # Display chart in bordered container
        with st.container(border=True):
            st.subheader("Top 10 Feature Contributions")
            st.markdown(f"Impact rate covers <span style='font-size: 1.2rem; color: #4C9BE8; font-weight: bold;'>{top10_pct}%</span> of total prediction influence.", unsafe_allow_html=True)
            st.plotly_chart(fig, width='stretch')