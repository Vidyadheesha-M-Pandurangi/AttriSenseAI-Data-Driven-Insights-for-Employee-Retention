import os
import streamlit as st
import pandas as pd
import joblib
from dotenv import load_dotenv
from google import genai
import plotly.graph_objects as go

# ENVIRONMENT

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="AttriSenseAI",
    page_icon="📊",
    layout="wide"
)

# LOAD TRAINED PIPELINE

@st.cache_resource
def load_model():
    return joblib.load("Final_Model.pkl")

model = load_model()

# GEMINI CLIENT

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# GAUGE

def attrition_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#374151"},
            "steps": [
                {"range": [0, 40], "color": "#22C55E"},
                {"range": [40, 70], "color": "#FACC15"},
                {"range": [70, 100], "color": "#EF4444"}
            ]
        }
    ))
    fig.update_layout(height=320)
    return fig

# HEADER

st.markdown(
    """
    <h1 style="text-align:center;">AttriSenseAI</h1>
    <p style="text-align:center; font-size:18px;">
        Data-Driven Insights for Employee Retention
    </p>
    <p style="text-align:center; font-style:italic; color:gray;">
        Predict. Prevent. Retain.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# EMPLOYEE PROFILE

st.subheader("Employee Profile")
c1, c2, c3 = st.columns(3)

with c1:
    age = st.number_input("Age", 18, 60, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    education_label = st.selectbox(
        "Education Level",
        [
            "1 - Below College",
            "2 - College",
            "3 - Bachelor",
            "4 - Master",
            "5 - Doctor"
        ]
    )
    education = int(education_label.split(" ")[0])

with c2:
    department = st.selectbox(
        "Department",
        ["Sales", "Research & Development", "Human Resources"]
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative",
            "Manager", "Sales Representative", "Research Director",
            "Human Resources"
        ]
    )

    job_level_label = st.selectbox(
        "Job Level",
        [
            "1 - Entry Level",
            "2 - Junior",
            "3 - Mid Level",
            "4 - Senior",
            "5 - Executive"
        ]
    )
    job_level = int(job_level_label.split(" ")[0])

with c3:
    education_field = st.selectbox(
        "Education Field",
        [
            "Life Sciences", "Medical", "Marketing",
            "Technical Degree", "Human Resources", "Other"
        ]
    )
    monthly_income = st.number_input("Monthly Income", min_value=1000, value=5000)

# COMPENSATION & WORK CONDITIONS

st.subheader("Compensation & Work Conditions")
c4, c5, c6 = st.columns(3)

with c4:
    percent_salary_hike = st.slider("Percent Salary Hike", 0, 30, 12)

    stock_label = st.selectbox(
        "Stock Option Level",
        [
            "0 - None",
            "1 - Low",
            "2 - Medium",
            "3 - High"
        ]
    )
    stock_option_level = int(stock_label.split(" ")[0])

    overtime = st.selectbox("OverTime", ["Yes", "No"])

with c5:
    business_travel = st.selectbox(
        "Business Travel",
        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    )

    total_working_years = st.slider("Total Working Years", 0, 40, 8)
    num_companies_worked = st.slider("Companies Worked", 0, 10, 2)

with c6:
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    years_in_current_role = st.slider("Years in Current Role", 0, 20, 3)
    years_with_manager = st.slider("Years With Current Manager", 0, 20, 3)

# SATISFACTION & GROWTH

st.subheader("Satisfaction & Growth Factors")
c7, c8, c9 = st.columns(3)

with c7:
    job_satisfaction = int(st.selectbox(
        "Job Satisfaction",
        ["1 - Low", "2 - Medium", "3 - High", "4 - Very High"]
    ).split(" ")[0])

    environment_satisfaction = int(st.selectbox(
        "Environment Satisfaction",
        ["1 - Low", "2 - Medium", "3 - High", "4 - Very High"]
    ).split(" ")[0])

with c8:
    relationship_satisfaction = int(st.selectbox(
        "Relationship Satisfaction",
        ["1 - Low", "2 - Medium", "3 - High", "4 - Very High"]
    ).split(" ")[0])

    work_life_balance = int(st.selectbox(
        "Work-Life Balance",
        ["1 - Bad", "2 - Good", "3 - Better", "4 - Best"]
    ).split(" ")[0])

with c9:
    job_involvement = int(st.selectbox(
        "Job Involvement",
        ["1 - Low", "2 - Medium", "3 - High", "4 - Very High"]
    ).split(" ")[0])

    years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 2)

# PREDICTION

if st.button("Predict Attrition", use_container_width=True):

    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "MaritalStatus": [marital_status],
        "Department": [department],
        "JobRole": [job_role],
        "Education": [education],
        "EducationField": [education_field],
        "JobLevel": [job_level],
        "MonthlyIncome": [monthly_income],
        "PercentSalaryHike": [percent_salary_hike],
        "StockOptionLevel": [stock_option_level],
        "OverTime": [overtime],
        "BusinessTravel": [business_travel],
        "JobSatisfaction": [job_satisfaction],
        "EnvironmentSatisfaction": [environment_satisfaction],
        "RelationshipSatisfaction": [relationship_satisfaction],
        "WorkLifeBalance": [work_life_balance],
        "JobInvolvement": [job_involvement],
        "TotalWorkingYears": [total_working_years],
        "YearsAtCompany": [years_at_company],
        "YearsInCurrentRole": [years_in_current_role],
        "YearsWithCurrManager": [years_with_manager],
        "YearsSinceLastPromotion": [years_since_last_promotion],
        "NumCompaniesWorked": [num_companies_worked]
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown(
        f"<h3 style='text-align:center;'>"
        f"{'⚠️ High Risk of Employee Attrition' if prediction == 1 else '✅ Low Risk of Employee Attrition'}"
        f"</h3>",
        unsafe_allow_html=True
    )

    st.markdown("<h3 style='text-align:center;'>📊 Attrition Risk Gauge</h3>", unsafe_allow_html=True)
    st.plotly_chart(attrition_gauge(probability), width="stretch")
    st.markdown(
        f"""
        <p style='text-align:center; font-size:16px;'>
        The predicted probability of attrition is <strong>{probability*100:.2f}%</strong>.
        </p>
        """,
        unsafe_allow_html=True
    )
    # AI SUGGESTIONS

    st.markdown(
        "<h3 style='text-align:center;'>🤖 AI-Driven Retention Suggestions</h3>",
        unsafe_allow_html=True
    )
    ai_placeholder = st.empty()
    if client:
        try:
            with ai_placeholder.container():
                st.markdown(
                    "<p style='text-align:center; font-size:16px;'>"
                    "⏳ Suggesting future retention steps…"
                    "</p>",
                    unsafe_allow_html=True
                )

            with st.spinner(""):
                response = client.models.generate_content(
                    model="models/gemini-2.5-flash",
                    contents=f"""
                    An employee has an attrition probability of {probability*100:.2f}%.

                    Suggest ethical, practical, and HR-focused actions
                    to reduce employee attrition.
                    """
                )
            st.info(response.text)

        except Exception as e:
            st.warning(
                """
                ⚠️ **AI Suggestions Unavailable**

                The AI service could not be reached.
                Possible reasons include quota limits,
                temporary service outage, or network restrictions.
                """
            )
            st.code(str(e))
    else:
        st.error("❌ Gemini API key not configured.")

# FOOTER

st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:14px;">
    Logistic Regression + Generative AI based Attrition Prevention System
    </p>
    """,
    unsafe_allow_html=True
)
