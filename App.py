import os
import streamlit as st
import pandas as pd
import joblib
from dotenv import load_dotenv
import google.generativeai as genai
import plotly.graph_objects as go

# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="AttriSenseAI",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# LOAD TRAINED MODEL
# ==================================================
@st.cache_resource
def load_model():
    return joblib.load("Final_Model.pkl")

model = load_model()

# ==================================================
# GEMINI CONFIG
# ==================================================
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
else:
    gemini_model = None

# ==================================================
# SESSION STATE
# ==================================================
if "ai_suggestion" not in st.session_state:
    st.session_state.ai_suggestion = None

# ==================================================
# SPEEDOMETER (GAUGE)
# ==================================================
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

# ==================================================
# HEADER
# ==================================================
st.markdown(
    """
    <h1 style="text-align:center;">AttriSenseAI</h1>
    <p style="text-align:center; font-size:18px;">
    Data-Driven Insights for Employee Retention
    </p>
    <p style="text-align:center; font-size:18px;">
    Predict. Prevent. Retain.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ==================================================
# INPUTS – EMPLOYEE PROFILE
# ==================================================
st.subheader("Employee Profile")

c1, c2, c3 = st.columns(3)

with c1:
    age = st.number_input("Age", 18, 60, 30, key="age")
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    marital_status = st.selectbox(
        "Marital Status", ["Single", "Married", "Divorced"], key="marital"
    )
    education = st.selectbox("Education Level", [1, 2, 3, 4, 5], key="education")

with c2:
    department = st.selectbox(
        "Department",
        ["Sales", "Research & Development", "Human Resources"],
        key="department"
    )
    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative",
            "Manager", "Sales Representative", "Research Director",
            "Human Resources"
        ],
        key="job_role"
    )
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], key="job_level")

with c3:
    education_field = st.selectbox(
        "Education Field",
        ["Life Sciences", "Medical", "Marketing",
         "Technical Degree", "Human Resources", "Other"],
        key="education_field"
    )
    monthly_income = st.number_input(
        "Monthly Income", min_value=1000, value=5000, key="monthly_income"
    )

# ==================================================
# COMPENSATION & WORK CONDITIONS
# ==================================================
st.subheader("Compensation & Work Conditions")

c4, c5, c6 = st.columns(3)

with c4:
    percent_salary_hike = st.slider(
        "Percent Salary Hike", 0, 30, 12, key="salary_hike"
    )
    stock_option_level = st.selectbox(
        "Stock Option Level", [0, 1, 2, 3], key="stock_option"
    )
    overtime = st.selectbox("OverTime", ["Yes", "No"], key="overtime")

with c5:
    business_travel = st.selectbox(
        "Business Travel",
        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
        key="business_travel"
    )
    total_working_years = st.slider(
        "Total Working Years", 0, 40, 8, key="total_working_years"
    )
    num_companies_worked = st.slider(
        "Companies Worked", 0, 10, 2, key="num_companies"
    )

with c6:
    years_at_company = st.slider(
        "Years at Company", 0, 40, 5, key="years_at_company"
    )
    years_in_current_role = st.slider(
        "Years in Current Role", 0, 20, 3, key="years_current_role"
    )
    years_with_manager = st.slider(
        "Years With Current Manager", 0, 20, 3, key="years_manager"
    )

# ==================================================
# SATISFACTION & GROWTH
# ==================================================
st.subheader("Satisfaction & Growth Factors")

c7, c8, c9 = st.columns(3)

with c7:
    job_satisfaction = st.selectbox(
        "Job Satisfaction", [1, 2, 3, 4], key="job_satisfaction"
    )
    environment_satisfaction = st.selectbox(
        "Environment Satisfaction", [1, 2, 3, 4], key="env_satisfaction"
    )

with c8:
    relationship_satisfaction = st.selectbox(
        "Relationship Satisfaction", [1, 2, 3, 4], key="rel_satisfaction"
    )
    work_life_balance = st.selectbox(
        "Work-Life Balance", [1, 2, 3, 4], key="work_life"
    )

with c9:
    job_involvement = st.selectbox(
        "Job Involvement", [1, 2, 3, 4], key="job_involvement"
    )
    years_since_last_promotion = st.slider(
        "Years Since Last Promotion", 0, 15, 2, key="years_promotion"
    )

# ==================================================
# PREDICTION
# ==================================================

if st.button("Predict Attrition", key="predict_btn", use_container_width=True):

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

    center = st.columns([1, 2, 1])[1]
    with center:
        st.plotly_chart(attrition_gauge(probability), use_container_width=True)

    # ==================================================
    # AI SUGGESTIONS
    # ==================================================
    st.markdown("<h3 style='text-align:center;'>🤖 AI-Driven Retention Suggestions</h3>", unsafe_allow_html=True)

    if gemini_model:
        prompt = f"""
        Attrition probability: {probability*100:.2f}%

        Job Satisfaction: {job_satisfaction}
        Work Life Balance: {work_life_balance}
        OverTime: {overtime}
        Monthly Income: {monthly_income}
        Years Since Last Promotion: {years_since_last_promotion}

        Provide practical and ethical steps to reduce attrition.
        """

        with st.spinner("Generating AI suggestions..."):
            try:
                response = gemini_model.generate_content(prompt)
                st.session_state.ai_suggestion = response.text
            except Exception:
                st.session_state.ai_suggestion = (
                    "AI suggestions are temporarily unavailable. "
                    "Consider improving work-life balance, compensation, "
                    "career growth opportunities, and employee recognition."
                )

    if st.session_state.ai_suggestion:
        st.info(st.session_state.ai_suggestion)

# ==================================================
# FOOTER
# ==================================================
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:14px;">
    Logistic Regression + Generative AI based Attrition Prevention System<br>
    MS Elevate & Edunet Foundation Internship Project
    </p>
    """,
    unsafe_allow_html=True
)
