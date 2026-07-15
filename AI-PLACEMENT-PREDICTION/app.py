import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# Page configuration
st.set_page_config(
    page_title="AI Placement Prediction",
    page_icon="🎓",
    layout="wide"
)
# Custom CSS Styling
st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

h1 {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

h3 {
    color: #2c3e50;
}

.stButton > button {
    width: 100%;
    height: 45px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
}

.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 15px;
}

[data-testid="stSidebar"] {
    background-color: #f1f3f6;
}

.stAlert {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# Load model
import os
import pickle

model_path = os.path.join(os.path.dirname(__file__), "placement_model.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)
st.title("🎓 AI Based Placement Prediction System")

st.markdown(
    """
    ### 🚀 Machine Learning Based Student Placement Analysis
    This system predicts placement probability based on academic performance,
    technical skills, coding ability and experience.
    """
)

st.divider()

# Sidebar

st.sidebar.header("📌 Student Profile")

# Academic Details
st.sidebar.subheader("🎓 Academic Details")

cgpa = st.sidebar.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.5
)

branch = st.sidebar.selectbox(
    "Branch",
    ["CSE", "Civil", "ECE", "EEE", "IT"]
)

college_tier = st.sidebar.selectbox(
    "College Tier",
    [1, 2, 3]
)


# Technical Skills
st.sidebar.subheader("💻 Technical Skills")

python_skill = st.sidebar.slider(
    "Python Skill",
    0, 100, 50
)

dsa_skill = st.sidebar.slider(
    "DSA Skill",
    0, 100, 50
)

ml_skill = st.sidebar.slider(
    "Machine Learning Skill",
    0, 100, 50
)

web_dev_skill = st.sidebar.slider(
    "Web Development Skill",
    0, 100, 50
)


# Placement Skills
st.sidebar.subheader("🎯 Placement Skills")

coding_score = st.sidebar.slider(
    "Coding Score",
    0, 100, 50
)

communication_score = st.sidebar.slider(
    "Communication Score",
    0, 100, 50
)

aptitude_score = st.sidebar.slider(
    "Aptitude Score",
    0, 100, 50
)


# Experience
st.sidebar.subheader("🚀 Experience")

internships = st.sidebar.number_input(
    "Internships",
    0, 10, 0
)

projects = st.sidebar.number_input(
    "Projects",
    0, 10, 1
)

backlogs = st.sidebar.number_input(
    "Backlogs",
    0, 10, 0
)


# Prediction
if st.button("🔮 Predict Placement Probability"):

    input_data = pd.DataFrame({

        "cgpa":[cgpa],
        "college_tier":[college_tier],
        "python_skill":[python_skill],
        "dsa_skill":[dsa_skill],
        "ml_skill":[ml_skill],
        "web_dev_skill":[web_dev_skill],
        "coding_score":[coding_score],
        "communication_score":[communication_score],
        "aptitude_score":[aptitude_score],
        "internships":[internships],
        "projects":[projects],
        "backlogs":[backlogs],
        "branch":[branch]

    })


    # Convert categorical feature
    input_data = pd.get_dummies(input_data)


    # Match model features
    model_features = model.feature_names_in_

    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0


    input_data = input_data[model_features]


    probability = model.predict_proba(input_data)[0][1]

    percentage = probability * 100


    st.subheader("📊 Placement Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        
     st.markdown(
        f"""
        <div style="
            background-color:#ffffff;
            padding:20px;
            border-radius:15px;
            text-align:center;
            box-shadow:0px 4px 10px rgba(0,0,0,0.1);
        ">
        <h3>📊 Placement Probability</h3>
        <h1>{percentage:.2f}%</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    with col2:
     st.markdown(
        f"""
        <div style="
            background-color:#ffffff;
            padding:20px;
            border-radius:15px;
            text-align:center;
            box-shadow:0px 4px 10px rgba(0,0,0,0.1);
        ">
        <h3>🎓 CGPA</h3>
        <h1>{cgpa}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )


    with col3:
     st.markdown(
        f"""
        <div style="
            background-color:#ffffff;
            padding:20px;
            border-radius:15px;
            text-align:center;
            box-shadow:0px 4px 10px rgba(0,0,0,0.1);
        ">
        <h3>🚀 Projects</h3>
        <h1>{projects}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.progress(probability)
    if percentage >= 80:
        st.success("🟢 Excellent Placement Potential! Keep maintaining your skills 🚀")

    elif percentage >= 60:
        st.warning("🟡 Good Placement Potential. Improve skills to increase your chances 📈")

    else:
        st.error("🔴 Placement Probability is low. Focus on skill development 💪")


    # Skill Analysis
        # Skill Analysis
    st.subheader("📊 Student Skill Analysis")

    skill_data = {
        "🐍 Python": python_skill,
        "🧠 DSA": dsa_skill,
        "🤖 Machine Learning": ml_skill,
        "🌐 Web Development": web_dev_skill,
        "💻 Coding": coding_score,
        "🗣️ Communication": communication_score,
        "📚 Aptitude": aptitude_score
    }


    col1, col2, col3 = st.columns(3)

    for index, (skill, score) in enumerate(skill_data.items()):

        if index % 3 == 0:
            col = col1

        elif index % 3 == 1:
            col = col2

        else:
            col = col3


        with col:
            st.markdown(
                f"""
                <div style="
                    background-color:white;
                    padding:15px;
                    border-radius:15px;
                    text-align:center;
                    box-shadow:0px 3px 8px rgba(0,0,0,0.1);
                ">
                <h4>{skill}</h4>
                <h2>{score}%</h2>
                </div>
                """,
                unsafe_allow_html=True
            )


    # Improvement Suggestions
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("💡 Improvement Suggestions")

    suggestions = []


    if dsa_skill < 70:
        suggestions.append(
            f"""
🧠 DSA Improvement:
Your DSA score is {dsa_skill}%.
Practice Arrays, Strings, Linked Lists, Trees and Graph algorithms.
Target: Solve coding problems daily.
"""
        )


    if python_skill < 70:
        suggestions.append(
            f"""
🐍 Python Improvement:
Your Python score is {python_skill}%.
Improve Python OOP, Libraries and Data Structures knowledge.
"""
        )


    if coding_score < 70:
        suggestions.append(
            f"""
💻 Coding Skills:
Your coding score is {coding_score}%.
Practice competitive coding and improve problem solving ability.
"""
        )


    if communication_score < 70:
        suggestions.append(
            f"""
🗣️ Communication:
Your communication score is {communication_score}%.
Practice mock interviews and technical explanations.
"""
        )


    if aptitude_score < 70:
        suggestions.append(
            f"""
📚 Aptitude:
Your aptitude score is {aptitude_score}%.
Practice quantitative aptitude and logical reasoning questions.
"""
        )


    if projects < 2:
        suggestions.append(
            f"""
🚀 Projects:
You have completed {projects} project(s).
Build more real-world projects to strengthen your resume.
"""
        )


    if internships == 0:
        suggestions.append(
            """
🏢 Internship:
No internship experience found.
Try applying for internships to gain industry exposure.
"""
        )


    if backlogs > 0:
        suggestions.append(
            """
⚠️ Academic:
Clear pending backlogs to improve placement opportunities.
"""
        )


    if suggestions:
        for s in suggestions:
            st.warning(s)

    else:
        st.success("Excellent profile! Keep improving your skills 🚀")
