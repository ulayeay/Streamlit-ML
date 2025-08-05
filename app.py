 import streamlit as st

st.set_page_config(
    page_title="Image Classification using Random Forest",
    page_icon="🔬",
    layout="centered"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        color: #FF6347;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .stSidebar > div:first-child {
        background-color: #1A3B66; 
        color: white; 
    }
    .stSidebar .stButton>button {
        background-color: #4CAF50; 
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 1.2em;
        width: 90%; 
        margin-bottom: 10px;
    }
    .stSidebar .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }

    .stSidebar .st-emotion-cache-1ddlmdq p { 
        color: white !important; 
        font-size: 1.1em;
        margin-bottom: 20px;
    }
    .team-info {
        background-color: #e6f7ff; 
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3399ff; 
        margin-top: 30px;
        box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
    }
    .team-info h3 {
        color: #3399ff;
        text-align: center;
        margin-bottom: 15px;
    }
    .team-info p {
        font-size: 1.1em;
        line-height: 1.6;
        color: #555555;
        text-align: center;
    }
    .st-emotion-cache-1cypcdb { 
        background-color: #d4edda !important;
        color: #155724 !important;
        border-color: #c3e6cb !important;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<p class="main-header">Malaria Cell Image Classification</p>', unsafe_allow_html=True)

st.sidebar.success("Select a page above")


st.markdown("""
<div class="team-info">
    <h3>Final Project Machine Learning Kelompok 8</h3>
    <p>
        <strong>Ulayya Azizna</strong> (1301220091) |
        <strong>Zahwa Ayska Fairana</strong> (1301223045) |
        <strong>M Daffa Raygama</strong> (1301223295)
    </p>
</div>
""", unsafe_allow_html=True)

st.info("💡 Use the sidebar on the left to navigate between the 'Dashboard' and 'Real-Time Detection' pages.")