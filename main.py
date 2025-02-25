import streamlit as st
import chatbotagent
import evaulationagent  # Fixed typo
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Set Streamlit Page Config
st.set_page_config(page_title="AI Job Assistant", layout="wide")

# Initialize Agents


chatbot = chatbotagent.ChatbotAgent(model="llama3")
evaluator = evaulationagent.EvaluationAgent()

# --- Initialize Session State ---
if "stage" not in st.session_state:
    st.session_state.stage = "onboarding"
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
if "cv_uploaded" not in st.session_state:
    st.session_state.cv_uploaded = False
    st.session_state.uploaded_file = None
if "cv_analysis_result" not in st.session_state:
    st.session_state.cv_analysis_result = None  # Stores analyzed CV report
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Stores chatbot messages
if "message_hist" not in st.session_state:
    st.session_state.message_hist = []

# --- Onboarding Section ---

def change_tocv_stage(name,location,job_interest,experience_level):
    if name.strip() and location.strip() and job_interest.strip():
            st.session_state.user_data = {
                "name": name,
                "location": location,
                "experience": experience_level,
                "job_interest": job_interest,
            }
            st.session_state.stage = "cv_upload"
    else:
            st.warning("Please fill in all the required fields!")

def cv_analysis():
    st.title("CV Analysis Report")
    st.session_state.cv_analysis_result = evaluator.run(st.session_state.uploaded_file)
    if st.session_state.cv_analysis_result:
            st.success("Here is your CV Analysis Report:")
            st.write(st.session_state.cv_analysis_result)
    else:
            st.warning("Your CV is still being analyzed. Please wait.")

if st.session_state.stage == "onboarding":
    st.title("Welcome to AI Job Assistant")
    # Collect user details
    name = st.text_input("Enter your name:", placeholder="John Doe")
    location = st.text_input("Enter your location:", placeholder="e.g., London, UK")
    experience_level = st.selectbox("Experience Level:", ["Entry Level", "Mid Level", "Senior Level"])
    job_interest = st.text_input("What is your job interest?", placeholder="e.g., Data Scientist")

    st.button("Save & Continue",on_click=change_tocv_stage,args=(name,location,job_interest,experience_level))

# --- CV Upload Section ---
if st.session_state.stage == "cv_upload":
    st.title("Upload Your CV (Optional)")
    uploaded_file = st.file_uploader("Upload your CV (PDF or DOCX)", type=["pdf", "docx"])

    col1, col2 = st.columns(2)
    with col1:
        if uploaded_file:
            st.session_state.cv_uploaded = True
            st.session_state.uploaded_file = uploaded_file
            st.success("CV Uploaded Successfully!")            
                
    with col2:
        if st.button("Skip"):
            st.session_state.stage = "main"

    if uploaded_file or st.session_state.cv_uploaded:
        if st.button("Continue"):
            st.session_state.stage = "main"

# --- Main Application Layout ---
if st.session_state.stage == "main":
    st.sidebar.title("Navigation")
    nav_option = st.sidebar.radio("Go to:", ["Chatbot", "CV Analysis Report"])

    st.sidebar.markdown("---")
    st.sidebar.subheader(f"User: {st.session_state.user_data.get('name', 'Unknown')}")
    st.sidebar.text(f"Location: {st.session_state.user_data.get('location', 'Unknown')}")
    st.sidebar.text(f"Interest: {st.session_state.user_data.get('job_interest', 'Unknown')}")

    if nav_option == "Chatbot":
        st.title("AI Job Assistant Chatbot")

        # Show Chat History
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Get User Input
        user_input = st.chat_input("Type your message...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.message_hist.append(HumanMessage(str(user_input)))

            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                response_container = st.empty()
                response_generator = chatbot.chat(st.session_state.message_hist,st.session_state.cv_analysis_result)

            # Stream output in real time
                streamed_response = st.write_stream(response_generator)

        # Store response in session state
            st.session_state.message_hist.append(AIMessage(streamed_response))
            st.session_state.chat_history.append({"role": "assistant", "content": streamed_response})

    if nav_option == "CV Analysis Report":
        st.button("Analyze?",on_click=cv_analysis)
        if not st.session_state.cv_uploaded:
            st.warning("You have not uploaded a CV. Please upload one in the sidebar.")
        

st.markdown("---")
st.caption("AI-powered Job Assistant")
