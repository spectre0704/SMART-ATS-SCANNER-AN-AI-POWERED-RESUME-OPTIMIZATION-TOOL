import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Backend Logic (The AI) ---
def extract_text_from_pdf(file):
    """Extracts text from an uploaded PDF file."""
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def calculate_match(job_desc, resume_text):
    """Calculates the percentage match between JD and Resume."""
    # Handle empty inputs
    if not job_desc or not resume_text:
        return 0.0

    text_list = [resume_text, job_desc]
    cv = TfidfVectorizer()
    count_matrix = cv.fit_transform(text_list)
    match_percentage = cosine_similarity(count_matrix)[0][1] * 100
    return round(match_percentage, 2)

# --- 2. Frontend UI (The Localhost Web App) ---
st.set_page_config(page_title="Smart ATS", page_icon="🤖")

st.title("🤖 Smart ATS Scanner")
st.markdown("### Optimize your resume for the Application Tracking System")
st.markdown("---")

# Layout: Two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.header("1. Job Description")
    job_description = st.text_area("Paste the Job Description here:", height=300)

with col2:
    st.header("2. Upload Resume")
    uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

# --- 3. Execution ---
if st.button("Analyze Resume"):
    if uploaded_file is not None and job_description:
        with st.spinner("Analyzing content..."):
            # Extract text from the PDF
            resume_text = extract_text_from_pdf(uploaded_file)
            
            # AI Calculation
            match_score = calculate_match(job_description, resume_text)
            
            # --- Result Display ---
            st.markdown("---")
            st.subheader("Analysis Result")
            
            # Create a nice metric and progress bar
            my_bar = st.progress(0, text="Matching...")
            
            # Animate the bar (visual effect)
            for percent_complete in range(int(match_score)):
                my_bar.progress(percent_complete + 1, text=f"Match Score: {percent_complete + 1}%")
            
            col_metric1, col_metric2 = st.columns(2)
            
            with col_metric1:
                st.metric(label="Match Percentage", value=f"{match_score}%")
            
            with col_metric2:
                if match_score >= 60:
                    st.success("Status: **Selected** ✅")
                else:
                    st.error("Status: **Rejected** ❌")
                    st.write("Tip: Add more keywords from the JD.")
                    
    else:
        st.warning("Please upload a resume and paste a job description first.")