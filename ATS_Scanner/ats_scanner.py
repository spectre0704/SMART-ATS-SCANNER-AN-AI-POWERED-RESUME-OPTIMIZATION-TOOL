from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_match(resume_text, job_description):
    # Create a list containing both texts
    text_list = [resume_text, job_description]
    
    # Initialize the Vectorizer
    # This converts text into a matrix of numbers (TF-IDF)
    cv = TfidfVectorizer()
    
    # Fit and transform the text
    count_matrix = cv.fit_transform(text_list)
    
    # Calculate Cosine Similarity
    # This checks the angle between the two text vectors. 
    # 0 = No match, 1 = Perfect match.
    match_percentage = cosine_similarity(count_matrix)[0][1] * 100
    
    return round(match_percentage, 2)

# --- Main Execution ---
if __name__ == "__main__":
    print("--- AI SMART RESUME SCREENER ---")
    
    # 1. Paste a Job Description here
    job_desc = """
    We are looking for a Python Developer with experience in AI and Machine Learning.
    Must know Scikit-Learn, Pandas, and how to build models. 
    Good communication skills and problem-solving ability required.
    """

    # 2. Paste a Resume text here
    # (Try changing this text to see how the score changes!)
    my_resume = """
    I am a final year BCA student specializing in AI and ML.
    I have skills in Python, Scikit-Learn, and Data Science.
    I have built projects using Pandas and Machine Learning models.
    I am a good problem solver.
    """

    print("\nAnalyzing match...")
    match_score = calculate_match(my_resume, job_desc)
    
    print(f"-----------------------------")
    print(f"Match Score: {match_score}%")
    print(f"-----------------------------")
    
    # Feedback logic
    if match_score > 60:
        print("Result: Resume Selected! ✅")
    else:
        print("Result: Resume Rejected. Keywords missing. ❌")
        
    print("\n(Tip: Add more keywords from the Job Description to the Resume to increase score!)")