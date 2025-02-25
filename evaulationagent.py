import baseagent
import pdfplumber
import io
import streamlit as st

instructions = """
You are an advanced AI-powered CV evaluation agent. Your role is to analyze and assess the content of the CV provided.

### **Evaluation Criteria**
1. **Profile Summary** – Extract and summarize key details about the candidate (name, profession, years of experience, industry).
2. **Skills Assessment** – Identify and categorize technical, soft, and domain-specific skills.
3. **Experience Evaluation** – Analyze past job roles, responsibilities, and achievements.
4. **Education & Certifications** – Evaluate educational background and additional certifications.
5. **Strengths & Weaknesses** – Highlight strong areas and suggest areas for improvement.
6. **Job Suitability** – Provide a recommendation on which job roles best match the candidate's profile.
7. **CV Formatting & Readability** – Assess the overall clarity, structure, and presentation.

### **Scoring System (Out of 100)**
- **90-100 (Excellent)** – Well-structured CV with strong experience, relevant skills, and professional formatting.
- **75-89 (Good)** – A strong CV with minor improvements needed.
- **50-74 (Average)** – Some gaps in skills, experience, or structure that need improvement.
- **Below 50 (Needs Improvement)** – Significant gaps in content, structure, or relevance.

### **Final Output**
1. **Summary of the CV**
2. **Strengths & Weaknesses**
3. **Job Suitability Recommendations**
4. **CV Score (Out of 100)**
5. **Actionable Improvement Tips**
"""

name = "llama3"

class EvaluationAgent(baseagent.BaseAgent):
    def __init__(self):
        super().__init__(name=name, instructions=instructions)

    def run(self, uploaded_file):
        print("Running CV Evaluation...")
        
        try:
            # Read the uploaded PDF file as bytes
            doc = io.BytesIO(uploaded_file.getvalue())
            text = ""
            
            # Use pdfplumber to extract text
            with pdfplumber.open(doc) as pdf:
                for page in pdf.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text + "\n"
            
            if not text.strip():
                return "⚠️ No readable text found in the PDF. It might be a scanned document."
            
            # Query AI model for CV evaluation
            ans = self._query_ollama(text, temperature=0.6, max_tokens=1100)
            #print(ans)
            return ans

        except Exception as e:
            return f"Error processing CV: {str(e)}"