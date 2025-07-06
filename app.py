import streamlit as st
from extract_text import extract_pdf_text, extract_docx_text
from matcher import clean_text, compute_similarity

st.title("Resume Matcher")

jd_file = st.file_uploader("Upload Job Description", type=['pdf', 'docx'])
resumes = st.file_uploader("Upload Resumes", type=['pdf', 'docx'], accept_multiple_files=True)

if st.button("Match"):
    if jd_file and resumes:
        jd_text = extract_pdf_text(jd_file) if jd_file.name.endswith('.pdf') else extract_docx_text(jd_file)
        jd_text = clean_text(jd_text)

        results = []
        for resume in resumes:
            resume_text = extract_pdf_text(resume) if resume.name.endswith('.pdf') else extract_docx_text(resume)
            resume_text = clean_text(resume_text)
            score = compute_similarity(resume_text, jd_text)
            results.append((resume.name, round(score * 100, 2)))

        results.sort(key=lambda x: x[1], reverse=True)
        st.subheader("Match Results")
        for name, score in results:
            st.write(f"{name}: {score}% match")
    else:
        st.warning("Please upload both JD and resumes.")
