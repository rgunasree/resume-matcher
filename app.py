import streamlit as st
from extract_text import extract_pdf_text, extract_docx_text
from matcher import clean_text, compute_similarity
import matplotlib.pyplot as plt
import pandas as pd
import io


st.title("📄 Resume Matcher")

# ✅ TEXTAREA for JD (no file upload)
jd_text = st.text_area("Paste the Job Description here", height=200)

# ✅ UPLOAD resumes
resumes = st.file_uploader("Upload candidate resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("Match"):
    if jd_text.strip() and resumes:
        # clean JD
        jd_text_clean = clean_text(jd_text)

        results = []
        for resume in resumes:
            # extract & clean resume
            resume_text = extract_pdf_text(resume) if resume.name.endswith('.pdf') else extract_docx_text(resume)
            resume_text = clean_text(resume_text)

            # compute similarity
            score = compute_similarity(resume_text, jd_text_clean)
            results.append((resume.name, round(score * 100, 2)))

        # sort by highest score
        results.sort(key=lambda x: x[1], reverse=True)

        st.subheader("Match Results")
        for name, score in results:
            st.write(f"✅ **{name}** : {score}% match")

        # ✅ CHART
        st.subheader("Match Scores Chart")
        names = [x[0] for x in results]
        scores = [x[1] for x in results]

        colors = ['green' if score > 80 else 'yellow' if score > 50 else 'red' for score in scores]

        fig, ax = plt.subplots()
        ax.barh(names, scores, color=colors)
        ax.set_xlabel("Match %")
        ax.set_title("Resume Match Scores")
        ax.invert_yaxis()

        ax.set_xlabel("Match %")
        ax.set_title("Resume Match Scores")
        ax.invert_yaxis()
        st.pyplot(fig)

                # ✅ EXCEL EXPORT
        st.subheader("Download Results as Excel")

        df = pd.DataFrame(results, columns=["Resume Name", "Match %"])

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="Match Scores")

        st.download_button(
            label="📥 Download Excel",
            data=buffer.getvalue(),
            file_name="resume_match_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:
        st.warning("Please paste the Job Description and upload resumes.")
