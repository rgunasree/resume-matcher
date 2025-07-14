# 📄 Resume Matcher

🚀 A simple NLP project to rank resumes against a Job Description using TF-IDF + cosine similarity.

## ✅ Features
- Upload a JD (PDF/DOCX)
- Upload multiple resumes (PDF/DOCX)
- Get % match scores
- Helps shortlist top candidates

## 🏗️ Tech Stack
- Python 🐍
- Streamlit for UI
- scikit-learn for NLP
- PyPDF2 & python-docx for parsing

## 🚀 How to run
```bash
# Clone the repo
git clone https://github.com/yourusername/resume-matcher.git
cd resume-matcher

# Setup virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Start the app
streamlit run app.py

#📝 Sample data
You can prepare:

sample_jd.pdf : The job description

sample_resume1.pdf : Example candidate

sample_resume2.pdf : Another candidate

#Author
Made by GUNASREE R 🚀

## 🚀 Live Demo
[Click here to try it!] https://resume-matcher-kcasoqieg8geqtaezodnfw.streamlit.app/
