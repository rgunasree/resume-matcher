# 📄 AI-Powered Resume Matcher

🚀 **A smart Streamlit application to rank resumes against job descriptions using hybrid BM25 + semantic similarity (FAISS + Sentence Transformers).**

✅ Features:
- Upload a Job Description (JD) as plain text
- Upload multiple resumes (PDF or DOCX)
- Get an AI-powered % match score
- See detailed breakdown: BM25 relevance vs Semantic similarity
- Visual charts & Excel download for easy reporting

🛠️ **Tech Stack:**
- Python 🐍
- Streamlit for UI
- Rank-BM25 for lexical scoring
- LangChain + FAISS + HuggingFace embeddings for semantic matching
- Matplotlib & Pandas for analysis
- PyPDF2 & python-docx for parsing resumes

---

## 🚀 How to run

```bash
# Clone this repo
git clone https://github.com/rgunasree/resume-matcher.git
cd resume-matcher

# Setup virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

---

##Screenshot of OUTPUT

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/350b9f2b-1b90-474a-bea6-51599eca6784" />

#After you apload all the resumes 
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/a7c14d27-d503-4bd5-8ed4-dc66706a2828" />

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/f128ec9a-d384-423e-80e1-1d893e378bdd" />

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/ff43da71-98f0-467a-bcf4-fc1f99c6efbd" />

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/e8f7ea98-8400-493c-bee4-be6a022be61a" />

---
## 🏆 Why this project?
- Helps hiring managers and data teams:

- Automate resume screening

- Visualize how close a candidate is to the JD

- Shortlist top matches with data-backed confidence

- Built as a portfolio project to showcase NLP, RAG & visualization skills.

## 👤 Author
🔗 [LinkedIn](https://www.linkedin.com/in/gunasree-r-55024224a)

💻 [GitHub](https://github.com/rgunasree)

##⭐ Give it a star if you like it!
