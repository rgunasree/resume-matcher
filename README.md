# 📄 AI-Powered Resume Matcher

![App Banner](https://via.placeholder.com/1200x400?text=Resume+Matcher+Screenshot)

A Streamlit application that automatically ranks resumes against job descriptions using hybrid BM25 + semantic similarity matching.

## ✨ Features
- **Hybrid Scoring** - Combines BM25 (keyword) and FAISS (semantic) matching
- **Multi-Format Support** - Works with PDF and DOCX resumes
- **Visual Analytics** - Interactive charts and score breakdowns
- **Excel Export** - Download full results for reporting
- **Fast Processing** - Optimized pipeline for quick analysis

## 🚀 Quick Start

1. Clone the repo:
```bash
git clone https://github.com/rgunasree/resume-matcher.git
cd resume-matcher
```

2. Set up environment:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. Run the app:
```bash
streamlit run app.py
```

## 🖥️ UI Screenshots

 ![Upload](![Uploading Screenshot_15-7-2025_14417_localhost.jpeg…]()
) 
 ![Results](https://via.placeholder.com/600x400?text=Results+View) 
 ![Analysis](https://via.placeholder.com/600x400?text=Score+Breakdown) 
 ![Export](https://via.placeholder.com/600x400?text=Excel+Export) 

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **NLP**: spaCy, NLTK
- **Matching**: BM25 + FAISS
- **Embeddings**: Sentence Transformers
- **Parsing**: PyPDF2, python-docx
- **Analysis**: Pandas, Matplotlib

## 📊 How It Works
1. Upload job description and resumes
2. System extracts and weights key sections
3. Calculates both lexical and semantic matches
4. Combines scores for final ranking
5. Presents visual breakdown of results

## 📝 Notes
- Minimum JD length: 20 characters
- Supported formats: PDF, DOCX
- Max file size: 10MB per resume

## 👤 Author
**Gunasree R**  
[LinkedIn](https://www.linkedin.com/in/gunasree-r-55024224a) 

[GitHub](https://github.com/rgunasree)

```
