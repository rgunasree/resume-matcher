# 📄 AI-Powered Resume Matcher
An intelligent Streamlit application that ranks resumes against job descriptions using hybrid BM25 + semantic similarity matching.

## ✨ Features
- **Hybrid Matching** - Combines keyword and semantic analysis
- **Multi-Format Support** - Processes PDF/DOCX resumes
- **Visual Analytics** - Interactive charts and comparisons
- **Excel Export** - Download full analysis reports

## 🖼️ Application Screenshots

### 1. Home Screen
![Home Interface](screenshot/HOME.jpeg)

### 2. Upload Job Description & Resumes
![Upload Interface](screenshot/UPLOAD_JD_AND_RESUME.jpeg)

### 3. Matching the JD and Resumes 
![Loading](screenshot/MATCHING.jpeg)

### 4. Visual Analytics Dashboard
![Analytics View](screenshot/VISUAL_ANALYTICS.jpeg)

### 5. Excel Export & Summary
![Export Options](screenshot/EXCEL_AND_SUMMARY.jpeg)

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/resume-matcher.git 
cd resume-matcher

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run application
streamlit run app.py
```

## 📁 Project Structure
```
resume-matcher/
├── app.py                 # Main application code
├── requirements.txt       # Dependencies
├── screenshot/            # Application screenshots
│   ├── home.jpeg
│   ├── upload_jd_and_resume.jpeg
│   ├── visual_analytics.jpeg
│   └── excel_and_summary.jpeg
└── resume_analysis_results.xlsx  # Sample output
```

## 📝 Notes
- All screenshots are stored in the `/screenshot` directory
- Sample Excel output included for reference
- Replace placeholder GitHub URL with your actual repository

[![GitHub](https://img.shields.io/badge/View_on_GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourusername/resume-matcher)
```
