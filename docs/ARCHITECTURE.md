# 🏗️ Resume Matcher Architecture

## UML Class Diagram

```mermaid
classDiagram
    class ResumeMatcher {
        +SKILL_KEYWORDS: list
        +MAX_FILE_SIZE: int
        +MIN_CONTENT_LENGTH: int
        +MIN_JD_LENGTH: int
        +load_dependencies() nlp
        +clean_text(text) str
        +extract_skills(text) list
        +extract_experience(text) int
        +parse_sections(text) dict
        +extract_pdf_text(file) str
        +extract_docx_text(file) str
        +create_weighted_content(text, sections) str
        +validate_file_size(file) tuple
        +process_resume(file) tuple
        +build_bm25_index(resume_data) BM25Okapi
        +build_faiss_index(resume_data) FAISS
        +calculate_scores(resume_data, bm25_index, vector_store, jd) list
        +display_results(results)
        +create_visualization(results)
        +create_excel_download(results)
        +main()
    }

    class DocumentProcessor {
        <<utility>>
        +extract_pdf_text(file) str
        +extract_docx_text(file) str
    }

    class TextAnalyzer {
        <<utility>>
        +clean_text(text) str
        +extract_skills(text) list
        +extract_experience(text) int
        +parse_sections(text) dict
    }

    class MatchingEngine {
        <<service>>
        +build_bm25_index(resume_data) BM25Okapi
        +build_faiss_index(resume_data) FAISS
        +calculate_scores(resume_data, bm25_index, vector_store, jd) list
    }

    class ResultVisualizer {
        <<service>>
        +display_results(results)
        +create_visualization(results)
        +create_excel_download(results)
    }

    class ResumeData {
        +name: str
        +original_text: str
        +weighted_content: str
        +skills: list
        +experience: int
        +sections: dict
    }

    ResumeMatcher --> DocumentProcessor
    ResumeMatcher --> TextAnalyzer
    ResumeMatcher --> MatchingEngine
    ResumeMatcher --> ResultVisualizer
    ResumeMatcher --> ResumeData
    MatchingEngine --> ResumeData
    ResultVisualizer --> ResumeData