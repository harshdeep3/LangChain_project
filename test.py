from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

if __name__ == "__main__":
    data_file_path = str(Path(__file__).parent / "data")

    file1 = "\\doc_1.pdf"
    file2 = "\\doc_2.pdf"

    # load pdf file data
    loader_file1 = PyPDFLoader(data_file_path + file1)
    loader_file2 = PyPDFLoader(data_file_path + file2)
    doc1 = loader_file1.load()
    doc2 = loader_file2.load()

    # Take first page or chunk for simplicity
    # doc1_test = doc1[0].page_content

    # Extract Structured Data
    model = ChatOllama(model="llama3.1")

    extract_template = PromptTemplate.from_template(
        """
    You are an intelligent assistant.
    Extract the following structured fields from the document:
    - Title
    - Authors
    - Abstract
    - Methods
    - Results
    - Conclusion

    Document:
    {document}

    Respond in the following JSON format:
    {{
        "title": "...",
        "authors": "...",
        "abstract": "...",
        "methods": "...",
        "results": "...",
        "conclusion": "..."
    }}
    """
    )

    extract_chain = extract_template | model

    structured_data_doc1 = extract_chain.invoke({"document": doc1})
    structured_data_doc2 = extract_chain.invoke({"document": doc2})

    # Second LLM Chain - simplify for audience
    # summarize_template = PromptTemplate.from_template(
    # """
    # Take the following structured research paper data and summarize it for a general audience
    # (e.g., a high school student). Be clear, engaging, and easy to understand.

    # Structured Data:
    # {structured}
    # """
    # )

    # Second LLM Chain - compare doc1 and doc2
    summarize_template = PromptTemplate.from_template(
        """
    Take the structured research paper data from both parpers and compare it. Summarise it for a general audience
    (e.g., a high school student). Be clear, engaging, and easy to understand.

    Structured Data:
    Document 1:
    {document_1}
    
    Document 2:
    {document_2}
    """
    )

    summarize_chain = summarize_template | model
    simplified_summary = summarize_chain.invoke(
        {
            "document_1": structured_data_doc1,
            "document_2": structured_data_doc2,
        }
    )
    print("\n[SIMPLIFIED SUMMARY]\n", simplified_summary)
