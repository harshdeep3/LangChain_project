import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


def extract_structured_data(model, document: str) -> dict:
    """
    Extract structured data from a research paper document.
    Args:
        document (str): The content of the research paper document.
    Returns:
        dict: A dictionary containing structured fields like title, authors, abstract, methods, results,
    """
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

    # First LLM Chain - extract structured data from the document
    # then check if the output is useful and in the correct format
    return check_return_info(model, extract_chain.invoke({"document": document}))


def check_return_info(model, llm_output: str) -> dict:
    """
    Check if the LLM output is useful and in the correct format.
    Args:
        llm_output (str): The output from the LLM.
    Returns:
        dict: A dictionary containing the structured data if the output is valid.
    """
    check_template = PromptTemplate.from_template(
        """
        Check the output of the LLM is useful and is in the following format:
        - Title
        - Authors
        - Abstract
        - Methods
        - Results
        - Conclusion
        
        Document:
        {llm_output}
        
        If the output is not in the correct format, please reformat it to match the required structure.
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

    check_chain = check_template | model
    return check_chain.invoke({"llm_output": llm_output})


def compare_doc(model, structured_data_doc1: dict, structured_data_doc2: dict) -> str:
    """
    This function is not used in this script but can be used to summarize the structured data.
    """
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

    return simplified_summary.content


if __name__ == "__main__":
    data_file_path = str(Path(__file__).parent / "data")

    # create argepares
    parser = argparse.ArgumentParser(description="File paths to document, which will be comapred")

    # add arguements for parser
    parser.add_argument("--doc1", type=str, required=True, help="File path to document 1")
    parser.add_argument("--doc2", type=str, required=True, help="File path to document 2")

    args = parser.parse_args()

    # load pdf file data
    loader_file1 = PyPDFLoader(args.doc1)
    loader_file2 = PyPDFLoader(args.doc2)
    doc1 = loader_file1.load()
    doc2 = loader_file2.load()

    # Extract Structured Data
    model = ChatOllama(model="llama3.1")

    # get structured data from both documents
    structured_data_doc1 = extract_structured_data(model, doc1)
    structured_data_doc2 = extract_structured_data(model, doc2)

    # compare the structured data
    compared_info = compare_doc(model, structured_data_doc1, structured_data_doc2)

    print("\n[SIMPLIFIED SUMMARY]\n", compared_info)
