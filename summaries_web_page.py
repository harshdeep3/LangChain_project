import requests
import argparse
from bs4 import BeautifulSoup
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


def fetch_webpage(url: str) -> str:
    """
    Connect to a url infromaiton from a url.

    Args:
        url (_type_): url of a webpage

    Returns:
        _type_: text from webpage.
    """

    response = requests.get(url)
    return response.text


def extract_text_from_html(html: str) -> str:
    """
    This converts unstrctured text from webpage request to clean data

    Args:
        html (str): text data

    Returns:
        str : structured text data
    """
    soup = BeautifulSoup(html, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def summarise_text(text: str, llm) -> str:
    """
    This summaries the text from the webpage using a LLM

    Args:
        text (str): text from webpage
        llm (_type_): LLM used to summarise
    """
    prompt = PromptTemplate.from_template(
        """
        Summarise the following web page content in simple terms for a general audience:

        {content}
        """
    )
    extract_chain = prompt | llm

    summarised_data = extract_chain.invoke({"content": text})

    return summarised_data.content


if __name__ == "__main__":
    
    # create argepares
    parser = argparse.ArgumentParser(description="URL of Webpage to be summarised")
    
    # add arguements for parser
    parser.add_argument('--webpage', type=str, required=True, help="URL of webpage")
    
    args = parser.parse_args()
    
    raw_html = fetch_webpage(args.webpage)
    cleaned_text = extract_text_from_html(raw_html)

    # Init LLM
    llm = ChatOllama(model="llama3.1")  # You can use any other Ollama-supported model

    # Summarise
    summary = summarise_text(cleaned_text, llm)  # Truncate if needed
    print("\n--- SUMMARY ---\n")
    print(summary)
