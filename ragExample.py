import argparse
from pathlib import Path

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from youtube_transcript_api import YouTubeTranscriptApi


# 🧾 Load PDF and Chunk
def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(pages)


# 📺 Load YouTube Transcript
def load_youtube_transcript(api: YouTubeTranscriptApi, video_id: str):

    transcript = api.fetch(video_id)
    combined_text = " ".join([snippet.text for snippet in transcript.snippets])
    return [Document(page_content=combined_text, metadata={"source": f"https://youtube.com/watch?v={video_id}"})]


# 📚 Build Vectorstore (FAISS)
def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)


# 🧠 Setup RetrievalQA with Ollama
def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()

    llm = ChatOllama(model="llama3.1")

    prompt = PromptTemplate.from_template(
        """
    Use the following context to answer the question:
    {context}

    Question: {question}
    """
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


# ❓ Ask Question
def ask_question(chain, question):
    return chain.invoke(question)


# 🚀 Example Run
if __name__ == "__main__":

    data_file_path = str(Path(__file__).parent / "data")

    # create argepares
    parser = argparse.ArgumentParser(description="File paths to document, which will be comapred")

    # add arguements for parser
    parser.add_argument("--doc", type=str, required=True, help="File path to document")
    parser.add_argument("--video", type=str, required=True, help="Video id for YouTube transcript")

    args = parser.parse_args()

    pdf_text = load_pdf(args.doc)

    api = YouTubeTranscriptApi()
    youtube_transcript = load_youtube_transcript(api, args.video)  # Example YouTube video ID

    all_docs = pdf_text + youtube_transcript

    vectorstore = build_vectorstore(all_docs)
    qa_chain = build_qa_chain(vectorstore)

    # Ask a question
    question = "Summarise the key ideas from the video and paper."
    result = ask_question(qa_chain, question)

    print("\n question:\n", result["query"])
    print("\n📌 Answer:\n", result["result"])
