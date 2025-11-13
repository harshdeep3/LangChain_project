"""
Quick Start RAG Example with ChromaDB
Simple example to get started quickly with RAG using LangChain + ChromaDB
"""

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Configuration
DATA_DIR = "./data"  # Directory with your documents
PERSIST_DIR = "./chroma_db_quickstart"  # Where to save the vector database
COLLECTION_NAME = "quickstart_collection"


def create_vectorstore(documents):
    """Create and persist ChromaDB vector store"""
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector store with persistence
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    print(f"✅ Vector store created with {len(documents)} documents")
    return vectorstore


def load_existing_vectorstore():
    """Load existing ChromaDB from disk"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    print(f"✅ Loaded existing vector store")
    return vectorstore


def main():
    # Step 1: Load documents
    print("📚 Loading documents...")
    loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    # Step 2: Split into chunks
    print("\n📄 Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Step 3: Create or load vector store
    print("\n🔧 Creating vector store...")
    # Uncomment the next line to load existing vectorstore instead
    # vectorstore = load_existing_vectorstore()
    vectorstore = create_vectorstore(chunks)

    # Step 4: Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Step 5: Initialize LLM
    llm = ChatOllama(model="llama3.1", temperature=0)

    # Step 6: Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    # Step 7: Ask questions
    questions = [
        "What is the main topic of the documents?",
        "Summarize the key points.",
    ]

    for question in questions:
        print(f"\n{'='*80}")
        print(f"❓ Question: {question}")
        print('='*80)

        result = qa_chain.invoke({"query": question})

        print(f"\n💡 Answer:\n{result['result']}\n")

        # Show sources
        print("📚 Sources:")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"  [{i}] {doc.metadata.get('source', 'Unknown')}")


if __name__ == "__main__":
    main()
