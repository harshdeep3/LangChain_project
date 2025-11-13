"""
Comprehensive RAG System using LangChain + ChromaDB
This example demonstrates best practices for building a production-ready RAG system
with persistent vector storage on your local machine.
"""

import argparse
from pathlib import Path
from typing import List, Optional

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


class RAGSystem:
    """Production-ready RAG system with ChromaDB persistence"""

    def __init__(
        self,
        collection_name: str = "langchain_collection",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "llama3.1",
    ):
        """
        Initialize RAG system with ChromaDB

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector database
            embedding_model: HuggingFace embedding model name
            llm_model: Ollama model name
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},  # Use "cuda" if GPU available
            encode_kwargs={"normalize_embeddings": True},  # Important for similarity search
        )

        # Initialize or load existing ChromaDB
        self.vectorstore = None
        self.load_or_create_vectorstore()

    def load_or_create_vectorstore(self):
        """Load existing ChromaDB or create a new one"""
        persist_path = Path(self.persist_directory)

        if persist_path.exists() and any(persist_path.iterdir()):
            print(f"📂 Loading existing ChromaDB from {self.persist_directory}")
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            print(f"✅ Loaded {self.vectorstore._collection.count()} documents")
        else:
            print(f"📂 Creating new ChromaDB at {self.persist_directory}")
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            print("✅ New ChromaDB created")

    def load_documents_from_directory(self, directory: str, file_types: Optional[List[str]] = None) -> List[Document]:
        """
        Load documents from a directory

        Args:
            directory: Path to directory containing documents
            file_types: List of file extensions to load (e.g., ['.pdf', '.txt', '.md'])

        Returns:
            List of loaded documents
        """
        if file_types is None:
            file_types = [".pdf", ".txt", ".md"]

        all_docs = []
        directory_path = Path(directory)

        print(f"\n📚 Loading documents from {directory}")

        for file_type in file_types:
            if file_type == ".pdf":
                loader = DirectoryLoader(
                    directory,
                    glob=f"**/*{file_type}",
                    loader_cls=PyPDFLoader,
                    show_progress=True,
                )
            elif file_type == ".md":
                loader = DirectoryLoader(
                    directory,
                    glob=f"**/*{file_type}",
                    loader_cls=UnstructuredMarkdownLoader,
                    show_progress=True,
                )
            else:  # .txt and others
                loader = DirectoryLoader(
                    directory,
                    glob=f"**/*{file_type}",
                    loader_cls=TextLoader,
                    show_progress=True,
                )

            try:
                docs = loader.load()
                all_docs.extend(docs)
                print(f"  ✓ Loaded {len(docs)} {file_type} documents")
            except Exception as e:
                print(f"  ✗ Error loading {file_type} files: {e}")

        return all_docs

    def chunk_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[Document]:
        """
        Split documents into chunks for better retrieval

        Args:
            documents: List of documents to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks (helps maintain context)

        Returns:
            List of chunked documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],  # Try to split on paragraphs first
        )

        chunks = text_splitter.split_documents(documents)
        print(f"📄 Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def add_documents(self, documents: List[Document]):
        """Add new documents to the vector store"""
        if not documents:
            print("⚠️  No documents to add")
            return

        print(f"\n➕ Adding {len(documents)} documents to ChromaDB...")
        self.vectorstore.add_documents(documents)
        print(f"✅ Added {len(documents)} documents successfully")
        print(f"📊 Total documents in database: {self.vectorstore._collection.count()}")

    def build_qa_chain(self, k: int = 4, search_type: str = "similarity"):
        """
        Build a RetrievalQA chain

        Args:
            k: Number of documents to retrieve
            search_type: Type of search ('similarity', 'mmr', 'similarity_score_threshold')

        Returns:
            RetrievalQA chain
        """
        # Configure retriever
        retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k},
        )

        # Initialize LLM
        llm = ChatOllama(model=self.llm_model, temperature=0)

        # Custom prompt for better answers
        prompt_template = """You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.
        If you don't know the answer based on the context, say "I don't have enough information to answer that question."

        Context:
        {context}

        Question: {question}

        Answer: """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Build QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # "stuff", "map_reduce", "refine", "map_rerank"
            retriever=retriever,
            return_source_documents=True,  # Return sources for citation
            chain_type_kwargs={"prompt": prompt},
        )

        return qa_chain

    def query(
        self,
        question: str,
        k: int = 4,
        search_type: str = "similarity",
        return_sources: bool = True,
    ):
        """
        Query the RAG system

        Args:
            question: Question to ask
            k: Number of documents to retrieve
            search_type: Type of search
            return_sources: Whether to return source documents

        Returns:
            Dictionary with answer and optionally source documents
        """
        qa_chain = self.build_qa_chain(k=k, search_type=search_type)
        result = qa_chain.invoke({"query": question})

        response = {"question": result["query"], "answer": result["result"]}

        if return_sources and "source_documents" in result:
            response["sources"] = []
            for i, doc in enumerate(result["source_documents"], 1):
                source_info = {
                    "number": i,
                    "content": doc.page_content[:200] + "...",  # First 200 chars
                    "metadata": doc.metadata,
                }
                response["sources"].append(source_info)

        return response

    def similarity_search(self, query: str, k: int = 4):
        """
        Perform similarity search without LLM

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of similar documents
        """
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs

    def delete_collection(self):
        """Delete the entire collection"""
        print(f"🗑️  Deleting collection '{self.collection_name}'...")
        self.vectorstore.delete_collection()
        print("✅ Collection deleted")

    def get_stats(self):
        """Get statistics about the vector database"""
        count = self.vectorstore._collection.count()
        return {"total_documents": count, "collection_name": self.collection_name}


def main():
    """Example usage of the RAG system"""

    parser = argparse.ArgumentParser(description="RAG System with ChromaDB")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["add", "query", "stats", "reset"],
        default="query",
        help="Mode of operation",
    )
    parser.add_argument("--docs-dir", type=str, help="Directory containing documents to add")
    parser.add_argument("--question", type=str, help="Question to ask")
    parser.add_argument(
        "--collection",
        type=str,
        default="langchain_docs",
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./chroma_db",
        help="Directory to persist ChromaDB",
    )

    args = parser.parse_args()

    # Initialize RAG system
    rag = RAGSystem(
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="llama3.1",
    )

    if args.mode == "add":
        if not args.docs_dir:
            print("❌ Please provide --docs-dir for adding documents")
            return

        # Load and process documents
        documents = rag.load_documents_from_directory(args.docs_dir)

        if documents:
            # Chunk documents
            chunks = rag.chunk_documents(documents, chunk_size=1000, chunk_overlap=200)

            # Add to vector store
            rag.add_documents(chunks)
        else:
            print("⚠️  No documents found in the directory")

    elif args.mode == "query":
        if not args.question:
            print("❌ Please provide --question for querying")
            return

        # Query the system
        result = rag.query(args.question, k=4, return_sources=True)

        print("\n" + "=" * 80)
        print(f"❓ Question: {result['question']}")
        print("=" * 80)
        print(f"\n💡 Answer:\n{result['answer']}\n")

        if "sources" in result:
            print("\n📚 Sources:")
            for source in result["sources"]:
                print(f"\n  [{source['number']}] {source['metadata']}")
                print(f"      {source['content']}")

    elif args.mode == "stats":
        stats = rag.get_stats()
        print("\n📊 Vector Database Statistics:")
        print(f"  Collection: {stats['collection_name']}")
        print(f"  Total Documents: {stats['total_documents']}")

    elif args.mode == "reset":
        confirm = input(f"⚠️  Are you sure you want to delete collection '{args.collection}'? (yes/no): ")
        if confirm.lower() == "yes":
            rag.delete_collection()


if __name__ == "__main__":
    # Example 1: Add documents
    # python rag_chromadb_example.py --mode add --docs-dir ./data

    # Example 2: Query
    # python rag_chromadb_example.py --mode query --question "What is the main topic?"

    # Example 3: Check stats
    # python rag_chromadb_example.py --mode stats

    # Example 4: Reset database
    # python rag_chromadb_example.py --mode reset

    main()
