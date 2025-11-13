"""
Advanced RAG System with ChromaDB
Demonstrates advanced features like metadata filtering, hybrid search, and custom retrieval
"""

from typing import Dict, List, Optional

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


class AdvancedRAG:
    """Advanced RAG system with enhanced retrieval strategies"""

    def __init__(self, persist_directory: str = "./chroma_advanced"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.llm = ChatOllama(model="llama3.1", temperature=0)
        self.vectorstore = None

    def load_and_process_documents(
        self, directory: str, add_metadata: bool = True
    ) -> List[Document]:
        """Load documents with enhanced metadata"""
        print(f"📚 Loading documents from {directory}...")

        loader = DirectoryLoader(
            directory, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
        )
        documents = loader.load()

        # Add custom metadata
        if add_metadata:
            for doc in documents:
                # Add source filename
                doc.metadata["filename"] = doc.metadata.get("source", "").split("/")[-1]
                # Add timestamp
                from datetime import datetime

                doc.metadata["indexed_at"] = datetime.now().isoformat()

        # Advanced chunking with metadata preservation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,  # Track position in original document
        )

        chunks = text_splitter.split_documents(documents)
        print(f"📄 Created {len(chunks)} chunks from {len(documents)} documents")

        return chunks

    def create_vectorstore(self, documents: List[Document]):
        """Create vectorstore with documents"""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        print("✅ Vector store created")

    def load_vectorstore(self):
        """Load existing vectorstore"""
        self.vectorstore = Chroma(
            embedding_function=self.embeddings, persist_directory=self.persist_directory
        )
        print("✅ Vector store loaded")

    def query_with_metadata_filter(
        self, question: str, metadata_filter: Optional[Dict] = None, k: int = 4
    ):
        """Query with metadata filtering"""
        if metadata_filter:
            # Example: {"filename": "document.pdf"}
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": k, "filter": metadata_filter}
            )
        else:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        return qa_chain.invoke({"query": question})

    def query_with_mmr(self, question: str, k: int = 4, fetch_k: int = 20):
        """
        Query using Maximum Marginal Relevance (MMR)
        MMR balances relevance and diversity in retrieved documents
        """
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5},
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        return qa_chain.invoke({"query": question})

    def query_with_score_threshold(
        self, question: str, score_threshold: float = 0.5, k: int = 4
    ):
        """Query with similarity score threshold"""
        retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score_threshold, "k": k},
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        return qa_chain.invoke({"query": question})

    def query_with_compression(self, question: str, k: int = 8):
        """
        Query with contextual compression
        Compresses retrieved documents to only relevant portions
        """
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        # Create compressor that extracts relevant portions
        compressor = LLMChainExtractor.from_llm(self.llm)

        # Wrap base retriever with compression
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=compression_retriever,
            return_source_documents=True,
        )

        return qa_chain.invoke({"query": question})

    def query_with_custom_prompt(self, question: str, k: int = 4):
        """Query with a custom prompt template"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        # Custom prompt for specific use case
        prompt_template = """You are an expert analyst. Use the following pieces of context to answer the question.
Provide a detailed answer with specific examples from the context.
If you cannot answer based on the context, explain what information is missing.

Context:
{context}

Question: {question}

Detailed Answer (include specific quotes and examples):"""

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

        return qa_chain.invoke({"query": question})

    def query_multi_query(self, question: str, k: int = 4):
        """
        Generate multiple variations of the query for better retrieval
        """
        # Generate query variations
        variations_prompt = f"""Generate 3 different versions of the following question to retrieve relevant documents:

Original question: {question}

Variations (one per line):"""

        variations_response = self.llm.invoke(variations_prompt)
        variations = [
            line.strip() for line in variations_response.content.split("\n") if line.strip()
        ][:3]

        print(f"🔍 Generated {len(variations)} query variations")

        # Retrieve documents for each variation
        all_docs = []
        for variation in variations:
            docs = self.vectorstore.similarity_search(variation, k=k)
            all_docs.extend(docs)

        # Remove duplicates based on content
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)

        print(f"📄 Retrieved {len(unique_docs)} unique documents")

        # Use retrieved docs to answer
        context = "\n\n".join([doc.page_content for doc in unique_docs[:k]])
        prompt = f"""Use the following context to answer the question:

Context:
{context}

Question: {question}

Answer:"""

        response = self.llm.invoke(prompt)

        return {
            "query": question,
            "result": response.content,
            "source_documents": unique_docs[:k],
        }


def main():
    """Demonstrate advanced RAG features"""

    rag = AdvancedRAG(persist_directory="./chroma_advanced")

    # Example: Load and create vectorstore
    # documents = rag.load_and_process_documents("./data")
    # rag.create_vectorstore(documents)

    # Load existing vectorstore
    rag.load_vectorstore()

    question = "What are the key findings in the research?"

    print("\n" + "=" * 80)
    print("1️⃣  Standard Query")
    print("=" * 80)
    result = rag.query_with_metadata_filter(question, k=4)
    print(f"💡 Answer: {result['result'][:200]}...\n")

    print("\n" + "=" * 80)
    print("2️⃣  Query with MMR (Maximum Marginal Relevance)")
    print("=" * 80)
    result = rag.query_with_mmr(question, k=4, fetch_k=20)
    print(f"💡 Answer: {result['result'][:200]}...\n")

    print("\n" + "=" * 80)
    print("3️⃣  Query with Score Threshold")
    print("=" * 80)
    result = rag.query_with_score_threshold(question, score_threshold=0.5, k=4)
    print(f"💡 Answer: {result['result'][:200]}...\n")

    print("\n" + "=" * 80)
    print("4️⃣  Query with Compression (extracts relevant portions)")
    print("=" * 80)
    result = rag.query_with_compression(question, k=8)
    print(f"💡 Answer: {result['result'][:200]}...\n")

    print("\n" + "=" * 80)
    print("5️⃣  Query with Custom Prompt")
    print("=" * 80)
    result = rag.query_with_custom_prompt(question, k=4)
    print(f"💡 Answer: {result['result'][:200]}...\n")

    print("\n" + "=" * 80)
    print("6️⃣  Multi-Query Approach")
    print("=" * 80)
    result = rag.query_multi_query(question, k=4)
    print(f"💡 Answer: {result['result'][:200]}...\n")

    # Example: Query with metadata filter
    # result = rag.query_with_metadata_filter(
    #     "What is mentioned about X?",
    #     metadata_filter={"filename": "specific_document.pdf"},
    #     k=4
    # )


if __name__ == "__main__":
    main()
