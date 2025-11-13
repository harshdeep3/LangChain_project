"""
Complete Example: Setting up and using Agentic RAG
This example shows how to:
1. Prepare multiple document collections
2. Use basic agentic RAG
3. Use advanced agentic RAG
4. Compare results
"""

import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Import our agentic RAG systems
from agentic_rag_basic import AgenticRAG
from agentic_rag_advanced import AdvancedAgenticRAG


def setup_document_collections():
    """
    Step 1: Create multiple document collections
    This simulates having different types of documents
    """
    print("="*80)
    print("STEP 1: Setting up Document Collections")
    print("="*80 + "\n")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Example: Create 3 different collections

    # Collection 1: Research Papers
    print("📚 Creating 'research_papers' collection...")
    research_dir = "./data/research"  # Your research papers directory

    if os.path.exists(research_dir):
        loader = DirectoryLoader(research_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        research_docs = loader.load()
        research_chunks = text_splitter.split_documents(research_docs)

        research_store = Chroma.from_documents(
            research_chunks,
            embeddings,
            collection_name="research_collection",
            persist_directory="./chroma_research"
        )
        print(f"✅ Created research collection with {len(research_chunks)} chunks\n")
    else:
        print(f"⚠️  Directory {research_dir} not found. Skipping...\n")

    # Collection 2: Technical Documentation
    print("📖 Creating 'technical_docs' collection...")
    docs_dir = "./data/docs"  # Your documentation directory

    if os.path.exists(docs_dir):
        loader = DirectoryLoader(docs_dir, glob="**/*.{txt,md}", loader_cls=TextLoader)
        doc_docs = loader.load()
        doc_chunks = text_splitter.split_documents(doc_docs)

        doc_store = Chroma.from_documents(
            doc_chunks,
            embeddings,
            collection_name="docs_collection",
            persist_directory="./chroma_docs"
        )
        print(f"✅ Created docs collection with {len(doc_chunks)} chunks\n")
    else:
        print(f"⚠️  Directory {docs_dir} not found. Skipping...\n")

    # Collection 3: Meeting Notes
    print("📝 Creating 'meeting_notes' collection...")
    meetings_dir = "./data/meetings"  # Your meeting notes directory

    if os.path.exists(meetings_dir):
        loader = DirectoryLoader(meetings_dir, glob="**/*.txt", loader_cls=TextLoader)
        meeting_docs = loader.load()
        meeting_chunks = text_splitter.split_documents(meeting_docs)

        meeting_store = Chroma.from_documents(
            meeting_chunks,
            embeddings,
            collection_name="meetings_collection",
            persist_directory="./chroma_meetings"
        )
        print(f"✅ Created meetings collection with {len(meeting_chunks)} chunks\n")
    else:
        print(f"⚠️  Directory {meetings_dir} not found. Skipping...\n")

    print("✅ Document collections setup complete!\n")


def example_basic_agentic_rag():
    """
    Step 2: Use Basic Agentic RAG
    """
    print("\n" + "="*80)
    print("STEP 2: Using Basic Agentic RAG")
    print("="*80 + "\n")

    # Initialize
    agent_rag = AgenticRAG(llm_model="llama3.1")

    # Add collections
    print("Adding collections to agent...\n")

    agent_rag.add_collection(
        name="research_papers",
        description="Academic research papers on AI, machine learning, and deep learning",
        persist_directory="./chroma_research",
        collection_name="research_collection"
    )

    agent_rag.add_collection(
        name="technical_docs",
        description="Technical documentation, API references, and implementation guides",
        persist_directory="./chroma_docs",
        collection_name="docs_collection"
    )

    agent_rag.add_collection(
        name="meeting_notes",
        description="Meeting notes, discussions, and decisions from team meetings",
        persist_directory="./chroma_meetings",
        collection_name="meetings_collection"
    )

    # Create agent
    agent_rag.create_agent()

    # Example queries
    print("\n" + "="*80)
    print("RUNNING QUERIES")
    print("="*80 + "\n")

    queries = [
        "What are the key findings in our research papers about neural networks?",
        "How should I implement user authentication according to our documentation?",
        "What was decided about the project timeline in recent meetings?",
    ]

    results = []
    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)

        result = agent_rag.query(query)
        results.append(result)

        print(f"\n💡 Answer:\n{result['answer']}\n")

    return results


def example_advanced_agentic_rag():
    """
    Step 3: Use Advanced Agentic RAG with more capabilities
    """
    print("\n" + "="*80)
    print("STEP 3: Using Advanced Agentic RAG")
    print("="*80 + "\n")

    # Initialize
    agent_rag = AdvancedAgenticRAG(llm_model="llama3.1")

    # Add collections
    print("Adding collections to advanced agent...\n")

    agent_rag.add_collection(
        name="research",
        description="Research papers on AI and ML",
        persist_directory="./chroma_research",
        collection_name="research_collection",
        metadata_fields=["source", "filename", "page", "date"]
    )

    agent_rag.add_collection(
        name="docs",
        description="Technical documentation",
        persist_directory="./chroma_docs",
        collection_name="docs_collection",
        metadata_fields=["source", "filename"]
    )

    # Create agent with advanced tools
    agent_rag.create_agent()

    # Advanced queries
    print("\n" + "="*80)
    print("RUNNING ADVANCED QUERIES")
    print("="*80 + "\n")

    advanced_queries = [
        "List all documents available, then search for information about transformers",
        "Find high-confidence results about API authentication",
        "Search research papers and synthesize findings about attention mechanisms",
    ]

    results = []
    for query in advanced_queries:
        print(f"\n{'='*80}")
        print(f"Advanced Query: {query}")
        print('='*80)

        result = agent_rag.query(query, use_history=True)
        results.append(result)

        print(f"\n💡 Answer:\n{result['answer']}\n")

    # Show search statistics
    print("\n📊 Search Statistics:")
    stats = agent_rag.get_search_stats()
    for collection, data in stats.items():
        print(f"\n{collection}:")
        print(f"  Total searches: {data['total_searches']}")

    return results


def compare_approaches():
    """
    Step 4: Compare different approaches
    """
    print("\n" + "="*80)
    print("STEP 4: Comparing Approaches")
    print("="*80 + "\n")

    test_query = "What are the main topics covered in the research papers?"

    print(f"Test Query: {test_query}\n")

    # Basic approach
    print("\n1️⃣  BASIC AGENTIC RAG")
    print("-"*80)
    basic_agent = AgenticRAG()
    basic_agent.add_collection(
        "research", "Research papers", "./chroma_research", "research_collection"
    )
    basic_agent.create_agent()
    basic_result = basic_agent.query(test_query)
    print(f"Answer: {basic_result['answer'][:200]}...\n")

    # Advanced approach
    print("\n2️⃣  ADVANCED AGENTIC RAG")
    print("-"*80)
    advanced_agent = AdvancedAgenticRAG()
    advanced_agent.add_collection(
        "research", "Research papers", "./chroma_research", "research_collection"
    )
    advanced_agent.create_agent()
    advanced_result = advanced_agent.query(test_query)
    print(f"Answer: {advanced_result['answer'][:200]}...\n")

    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print("""
Basic Agentic RAG:
  ✅ Simple to use
  ✅ Fast setup
  ✅ Good for straightforward queries
  ❌ Limited tools
  ❌ No metadata filtering
  ❌ No query rewriting

Advanced Agentic RAG:
  ✅ Multiple specialized tools
  ✅ Metadata filtering
  ✅ Query rewriting
  ✅ High-confidence search
  ✅ Search statistics
  ✅ Conversation history
  ❌ More complex
  ❌ Slower due to more reasoning
    """)


def main():
    """
    Main example workflow
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    AGENTIC RAG COMPLETE EXAMPLE                              ║
║                                                                              ║
║  This example demonstrates how to build and use agentic RAG systems          ║
║  that can intelligently search multiple document collections.                ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Uncomment to setup collections (only need to do this once)
    # setup_document_collections()

    # Example 1: Basic Agentic RAG
    print("\n🤖 Example 1: Basic Agentic RAG")
    try:
        example_basic_agentic_rag()
    except Exception as e:
        print(f"❌ Error in basic example: {e}")
        print("Make sure you have document collections set up!")

    # Example 2: Advanced Agentic RAG
    print("\n🤖 Example 2: Advanced Agentic RAG")
    try:
        example_advanced_agentic_rag()
    except Exception as e:
        print(f"❌ Error in advanced example: {e}")

    # Example 3: Compare approaches
    print("\n🤖 Example 3: Comparing Approaches")
    try:
        compare_approaches()
    except Exception as e:
        print(f"❌ Error in comparison: {e}")

    print("\n" + "="*80)
    print("✅ EXAMPLES COMPLETE")
    print("="*80)
    print("""
Next Steps:
1. Customize the collections for your specific use case
2. Adjust chunk sizes and overlap based on your documents
3. Tune the LLM prompts for better results
4. Add more specialized tools in advanced agent
5. Try the LangGraph version for explicit workflow control

For production use:
- Add error handling and logging
- Implement caching for repeated queries
- Add authentication and rate limiting
- Monitor agent performance and costs
- Use async operations for better performance
    """)


if __name__ == "__main__":
    main()
