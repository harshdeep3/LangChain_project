"""
RAG Debugging Script - Diagnose Retrieval Issues
This script helps identify why your RAG is retrieving wrong information
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def debug_retrieval(vectorstore, query: str, k: int = 4):
    """
    Detailed debugging of what's being retrieved
    """
    print("\n" + "=" * 80)
    print(f"🔍 DEBUGGING QUERY: {query}")
    print("=" * 80)

    # Method 1: Similarity search with scores
    print("\n1️⃣  SIMILARITY SEARCH WITH SCORES")
    print("-" * 80)
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)

    for i, (doc, score) in enumerate(docs_with_scores, 1):
        print(f"\n📄 Result #{i} | Score: {score:.4f} (lower is better)")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content (first 300 chars):\n{doc.page_content[:300]}...")
        print(f"\nFull metadata: {doc.metadata}")
        print("-" * 80)

    # Method 2: Check what the retriever actually returns
    print("\n2️⃣  RETRIEVER OUTPUT (what RAG sees)")
    print("-" * 80)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.get_relevant_documents(query)

    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n📄 Document #{i}")
        print(f"Content length: {len(doc.page_content)} chars")
        print(f"Content preview:\n{doc.page_content[:300]}...")
        print(f"Metadata: {doc.metadata}")
        print("-" * 80)

    # Method 3: Show embedding similarity
    print("\n3️⃣  EMBEDDING ANALYSIS")
    print("-" * 80)
    embeddings = vectorstore._embedding_function
    query_embedding = embeddings.embed_query(query)
    print(f"Query embedding dimension: {len(query_embedding)}")
    print(f"Query embedding (first 10 values): {query_embedding[:10]}")

    # Method 4: Collection stats
    print("\n4️⃣  COLLECTION STATISTICS")
    print("-" * 80)
    collection = vectorstore._collection
    print(f"Total documents in collection: {collection.count()}")

    # Sample a few random documents to see what's stored
    sample = collection.get(limit=3, include=["documents", "metadatas"])
    print(f"\n📋 Random sample from collection:")
    for i, (doc, meta) in enumerate(zip(sample['documents'], sample['metadatas']), 1):
        print(f"\nSample #{i}")
        print(f"Content (first 200 chars): {doc[:200]}...")
        print(f"Metadata: {meta}")


def test_chunking_strategies(text: str, query: str):
    """
    Test different chunking strategies to see which retrieves best
    """
    print("\n" + "=" * 80)
    print("🔬 TESTING DIFFERENT CHUNKING STRATEGIES")
    print("=" * 80)

    strategies = [
        {"name": "Small chunks (500/100)", "chunk_size": 500, "chunk_overlap": 100},
        {"name": "Medium chunks (1000/200)", "chunk_size": 1000, "chunk_overlap": 200},
        {"name": "Large chunks (1500/300)", "chunk_size": 1500, "chunk_overlap": 300},
        {"name": "High overlap (1000/500)", "chunk_size": 1000, "chunk_overlap": 500},
    ]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"Testing: {strategy['name']}")
        print(f"{'='*80}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=strategy['chunk_size'],
            chunk_overlap=strategy['chunk_overlap']
        )

        docs = [Document(page_content=text, metadata={"source": "test"})]
        chunks = splitter.split_documents(docs)

        print(f"Number of chunks created: {len(chunks)}")
        print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")

        # Create temporary vectorstore
        temp_store = Chroma.from_documents(
            chunks,
            embeddings,
            collection_name=f"test_{strategy['chunk_size']}"
        )

        # Test retrieval
        results = temp_store.similarity_search_with_score(query, k=2)
        print(f"\nTop result score: {results[0][1]:.4f}")
        print(f"Top result preview: {results[0][0].page_content[:150]}...")

        # Cleanup
        temp_store.delete_collection()


def check_embedding_quality(texts: list[str], queries: list[str]):
    """
    Test if embeddings are working correctly
    """
    print("\n" + "=" * 80)
    print("🧪 TESTING EMBEDDING QUALITY")
    print("=" * 80)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("\n1️⃣  Embedding similar texts")
    print("-" * 80)

    # Test if similar texts have similar embeddings
    similar_texts = [
        "The cat sat on the mat.",
        "A cat was sitting on a mat.",
        "Dogs are playing in the park."
    ]

    embeddings_list = [embeddings.embed_query(text) for text in similar_texts]

    # Calculate cosine similarity
    import numpy as np

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f"Similarity (text1 vs text2): {cosine_similarity(embeddings_list[0], embeddings_list[1]):.4f}")
    print(f"Similarity (text1 vs text3): {cosine_similarity(embeddings_list[0], embeddings_list[2]):.4f}")
    print("Expected: First should be higher (more similar)")

    print("\n2️⃣  Testing your actual queries")
    print("-" * 80)

    for i, query in enumerate(queries, 1):
        emb = embeddings.embed_query(query)
        print(f"Query {i}: {query}")
        print(f"  Embedding length: {len(emb)}")
        print(f"  Embedding norm: {np.linalg.norm(emb):.4f}")


def fix_retrieval_issues(vectorstore, query: str):
    """
    Apply fixes and show results
    """
    print("\n" + "=" * 80)
    print("🔧 APPLYING FIXES")
    print("=" * 80)

    print("\n✅ FIX 1: Increase number of retrieved documents")
    print("-" * 80)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.get_relevant_documents(query)
    print(f"Retrieved {len(docs)} documents instead of default 4")
    print(f"Top result: {docs[0].page_content[:200]}...")

    print("\n✅ FIX 2: Use MMR for diversity")
    print("-" * 80)
    retriever_mmr = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
    )
    docs_mmr = retriever_mmr.get_relevant_documents(query)
    print(f"MMR retrieved {len(docs_mmr)} diverse documents")
    print(f"Top result: {docs_mmr[0].page_content[:200]}...")

    print("\n✅ FIX 3: Use score threshold")
    print("-" * 80)
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=10)
    # Filter by score (ChromaDB uses L2 distance, lower is better)
    threshold = 1.0  # Adjust based on your data
    filtered_docs = [(doc, score) for doc, score in docs_with_scores if score < threshold]
    print(f"Documents passing threshold {threshold}: {len(filtered_docs)}")
    if filtered_docs:
        print(f"Best score: {filtered_docs[0][1]:.4f}")
        print(f"Top result: {filtered_docs[0][0].page_content[:200]}...")


def main():
    """
    Main diagnostic routine
    """
    print("🏥 RAG DIAGNOSTIC TOOL")
    print("=" * 80)
    print("This tool helps diagnose why your RAG is retrieving wrong information\n")

    # Load your existing vectorstore
    print("📂 Loading your existing ChromaDB...")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Adjust these to match your setup
        vectorstore = Chroma(
            collection_name="langchain_collection",  # Change to your collection name
            embedding_function=embeddings,
            persist_directory="./chroma_db"  # Change to your persist directory
        )

        print(f"✅ Loaded collection with {vectorstore._collection.count()} documents\n")

        # Test with a sample query
        test_query = input("Enter a test query (or press Enter for default): ").strip()
        if not test_query:
            test_query = "What is the main topic?"

        # Run diagnostics
        debug_retrieval(vectorstore, test_query, k=5)

        # Show fixes
        fix_retrieval_issues(vectorstore, test_query)

        # Additional tests
        print("\n\n" + "=" * 80)
        print("💡 RECOMMENDATIONS")
        print("=" * 80)

        docs_with_scores = vectorstore.similarity_search_with_score(test_query, k=5)
        avg_score = sum(score for _, score in docs_with_scores) / len(docs_with_scores)

        print(f"\nAverage retrieval score: {avg_score:.4f}")

        if avg_score > 1.5:
            print("⚠️  HIGH SCORES - Your documents may not match your queries well")
            print("   Solutions:")
            print("   1. Check if you're querying about content that's actually in your docs")
            print("   2. Try smaller chunk sizes (500-800 instead of 1000+)")
            print("   3. Increase chunk overlap (300-500)")
            print("   4. Add more context to your queries")
        elif avg_score > 1.0:
            print("⚠️  MODERATE SCORES - Retrieval could be better")
            print("   Solutions:")
            print("   1. Increase chunk overlap to 200-300")
            print("   2. Use MMR search for better diversity")
            print("   3. Retrieve more documents (k=6-8)")
        else:
            print("✅ GOOD SCORES - Retrieval is working well")
            print("   If answers are still wrong:")
            print("   1. Check your prompt template")
            print("   2. Verify LLM is using the retrieved context")
            print("   3. Try a different LLM model")

        # Check for common issues
        print("\n🔍 COMMON ISSUES CHECK")
        print("-" * 80)

        sample = vectorstore._collection.get(limit=5, include=["documents"])
        avg_chunk_length = sum(len(doc) for doc in sample['documents']) / len(sample['documents'])

        print(f"Average chunk length: {avg_chunk_length:.0f} characters")

        if avg_chunk_length < 200:
            print("⚠️  Chunks are too small - try larger chunk_size (800-1200)")
        elif avg_chunk_length > 2000:
            print("⚠️  Chunks are too large - try smaller chunk_size (600-1000)")
        else:
            print("✅ Chunk size looks good")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure to update the collection_name and persist_directory in the script")
        print("to match your actual ChromaDB setup")


if __name__ == "__main__":
    main()
