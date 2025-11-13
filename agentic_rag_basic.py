"""
Basic Agentic RAG System
An agent that can intelligently route queries to different document collections
and decide which documents to search based on the question.
"""

from typing import List, Dict, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


class DocumentCollection:
    """Represents a collection of documents on a specific topic"""

    def __init__(self, name: str, description: str, vectorstore: Chroma):
        self.name = name
        self.description = description
        self.vectorstore = vectorstore

    def search(self, query: str, k: int = 4) -> str:
        """Search this collection and return formatted results"""
        docs = self.vectorstore.similarity_search(query, k=k)

        if not docs:
            return f"No relevant information found in {self.name}"

        results = [f"=== Results from {self.name} ===\n"]
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            results.append(f"\n[{i}] Source: {source}")
            results.append(f"Content: {doc.page_content}\n")

        return "\n".join(results)


class AgenticRAG:
    """
    Agentic RAG system that can:
    1. Route queries to appropriate document collections
    2. Search multiple collections if needed
    3. Synthesize information from multiple sources
    4. Reason about which documents to use
    """

    def __init__(self, llm_model: str = "llama3.1"):
        self.llm = ChatOllama(model=llm_model, temperature=0)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
        self.collections: Dict[str, DocumentCollection] = {}
        self.agent = None
        self.agent_executor = None

    def add_collection(
        self,
        name: str,
        description: str,
        persist_directory: str,
        collection_name: Optional[str] = None
    ):
        """
        Add a document collection that the agent can search

        Args:
            name: Short name for the collection (e.g., "research_papers")
            description: Description of what's in this collection (used by agent to decide when to use it)
            persist_directory: Path to ChromaDB directory
            collection_name: ChromaDB collection name
        """
        vectorstore = Chroma(
            collection_name=collection_name or name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

        collection = DocumentCollection(name, description, vectorstore)
        self.collections[name] = collection

        print(f"✅ Added collection '{name}': {description}")
        print(f"   Documents: {vectorstore._collection.count()}")

    def create_agent(self):
        """Create the agent with tools for each document collection"""

        if not self.collections:
            raise ValueError("No document collections added. Use add_collection() first.")

        # Create a tool for each document collection
        tools = []
        for name, collection in self.collections.items():
            tool = Tool(
                name=f"search_{name}",
                func=lambda q, coll=collection: coll.search(q, k=4),
                description=f"{collection.description}. Use this to search: {name}"
            )
            tools.append(tool)

        # Create agent prompt
        tool_names = ", ".join([t.name for t in tools])
        tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])

        agent_prompt = PromptTemplate(
            template="""You are an intelligent research assistant with access to multiple document collections.
Your job is to answer questions by searching the appropriate document collections.

You have access to these tools:
{tool_descriptions}

IMPORTANT INSTRUCTIONS:
1. Think carefully about which document collection(s) to search based on the question
2. You can search multiple collections if the question requires information from different sources
3. After searching, synthesize the information into a coherent answer
4. If you can't find relevant information, say so clearly
5. Always cite which collection you found information in

Question: {input}

Thought Process:
{agent_scratchpad}""",
            input_variables=["input", "tool_descriptions", "agent_scratchpad", "tool_names"],
            partial_variables={"tool_descriptions": tool_descriptions, "tool_names": tool_names}
        )

        # Create ReAct agent
        self.agent = create_react_agent(self.llm, tools, agent_prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )

        print(f"\n🤖 Agent created with {len(tools)} tools")

    def query(self, question: str) -> Dict:
        """
        Query the agentic RAG system

        Args:
            question: User's question

        Returns:
            Dictionary with answer and metadata
        """
        if not self.agent_executor:
            raise ValueError("Agent not created. Call create_agent() first.")

        print(f"\n{'='*80}")
        print(f"❓ Question: {question}")
        print(f"{'='*80}\n")

        result = self.agent_executor.invoke({"input": question})

        return {
            "question": question,
            "answer": result["output"],
            "intermediate_steps": result.get("intermediate_steps", [])
        }

    def multi_collection_search(self, question: str, collection_names: List[str], k: int = 4) -> str:
        """
        Manually search multiple collections (without agent reasoning)

        Args:
            question: Search query
            collection_names: List of collection names to search
            k: Number of documents to retrieve from each

        Returns:
            Combined results from all collections
        """
        results = []

        for name in collection_names:
            if name in self.collections:
                print(f"🔍 Searching {name}...")
                collection_results = self.collections[name].search(question, k=k)
                results.append(collection_results)
            else:
                print(f"⚠️  Collection '{name}' not found")

        if not results:
            return "No results found in any collection"

        # Synthesize results with LLM
        combined_results = "\n\n".join(results)

        synthesis_prompt = f"""Based on the following search results from multiple document collections, provide a comprehensive answer to the question.

Question: {question}

Search Results:
{combined_results}

Comprehensive Answer (cite sources):"""

        response = self.llm.invoke(synthesis_prompt)
        return response.content


def main():
    """Example usage of Agentic RAG"""

    print("🤖 AGENTIC RAG SYSTEM - BASIC\n")

    # Initialize
    agent_rag = AgenticRAG(llm_model="llama3.1")

    # Add document collections
    # Example: Add different collections for different topics

    # Collection 1: Research papers
    agent_rag.add_collection(
        name="research_papers",
        description="Academic research papers on AI and machine learning",
        persist_directory="./chroma_research",
        collection_name="research_collection"
    )

    # Collection 2: Technical documentation
    agent_rag.add_collection(
        name="technical_docs",
        description="Technical documentation and API references",
        persist_directory="./chroma_docs",
        collection_name="docs_collection"
    )

    # Collection 3: Meeting notes
    agent_rag.add_collection(
        name="meeting_notes",
        description="Company meeting notes and discussions",
        persist_directory="./chroma_meetings",
        collection_name="meetings_collection"
    )

    # Create the agent
    agent_rag.create_agent()

    # Example queries
    questions = [
        "What are the latest findings in machine learning research?",
        "How do I use the authentication API?",
        "What was decided in the last project meeting?",
        "Compare the research papers with our technical implementation"
    ]

    for question in questions:
        result = agent_rag.query(question)
        print(f"\n💡 Final Answer:\n{result['answer']}\n")
        print("="*80)

    # Example: Manual multi-collection search
    print("\n\n🔍 MANUAL MULTI-COLLECTION SEARCH")
    print("="*80)

    answer = agent_rag.multi_collection_search(
        "What are the main topics discussed?",
        collection_names=["research_papers", "meeting_notes"],
        k=3
    )
    print(f"\n💡 Synthesized Answer:\n{answer}")


if __name__ == "__main__":
    # Note: Make sure you have document collections created first
    # You can use rag_chromadb_example.py to create collections

    main()
