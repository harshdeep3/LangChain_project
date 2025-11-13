"""
Advanced Agentic RAG System
Features:
- Multiple specialized tools
- Document metadata filtering
- Query rewriting and expansion
- Self-reflection and validation
- Memory of previous searches
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool, StructuredTool
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    """Input schema for search tools"""
    query: str = Field(description="The search query")
    k: int = Field(default=4, description="Number of documents to retrieve")


class MetadataSearchInput(BaseModel):
    """Input schema for metadata filtered search"""
    query: str = Field(description="The search query")
    filename: Optional[str] = Field(default=None, description="Filter by filename")
    date_after: Optional[str] = Field(default=None, description="Filter documents after this date (YYYY-MM-DD)")
    k: int = Field(default=4, description="Number of documents to retrieve")


class AdvancedDocumentCollection:
    """Enhanced document collection with metadata support"""

    def __init__(self, name: str, description: str, vectorstore: Chroma, metadata_fields: List[str] = None):
        self.name = name
        self.description = description
        self.vectorstore = vectorstore
        self.metadata_fields = metadata_fields or ["source", "filename", "page"]
        self.search_history = []

    def search(self, query: str, k: int = 4) -> str:
        """Basic search"""
        docs = self.vectorstore.similarity_search(query, k=k)
        self.search_history.append({"query": query, "results": len(docs), "timestamp": datetime.now()})

        if not docs:
            return f"No relevant information found in {self.name}"

        return self._format_results(docs)

    def search_with_metadata(
        self,
        query: str,
        k: int = 4,
        filename: Optional[str] = None,
        date_after: Optional[str] = None
    ) -> str:
        """Search with metadata filtering"""

        # Build metadata filter
        metadata_filter = {}
        if filename:
            metadata_filter["filename"] = filename
        if date_after:
            metadata_filter["date"] = {"$gte": date_after}

        if metadata_filter:
            docs = self.vectorstore.similarity_search(query, k=k, filter=metadata_filter)
        else:
            docs = self.vectorstore.similarity_search(query, k=k)

        self.search_history.append({
            "query": query,
            "filter": metadata_filter,
            "results": len(docs),
            "timestamp": datetime.now()
        })

        if not docs:
            return f"No documents matching criteria in {self.name}"

        return self._format_results(docs, show_metadata=True)

    def search_with_score(self, query: str, k: int = 4, score_threshold: float = 1.5) -> str:
        """Search with score filtering"""
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)

        # Filter by score
        filtered_docs = [(doc, score) for doc, score in docs_with_scores if score < score_threshold]

        if not filtered_docs:
            return f"No high-confidence results found in {self.name} (threshold: {score_threshold})"

        results = [f"=== High-Confidence Results from {self.name} ===\n"]
        for i, (doc, score) in enumerate(filtered_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            results.append(f"\n[{i}] Confidence Score: {score:.4f}")
            results.append(f"Source: {source}")
            results.append(f"Content: {doc.page_content}\n")

        return "\n".join(results)

    def get_document_list(self) -> str:
        """Get list of available documents in this collection"""
        try:
            # Get sample of documents to extract unique sources
            sample = self.vectorstore._collection.get(limit=100, include=["metadatas"])
            sources = set()
            for metadata in sample['metadatas']:
                source = metadata.get('source') or metadata.get('filename', 'Unknown')
                sources.add(source)

            results = [f"=== Documents in {self.name} ==="]
            results.append(f"Total unique sources: {len(sources)}\n")
            for source in sorted(sources):
                results.append(f"  - {source}")

            return "\n".join(results)
        except Exception as e:
            return f"Error listing documents: {e}"

    def _format_results(self, docs: List, show_metadata: bool = False) -> str:
        """Format search results"""
        results = [f"=== Results from {self.name} ===\n"]

        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            results.append(f"\n[{i}] Source: {source}")

            if show_metadata:
                # Show all metadata
                for key, value in doc.metadata.items():
                    if key != 'source':
                        results.append(f"    {key}: {value}")

            results.append(f"Content: {doc.page_content}\n")
            results.append("-" * 80)

        return "\n".join(results)


class AdvancedAgenticRAG:
    """
    Advanced agentic RAG with multiple capabilities:
    - Query rewriting
    - Multi-step reasoning
    - Self-validation
    - Memory of conversation
    """

    def __init__(self, llm_model: str = "llama3.1"):
        self.llm = ChatOllama(model=llm_model, temperature=0)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
        self.collections: Dict[str, AdvancedDocumentCollection] = {}
        self.agent_executor = None
        self.conversation_history = []

    def add_collection(
        self,
        name: str,
        description: str,
        persist_directory: str,
        collection_name: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None
    ):
        """Add a document collection"""
        vectorstore = Chroma(
            collection_name=collection_name or name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

        collection = AdvancedDocumentCollection(
            name, description, vectorstore, metadata_fields
        )
        self.collections[name] = collection

        print(f"✅ Added collection '{name}'")
        print(f"   Documents: {vectorstore._collection.count()}")

    def _create_tools(self) -> List[Tool]:
        """Create all tools for the agent"""
        tools = []

        for name, collection in self.collections.items():
            # Tool 1: Basic search
            basic_search = StructuredTool.from_function(
                func=collection.search,
                name=f"search_{name}",
                description=f"Search {collection.description}. Use this for general queries about {name}.",
                args_schema=SearchInput
            )
            tools.append(basic_search)

            # Tool 2: Metadata filtered search
            metadata_search = StructuredTool.from_function(
                func=collection.search_with_metadata,
                name=f"search_{name}_filtered",
                description=f"Search {name} with metadata filters (filename, date). Use when you need specific documents.",
                args_schema=MetadataSearchInput
            )
            tools.append(metadata_search)

            # Tool 3: High-confidence search
            score_search = Tool(
                name=f"search_{name}_confident",
                func=lambda q, coll=collection: coll.search_with_score(q, k=4, score_threshold=1.2),
                description=f"Search {name} and return only high-confidence results. Use when accuracy is critical."
            )
            tools.append(score_search)

            # Tool 4: List documents
            list_docs = Tool(
                name=f"list_{name}_documents",
                func=lambda coll=collection: coll.get_document_list(),
                description=f"List all documents available in {name}. Use this to see what's available before searching."
            )
            tools.append(list_docs)

        # Tool 5: Query rewriter
        def rewrite_query(query: str) -> str:
            """Rewrite query for better retrieval"""
            rewrite_prompt = f"""Rewrite the following query to be more effective for document search.
Make it more specific and include synonyms or related terms.

Original query: {query}

Rewritten query (one line only):"""

            response = self.llm.invoke(rewrite_prompt)
            rewritten = response.content.strip()
            return f"Rewritten query: '{rewritten}'\nUse this for your searches."

        query_rewriter = Tool(
            name="rewrite_query",
            func=rewrite_query,
            description="Rewrite a query to make it more effective for searching. Use this if initial searches don't yield good results."
        )
        tools.append(query_rewriter)

        # Tool 6: Synthesizer
        def synthesize_results(results: str) -> str:
            """Synthesize multiple search results"""
            synthesis_prompt = f"""Synthesize the following search results into a coherent, comprehensive answer.
Cite sources for each piece of information.

Search Results:
{results}

Synthesized Answer:"""

            response = self.llm.invoke(synthesis_prompt)
            return response.content

        synthesizer = Tool(
            name="synthesize_information",
            func=synthesize_results,
            description="Synthesize multiple search results into a coherent answer. Use this after gathering information from multiple sources."
        )
        tools.append(synthesizer)

        return tools

    def create_agent(self):
        """Create the advanced agent"""
        if not self.collections:
            raise ValueError("No collections added")

        tools = self._create_tools()

        # Advanced prompt with reasoning
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an advanced research assistant with access to multiple document collections and specialized tools.

Your capabilities:
1. Search multiple document collections
2. Filter searches by metadata (filename, date)
3. Get high-confidence results only
4. List available documents
5. Rewrite queries for better results
6. Synthesize information from multiple sources

STRATEGY:
1. For complex questions, break them into sub-questions
2. Search relevant collections for each sub-question
3. If results are poor, try rewriting the query
4. Use filtered searches when you need specific documents
5. Use confident search when accuracy is critical
6. Synthesize information from multiple sources
7. Always cite sources

Available collections: {collections}

Think step-by-step and explain your reasoning."""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        collections_desc = "\n".join([f"- {name}: {coll.description}" for name, coll in self.collections.items()])
        prompt = prompt.partial(collections=collections_desc)

        # Create agent - Note: Using react agent as create_openai_tools_agent requires OpenAI-compatible API
        from langchain.agents import create_react_agent

        react_prompt = PromptTemplate(
            template="""You are an advanced research assistant with access to multiple document collections.

Available collections:
{collections}

You have access to these tools:
{tools}

Tool Names: {tool_names}

STRATEGY:
1. Think about which collections to search
2. Use appropriate search tools (basic, filtered, or confident)
3. If results are poor, rewrite the query
4. Synthesize information from multiple searches
5. Always cite sources

Question: {input}

Thought Process:
{agent_scratchpad}""",
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "collections": collections_desc,
                "tools": "\n".join([f"- {t.name}: {t.description}" for t in tools]),
                "tool_names": ", ".join([t.name for t in tools])
            }
        )

        agent = create_react_agent(self.llm, tools, react_prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        print(f"\n🤖 Advanced agent created with {len(tools)} tools")

    def query(self, question: str, use_history: bool = False) -> Dict[str, Any]:
        """Query with optional conversation history"""
        if not self.agent_executor:
            raise ValueError("Agent not created")

        print(f"\n{'='*80}")
        print(f"❓ Question: {question}")
        print(f"{'='*80}\n")

        # Prepare input
        agent_input = {"input": question}
        if use_history and self.conversation_history:
            agent_input["chat_history"] = self.conversation_history

        # Execute
        result = self.agent_executor.invoke(agent_input)

        # Update history
        if use_history:
            self.conversation_history.append(HumanMessage(content=question))
            self.conversation_history.append(AIMessage(content=result["output"]))

        return {
            "question": question,
            "answer": result["output"],
            "intermediate_steps": result.get("intermediate_steps", [])
        }

    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about searches"""
        stats = {}
        for name, collection in self.collections.items():
            stats[name] = {
                "total_searches": len(collection.search_history),
                "recent_searches": collection.search_history[-5:] if collection.search_history else []
            }
        return stats


def main():
    """Example usage"""
    print("🤖 ADVANCED AGENTIC RAG SYSTEM\n")

    agent_rag = AdvancedAgenticRAG(llm_model="llama3.1")

    # Add collections
    agent_rag.add_collection(
        name="research",
        description="Research papers on AI and ML",
        persist_directory="./chroma_research"
    )

    agent_rag.add_collection(
        name="docs",
        description="Technical documentation",
        persist_directory="./chroma_docs"
    )

    # Create agent
    agent_rag.create_agent()

    # Example queries
    questions = [
        "What documents are available about machine learning?",
        "Search for information about neural networks in research papers",
        "Find technical documentation about API authentication with high confidence",
        "Compare findings from research papers with our technical implementation"
    ]

    for question in questions:
        result = agent_rag.query(question, use_history=True)
        print(f"\n💡 Answer:\n{result['answer']}\n")
        print("="*80)

    # Show stats
    print("\n📊 Search Statistics:")
    print(agent_rag.get_search_stats())


if __name__ == "__main__":
    main()
