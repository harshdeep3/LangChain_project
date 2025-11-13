"""
Agentic RAG with LangGraph
Uses LangGraph for explicit workflow control with conditional routing
This gives you full control over the agent's decision-making process
"""

from typing import TypedDict, List, Annotated, Literal
from operator import add
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# Try to import langgraph, provide fallback message
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️  LangGraph not installed. Install with: pip install langgraph")
    print("This is an optional dependency for advanced workflow control\n")


class AgentState(TypedDict):
    """State of the agentic RAG workflow"""
    question: str
    current_step: str
    rewritten_query: str
    selected_collections: List[str]
    search_results: Annotated[List[str], add]
    final_answer: str
    confidence_score: float
    needs_more_info: bool
    iteration: int


class QueryPlan(BaseModel):
    """Plan for answering a query"""
    collections_to_search: List[str] = Field(description="Which collections to search")
    search_strategy: Literal["basic", "filtered", "confident"] = Field(
        description="Which search strategy to use"
    )
    should_rewrite: bool = Field(description="Whether to rewrite the query first")
    reasoning: str = Field(description="Reasoning for this plan")


class LangGraphAgenticRAG:
    """
    Agentic RAG using LangGraph for explicit workflow control

    Workflow:
    1. Analyze question
    2. Plan which collections to search
    3. Optionally rewrite query
    4. Execute searches
    5. Evaluate results
    6. Synthesize answer or iterate
    """

    def __init__(self, llm_model: str = "llama3.1"):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph is required. Install with: pip install langgraph")

        self.llm = ChatOllama(model=llm_model, temperature=0)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
        self.collections = {}
        self.workflow = None

    def add_collection(self, name: str, description: str, persist_directory: str):
        """Add a document collection"""
        vectorstore = Chroma(
            collection_name=name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

        self.collections[name] = {
            "description": description,
            "vectorstore": vectorstore
        }
        print(f"✅ Added collection '{name}': {description}")

    # Workflow nodes
    def analyze_question(self, state: AgentState) -> AgentState:
        """Analyze the question and create a search plan"""
        print("\n🔍 Step 1: Analyzing question...")

        collections_desc = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.collections.items()
        ])

        prompt = f"""Analyze this question and decide which document collections to search.

Available collections:
{collections_desc}

Question: {state['question']}

Provide your analysis:
1. Which collection(s) should be searched? (list names)
2. Should the query be rewritten for better results? (yes/no)
3. What search strategy: basic, filtered, or confident?
4. Reasoning for your choices

Format your response as:
COLLECTIONS: collection1, collection2
REWRITE: yes/no
STRATEGY: basic/filtered/confident
REASONING: your reasoning here"""

        response = self.llm.invoke(prompt)
        analysis = response.content

        # Parse response (simple parsing)
        selected_collections = []
        should_rewrite = False

        for line in analysis.split('\n'):
            if line.startswith('COLLECTIONS:'):
                colls = line.replace('COLLECTIONS:', '').strip()
                selected_collections = [c.strip() for c in colls.split(',')]
            elif line.startswith('REWRITE:'):
                should_rewrite = 'yes' in line.lower()

        state["selected_collections"] = selected_collections
        state["current_step"] = "rewrite" if should_rewrite else "search"
        state["iteration"] = 0

        print(f"   Selected collections: {selected_collections}")
        print(f"   Rewrite query: {should_rewrite}")

        return state

    def rewrite_query(self, state: AgentState) -> AgentState:
        """Rewrite query for better retrieval"""
        print("\n✍️  Step 2: Rewriting query...")

        prompt = f"""Rewrite this query to be more effective for document search.
Make it more specific and include relevant keywords.

Original: {state['question']}

Rewritten query (one line):"""

        response = self.llm.invoke(prompt)
        rewritten = response.content.strip()

        state["rewritten_query"] = rewritten
        state["current_step"] = "search"

        print(f"   Rewritten: {rewritten}")

        return state

    def search_collections(self, state: AgentState) -> AgentState:
        """Search the selected collections"""
        print("\n🔎 Step 3: Searching collections...")

        query = state.get("rewritten_query", state["question"])
        results = []

        for coll_name in state["selected_collections"]:
            if coll_name not in self.collections:
                print(f"   ⚠️  Collection '{coll_name}' not found")
                continue

            print(f"   Searching {coll_name}...")
            vectorstore = self.collections[coll_name]["vectorstore"]
            docs = vectorstore.similarity_search(query, k=4)

            if docs:
                formatted = f"=== Results from {coll_name} ===\n"
                for i, doc in enumerate(docs, 1):
                    formatted += f"\n[{i}] {doc.metadata.get('source', 'Unknown')}\n"
                    formatted += f"{doc.page_content}\n"
                results.append(formatted)
                print(f"   Found {len(docs)} documents")
            else:
                print(f"   No results in {coll_name}")

        state["search_results"] = results
        state["current_step"] = "evaluate"

        return state

    def evaluate_results(self, state: AgentState) -> AgentState:
        """Evaluate if we have enough information"""
        print("\n📊 Step 4: Evaluating results...")

        if not state["search_results"]:
            print("   No results found")
            state["needs_more_info"] = False
            state["current_step"] = "synthesize"
            state["confidence_score"] = 0.0
            return state

        # Ask LLM if results are sufficient
        combined_results = "\n\n".join(state["search_results"])

        eval_prompt = f"""Evaluate if these search results contain enough information to answer the question.

Question: {state['question']}

Search Results:
{combined_results}

Can this question be answered with these results? Respond with:
SUFFICIENT: yes/no
CONFIDENCE: 0.0-1.0
REASONING: brief explanation"""

        response = self.llm.invoke(eval_prompt)
        evaluation = response.content

        # Parse evaluation
        is_sufficient = "yes" in evaluation.split("SUFFICIENT:")[1].split("\n")[0].lower()
        confidence = 0.7  # Default

        try:
            conf_line = [l for l in evaluation.split('\n') if 'CONFIDENCE:' in l][0]
            confidence = float(conf_line.split('CONFIDENCE:')[1].strip().split()[0])
        except:
            pass

        state["confidence_score"] = confidence
        state["needs_more_info"] = not is_sufficient and state["iteration"] < 2
        state["current_step"] = "search" if state["needs_more_info"] else "synthesize"
        state["iteration"] += 1

        print(f"   Sufficient: {is_sufficient}")
        print(f"   Confidence: {confidence:.2f}")

        return state

    def synthesize_answer(self, state: AgentState) -> AgentState:
        """Synthesize final answer from search results"""
        print("\n💡 Step 5: Synthesizing answer...")

        if not state["search_results"]:
            state["final_answer"] = "I couldn't find relevant information to answer this question."
            return state

        combined_results = "\n\n".join(state["search_results"])

        synthesis_prompt = f"""Based on the search results below, provide a comprehensive answer to the question.
Cite sources for each piece of information.

Question: {state['question']}

Search Results:
{combined_results}

Comprehensive Answer (with citations):"""

        response = self.llm.invoke(synthesis_prompt)
        state["final_answer"] = response.content
        state["current_step"] = "end"

        return state

    def should_continue(self, state: AgentState) -> Literal["search", "synthesize", "end"]:
        """Router: decide next step"""
        if state["needs_more_info"]:
            return "search"
        elif state["current_step"] == "synthesize":
            return "synthesize"
        else:
            return "end"

    def build_workflow(self):
        """Build the LangGraph workflow"""
        print("\n🔧 Building workflow...")

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze", self.analyze_question)
        workflow.add_node("rewrite", self.rewrite_query)
        workflow.add_node("search", self.search_collections)
        workflow.add_node("evaluate", self.evaluate_results)
        workflow.add_node("synthesize", self.synthesize_answer)

        # Set entry point
        workflow.set_entry_point("analyze")

        # Add edges
        workflow.add_conditional_edges(
            "analyze",
            lambda state: state["current_step"],
            {
                "rewrite": "rewrite",
                "search": "search"
            }
        )

        workflow.add_edge("rewrite", "search")
        workflow.add_edge("search", "evaluate")

        workflow.add_conditional_edges(
            "evaluate",
            self.should_continue,
            {
                "search": "search",      # Need more info
                "synthesize": "synthesize",  # Have enough info
                "end": END
            }
        )

        workflow.add_edge("synthesize", END)

        self.workflow = workflow.compile()
        print("✅ Workflow built")

    def query(self, question: str) -> dict:
        """Execute the workflow for a question"""
        if not self.workflow:
            self.build_workflow()

        print(f"\n{'='*80}")
        print(f"❓ Question: {question}")
        print(f"{'='*80}")

        initial_state = {
            "question": question,
            "current_step": "analyze",
            "rewritten_query": "",
            "selected_collections": [],
            "search_results": [],
            "final_answer": "",
            "confidence_score": 0.0,
            "needs_more_info": False,
            "iteration": 0
        }

        # Run workflow
        final_state = self.workflow.invoke(initial_state)

        return {
            "question": question,
            "answer": final_state["final_answer"],
            "confidence": final_state["confidence_score"],
            "collections_used": final_state["selected_collections"],
            "iterations": final_state["iteration"]
        }


def main():
    """Example usage"""
    if not LANGGRAPH_AVAILABLE:
        print("Please install langgraph: pip install langgraph")
        return

    print("🤖 LANGGRAPH AGENTIC RAG SYSTEM\n")

    agent_rag = LangGraphAgenticRAG(llm_model="llama3.1")

    # Add collections
    agent_rag.add_collection(
        name="research",
        description="Research papers on AI and machine learning",
        persist_directory="./chroma_research"
    )

    agent_rag.add_collection(
        name="docs",
        description="Technical documentation and guides",
        persist_directory="./chroma_docs"
    )

    # Build workflow
    agent_rag.build_workflow()

    # Example queries
    questions = [
        "What are the latest machine learning techniques?",
        "How do I implement authentication in the API?",
    ]

    for question in questions:
        result = agent_rag.query(question)

        print(f"\n{'='*80}")
        print(f"💡 FINAL ANSWER")
        print(f"{'='*80}")
        print(f"\n{result['answer']}\n")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Collections used: {result['collections_used']}")
        print(f"Iterations: {result['iterations']}")
        print("="*80)


if __name__ == "__main__":
    main()
