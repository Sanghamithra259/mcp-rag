from mcp.server.fastmcp import FastMCP
from processor import RAGEngine

# Initialize FastMCP server
mcp = FastMCP("mcp-retrieval-engine")

# Initialize RAG Engine
# We initialize it at module level so it persists across tool calls
rag_engine = RAGEngine()

@mcp.tool()
def index_data() -> str:
    """
    Triggers the ingestion of documents from the ./data folder into the vector store.
    Returns a status message.
    """
    try:
        rag_engine.ingest_data()
        return "Data ingestion completed successfully."
    except Exception as e:
        return f"Error during data ingestion: {str(e)}"

@mcp.tool()
def search_notes(query: str) -> str:
    """
    Searches the ingested notes for the given query.
    Returns the top 3 most relevant snippets.
    
    Args:
        query: The search query string.
    """
    try:
        results = rag_engine.query(query)
        if not results:
            return "No relevant notes found."
        
        # Format results for better readability
        formatted_results = "\n\n---\n\n".join(results)
        return f"Found relevant notes:\n\n{formatted_results}"
    except Exception as e:
        return f"Error during search: {str(e)}"

if __name__ == "__main__":
    mcp.run()
