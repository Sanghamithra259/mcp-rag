# mcp-retrieval-engine

A production-ready **Model Context Protocol (MCP)** server implementing a local **RAG (Retrieval-Augmented Generation)** pipeline. This project enables AI agents to securely search and retrieve information from local document repositories with high semantic accuracy.

---

## Overview

Unlike standard chatbots, this system bridges the "context gap" by allowing LLMs to dynamically query private files. By using the MCP standard, the AI can decide when it needs to "look up" information in your local database before answering.

### Key Features

* **Semantic Search:** Uses `all-MiniLM-L6-v2` embeddings to find information based on meaning, not just keywords.
* **Agentic Integration:** Fully compatible with MCP-native clients like **Claude Desktop** and **Cursor**.
* **Local-First Privacy:** All document indexing and vector searches are performed on-device via **ChromaDB**.
* **Multi-Format Support:** Native parsing for **PDF, TXT, and Markdown** files.

---

##  Architecture & Algorithm

The system operates in two distinct phases:

1. **Ingestion:** Documents are loaded, split into 500-character chunks using **Recursive Character Splitting**, and converted into 384-dimensional vectors.
2. **Retrieval:** When a tool is called, the system performs a **Cosine Similarity** search between the user's query vector and the document vectors stored in the local database.

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| **Language** | Python 3.10+ |
| **Protocol** | FastMCP (Model Context Protocol SDK) |
| **Orchestration** | LangChain |
| **Vector DB** | ChromaDB (Persistent) |
| **Embeddings** | Sentence-Transformers |

---

## Project Structure

```text
mcp-retrieval-engine/
├── data/           # Raw document storage (.pdf, .txt, .md)
├── chroma_db/      # Persistent vector database
├── src/
│   ├── main.py       # MCP Server entry point and tool definitions
│   └── processor.py  # RAG Engine: Chunking and Retrieval logic
└── pyproject.toml  # Dependency management via uv

```

---

##  Getting Started

### 1. Installation

Ensure you have `uv` installed, then sync the environment:

```bash
uv sync

```

### 2. Ingesting Data

Place your documents in the `/data` directory and run the indexing tool through the MCP Inspector:

```bash
mcp inspector src/main.py

```

### 3. Claude Desktop Integration

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "local-retriever": {
      "command": "python",
      "args": ["/absolute/path/to/src/main.py"]
    }
  }
}

```

---

## Impact

This project demonstrates the ability to build **AI Infrastructure** that solves data privacy and context window limitations in enterprise and academic workflows.
