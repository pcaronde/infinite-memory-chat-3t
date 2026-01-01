# Infinite Memory Chat

A proof-of-concept chatbot with virtually unlimited conversation memory using OpenAI's Responses API and pluggable vector backends (OpenAI Vector Store or MongoDB Vector Search).

## The Problem

LLMs have a fixed context window. Once you exceed it, older messages are lost. Traditional solutions include:
- Truncating old messages (loses information)
- Summarizing history (loses detail)
- Manual RAG implementations (complex to build)

## The Solution

This project implements a **self-populating RAG system** that automatically archives conversation history to a vector store, creating a seamless long-term memory.

```
┌─────────────────────────────────────────────────────────────┐
│  LEVEL 1: Active Conversation (20 messages)                 │
│  Full detail, immediate context                             │
└─────────────────────┬───────────────────────────────────────┘
                      │ at 20 messages, archive oldest 10
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LEVEL 2: Vector Store Archive (max 100 files)              │
│  Searchable via file_search tool                            │
│  Each file = 10 messages as JSON                            │
└─────────────────────┬───────────────────────────────────────┘
                      │ at 100 files, merge 50 → 1
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LEVEL 3: Consolidated Archives                             │
│  50 files merged into 1 (all data preserved)                │
│  Max ~500 MB per file                                       │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

1. **Normal Chat**: Messages accumulate in the active conversation window
2. **Auto-Archive**: When the window reaches 20 messages, the oldest 10 are saved to the vector store as a JSON file
3. **Semantic Search**: The LLM uses `file_search` to retrieve relevant history when needed
4. **Auto-Consolidate**: When archive files reach 100, the oldest 50 are merged into one file
5. **Size Management**: Large consolidated files (>400 MB) are preserved separately

## Theoretical Limits

| Constraint | Limit | Implication |
|------------|-------|-------------|
| Files per vector store | 10,000 | With consolidation: effectively unlimited |
| File size | 512 MB | ~1 million messages per file |
| Theoretical maximum | ~25 GB | ~50 million messages |

At 100 messages/day, that's **~1,370 years** of conversation.

## Vector Backends

Choose between two vector storage backends based on your scale and requirements:

### 🤖 OpenAI Vector Store (Default)
**Best for**: Small to medium scale (<10K messages/month)
- **Cost**: $3-33/month
- **Setup**: Zero configuration required
- **Pros**: Immediate setup, no infrastructure
- **Cons**: Vendor lock-in, expensive at scale

### 🍃 MongoDB Vector Search
**Best for**: Large scale (>50K messages/month)
- **Cost**: $300-700/month (Atlas) or self-hosted
- **Setup**: Requires MongoDB Atlas or local MongoDB
- **Pros**: Data ownership, cost-effective at scale, horizontal scaling
- **Cons**: Requires setup and infrastructure management

## Key Features

- **Pluggable backends**: Switch between OpenAI and MongoDB
- **Zero information loss**: All messages are preserved, never summarized
- **Automatic**: No manual intervention needed
- **Cost efficient**: Small active context window = fewer tokens per request
- **Semantic retrieval**: Finds relevant history by meaning, not just recency
- **Production ready**: Comprehensive test coverage (27/27 tests passing)

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-key"

# Run
python3 infinite_memory_chat.py
```

Or simply:
```bash
chmod +x run.sh
export OPENAI_API_KEY="your-key"
./run.sh
```

## Usage Example

```
You: My name is Alice and I'm working on Project Neptune.
Assistant: Nice to meet you, Alice! I will remember that...

[... 500 messages later, archival happens automatically ...]

📁 Archiving 10 messages to vector store...
✅ Archive #1 saved and indexed

[... conversation continues ...]

You: What project am I working on?
Assistant: You are working on Project Neptune, Alice! 
         (Retrieved from archived conversation history)
```

The assistant found your name and project from archived history, even though that information left the active context window hundreds of messages ago.

## Configuration

### Quick Start (OpenAI Backend)

1. **Create a `.env` file:**
```bash
cp env.example .env
```

2. **Add your OpenAI API key:**
```bash
VECTOR_BACKEND=openai
OPENAI_API_KEY=your_openai_api_key_here
```

3. **Run the application:**
```bash
./run.sh
```

### MongoDB Backend Setup

1. **Set up MongoDB** (Atlas or local)
2. **Configure `.env` file:**
```bash
VECTOR_BACKEND=mongodb
OPENAI_API_KEY=your_openai_api_key_here
MONGODB_CONNECTION_STRING=mongodb+srv://user:pass@cluster.mongodb.net/
MONGODB_DATABASE=infinite_memory_chat
```

3. **Create vector search index** in MongoDB Atlas:
   - Index name: `vector_index`
   - Field: `content_embedding` (vector, 1536 dimensions, cosine similarity)
   - Filter fields: `session_id`, `archive_id`, `type`

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_BACKEND` | `openai` | Backend: `openai` or `mongodb` |
| `OPENAI_API_KEY` | Required | OpenAI API key (chat + embeddings) |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat model to use |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `MONGODB_CONNECTION_STRING` | - | MongoDB connection (if backend=mongodb) |
| `MONGODB_DATABASE` | `infinite_memory_chat` | MongoDB database name |

See `env.example` for a complete configuration template with cost optimization guide.

## How Consolidation Works

When consolidation is triggered, older archives cascade:

**Cycle 1:** 50 files (500 messages) → 1 file  
**Cycle 2:** 1 consolidated (500 msg) + 49 files (490 msg) → 1 file (990 msg)  
**Cycle 3:** 1 consolidated (990 msg) + 49 files (490 msg) → 1 file (1480 msg)  

The consolidated file keeps growing until it approaches the 500 MB limit, then a new consolidated file is started.

## Future Improvements

- [ ] Session persistence (save/load vector store ID)
- [ ] Message overlap for better context continuity
- [ ] Logging of vector store searches
- [ ] Multiple users/sessions
- [ ] Web interface

## License

CC-BY Anders Bjarby

