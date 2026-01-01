# MongoDB Vector Backend Integration Implementation Plan

## Overview
Implement MongoDB Vector Search as an optional backend using an abstract backend pattern while maintaining OpenAI Vector Store as the default, enabling cost-effective scaling and vendor diversification without breaking existing functionality.

## Current State Analysis
The current `InfiniteMemoryChat` class (lines 69-342) has tight coupling with OpenAI's vector store API through direct client calls. The system implements a 3-level memory hierarchy:
- **Level 1**: Active conversation (20 messages max)
- **Level 2**: Vector store archive (100 JSON files max)
- **Level 3**: Consolidated archives (50 files → 1 file, 500MB max)

## Desired End State
A pluggable backend system where users can choose between OpenAI Vector Store and MongoDB Vector Search at runtime, with identical functionality and seamless migration capabilities.

### Key Discoveries:
- **Direct API coupling** (lines 82-87, 124-130, 246-262): OpenAI client calls scattered throughout main class
- **File-based archival pattern** (lines 117-119): Creates temporary JSON files for vector store upload
- **Session-based naming** (lines 75, 83): Uses timestamp session IDs for vector store identification
- **Complex consolidation logic** (lines 166-280): Merges files to work within OpenAI's 10,000 file limit
- **Search integration** (lines 296-304): Uses OpenAI's `file_search` tool in chat responses

## What We're NOT Doing
- Changing the 3-level memory hierarchy architecture
- Modifying the core conversation flow or user interface
- Implementing real-time backend switching (requires restart)
- Building a GUI for backend configuration
- Supporting multiple backends simultaneously in a single session

## Implementation Approach
Use the Abstract Factory pattern to create interchangeable vector backends while maintaining the existing `InfiniteMemoryChat` interface. This allows for gradual migration and testing without disrupting current functionality.

## Phase 1: Backend Abstraction Layer

### Overview
Extract vector storage operations into an abstract interface and refactor the existing OpenAI implementation to use this interface.

### Changes Required:

#### 1. Vector Backend Interface
**File**: `vector_backends/base.py`
**Changes**: Create abstract base class defining the vector backend interface

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

class VectorBackend(ABC):
    """Abstract base class for vector storage backends."""
    
    @abstractmethod
    def setup_store(self, session_id: str) -> str:
        """Create a new vector store for the session."""
        pass
    
    @abstractmethod
    def archive_messages(self, messages: List[Dict], archive_id: int, session_id: str) -> bool:
        """Archive messages to vector store."""
        pass
    
    @abstractmethod
    def search_archives(self, query_text: str, limit: int = 10) -> List[Dict]:
        """Search archived messages for relevant content."""
        pass
    
    @abstractmethod
    def consolidate_archives(self, archive_ids: List[str]) -> str:
        """Consolidate multiple archives into a single unit."""
        pass
    
    @abstractmethod
    def get_archive_count(self) -> int:
        """Get the number of archived message groups."""
        pass
    
    @abstractmethod
    def cleanup_archives(self, archive_ids: List[str]) -> bool:
        """Remove specified archives from storage."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get backend-specific status information."""
        pass
```

#### 2. OpenAI Backend Implementation
**File**: `vector_backends/openai_backend.py`
**Changes**: Extract existing OpenAI logic into backend class

```python
import os
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from openai import OpenAI

from .base import VectorBackend

class OpenAIBackend(VectorBackend):
    """OpenAI Vector Store backend implementation."""
    
    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client or OpenAI()
        self.vector_store_id = None
        self.archived_files = []
        self.consolidation_count = 0
        
    def setup_store(self, session_id: str) -> str:
        """Create a new vector store for this session."""
        print("📦 Creating OpenAI vector store...")
        vector_store = self.client.vector_stores.create(
            name=f"chat_memory_{session_id}"
        )
        self.vector_store_id = vector_store.id
        print(f"✅ Vector store created: {self.vector_store_id}")
        return self.vector_store_id
    
    def archive_messages(self, messages: List[Dict], archive_id: int, session_id: str) -> bool:
        """Archive messages to OpenAI vector store."""
        # Create archive data structure
        archive_data = {
            "archive_id": archive_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "message_range": f"messages {(archive_id-1)*10 + 1}-{archive_id*10}",
            "messages": messages
        }
        
        # Create summary text for better embeddings
        summary_text = f"Conversation archive #{archive_id}\n"
        summary_text += f"Timestamp: {archive_data['timestamp']}\n"
        summary_text += f"Content:\n"
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            summary_text += f"\n{role}: {msg['content']}\n"
        
        # Save to temporary file
        filename = f"/tmp/archive_{session_id}_{archive_id}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"summary": summary_text, "data": archive_data}, f, ensure_ascii=False, indent=2)
        
        print(f"📁 Archiving {len(messages)} messages to OpenAI vector store...")
        
        # Upload to vector store
        with open(filename, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="assistants")
        
        self.client.vector_stores.files.create(
            vector_store_id=self.vector_store_id,
            file_id=file_obj.id
        )
        
        # Track file for consolidation
        self.archived_files.append({
            "file_id": file_obj.id,
            "archive_id": archive_id,
            "filename": filename
        })
        
        # Wait for indexing
        success = self._wait_for_indexing(file_obj.id)
        if success:
            print(f"✅ Archive #{archive_id} saved and indexed")
        
        return success
    
    def search_archives(self, query_text: str, limit: int = 10) -> List[Dict]:
        """OpenAI handles search through the file_search tool in responses."""
        # This is handled by the chat method using OpenAI's file_search tool
        # Return empty list as search is integrated into chat responses
        return []
    
    # ... (implement remaining abstract methods)
```

#### 3. Refactor Main Chat Class
**File**: `infinite_memory_chat.py`
**Changes**: Modify to use backend abstraction

```python
from vector_backends.base import VectorBackend
from vector_backends.openai_backend import OpenAIBackend

class InfiniteMemoryChat:
    def __init__(self, backend: Optional[VectorBackend] = None):
        self.client = OpenAI()  # Still needed for chat responses
        self.backend = backend or OpenAIBackend(self.client)
        self.conversation_history = []
        self.archive_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def setup_vector_store(self):
        """Create a new vector store for this session."""
        store_id = self.backend.setup_store(self.session_id)
        return store_id
    
    def archive_messages(self):
        """Archive the oldest messages using the selected backend."""
        if len(self.conversation_history) < MAX_MESSAGES:
            return
        
        messages_to_archive = self.conversation_history[:ARCHIVE_COUNT]
        self.conversation_history = self.conversation_history[ARCHIVE_COUNT:]
        
        self.archive_count += 1
        success = self.backend.archive_messages(
            messages_to_archive, 
            self.archive_count, 
            self.session_id
        )
        
        if success and self.backend.get_archive_count() >= MAX_ARCHIVE_FILES:
            print(f"\n🔄 Max archive files reached - consolidating...")
            # Backend handles consolidation logic
```

### Success Criteria:

#### Automated Verification:
- [ ] Python imports work: `python -c "from vector_backends.base import VectorBackend"`
- [ ] Tests pass: `python -m pytest tests/test_backends.py`
- [ ] Type checking passes: `mypy vector_backends/`

#### Manual Verification:
- [ ] Existing OpenAI functionality works unchanged
- [ ] Chat sessions can be created and messages archived
- [ ] No regressions in conversation flow

**Implementation Note**: After completing this phase and automated verification passes, pause for manual confirmation before proceeding to next phase.

---

## Phase 2: MongoDB Backend Implementation

### Overview
Implement the MongoDB vector backend with full feature parity to the OpenAI backend, including vector search, archival, and consolidation.

### Changes Required:

#### 1. MongoDB Backend Class
**File**: `vector_backends/mongodb_backend.py`
**Changes**: Complete MongoDB Vector Search implementation

```python
import os
import json
import struct
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from bson import Binary
from openai import OpenAI

from .base import VectorBackend

class MongoDBBackend(VectorBackend):
    """MongoDB Vector Search backend implementation."""
    
    def __init__(self, connection_string: str, database_name: str = "infinite_memory_chat", 
                 embedding_client: Optional[OpenAI] = None):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection_name = None
        self.collection: Optional[Collection] = None
        self.embedding_client = embedding_client or OpenAI()
        self.archive_count = 0
        
    def setup_store(self, session_id: str) -> str:
        """Create MongoDB collection and vector search index."""
        print("📦 Creating MongoDB collection and vector index...")
        
        self.collection_name = f"chat_memory_{session_id}"
        self.collection = self.db[self.collection_name]
        
        # Create vector search index
        index_definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "content_embedding",
                    "numDimensions": 1536,  # text-embedding-3-small dimensions
                    "similarity": "cosine"
                },
                {
                    "type": "filter",
                    "path": "session_id"
                },
                {
                    "type": "filter",
                    "path": "archive_id"
                }
            ]
        }
        
        # Note: Vector index creation is async in MongoDB Atlas
        # In production, this would need proper index management
        print(f"✅ MongoDB collection created: {self.collection_name}")
        return self.collection_name
    
    def archive_messages(self, messages: List[Dict], archive_id: int, session_id: str) -> bool:
        """Archive messages to MongoDB with vector embeddings."""
        print(f"📁 Archiving {len(messages)} messages to MongoDB...")
        
        # Create summary text for embedding
        summary_text = f"Conversation archive #{archive_id}\n"
        summary_text += f"Timestamp: {datetime.now().isoformat()}\n"
        summary_text += f"Content:\n"
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            summary_text += f"\n{role}: {msg['content']}\n"
        
        # Generate embedding using OpenAI
        embedding_response = self.embedding_client.embeddings.create(
            model="text-embedding-3-small",
            input=summary_text
        )
        embedding_vector = embedding_response.data[0].embedding
        
        # Convert to BinData for efficient storage
        embedding_bindata = self._vector_to_bindata(embedding_vector)
        
        # Create document
        archive_doc = {
            "_id": f"{session_id}_{archive_id}",
            "session_id": session_id,
            "archive_id": archive_id,
            "timestamp": datetime.now(),
            "messages": messages,
            "content_text": summary_text,
            "content_embedding": embedding_bindata,
            "message_count": len(messages),
            "type": "archive"
        }
        
        try:
            result = self.collection.insert_one(archive_doc)
            self.archive_count += 1
            print(f"✅ Archive #{archive_id} saved to MongoDB")
            return True
        except Exception as e:
            print(f"❌ Error archiving to MongoDB: {e}")
            return False
    
    def search_archives(self, query_text: str, limit: int = 10) -> List[Dict]:
        """Search archived messages using MongoDB vector search."""
        if not self.collection:
            return []
        
        # Generate query embedding
        embedding_response = self.embedding_client.embeddings.create(
            model="text-embedding-3-small",
            input=query_text
        )
        query_vector = embedding_response.data[0].embedding
        
        # MongoDB vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": f"{self.collection_name}_vector_index",
                    "path": "content_embedding",
                    "queryVector": query_vector,
                    "numCandidates": limit * 20,  # 20x limit for better recall
                    "limit": limit,
                    "filter": {"type": "archive"}
                }
            },
            {
                "$project": {
                    "messages": 1,
                    "archive_id": 1,
                    "timestamp": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        try:
            results = list(self.collection.aggregate(pipeline))
            return results
        except Exception as e:
            print(f"⚠️ Error searching MongoDB: {e}")
            return []
    
    def _vector_to_bindata(self, vector: List[float]) -> Binary:
        """Convert float vector to MongoDB BinData for efficient storage."""
        byte_data = struct.pack(f'{len(vector)}f', *vector)
        return Binary(byte_data, subtype=4)  # float32 subtype
    
    # ... (implement remaining abstract methods)
```

#### 2. Enhanced Configuration System
**File**: `config.py`
**Changes**: Add backend configuration management

```python
import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class BackendType(Enum):
    OPENAI = "openai"
    MONGODB = "mongodb"

@dataclass
class BackendConfig:
    backend_type: BackendType = BackendType.OPENAI
    
    # OpenAI specific
    openai_api_key: Optional[str] = None
    
    # MongoDB specific
    mongodb_connection_string: Optional[str] = None
    mongodb_database: str = "infinite_memory_chat"
    embedding_model: str = "text-embedding-3-small"
    
    @classmethod
    def from_env(cls) -> 'BackendConfig':
        """Load configuration from environment variables."""
        backend_type = BackendType(os.getenv('VECTOR_BACKEND', 'openai'))
        
        return cls(
            backend_type=backend_type,
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            mongodb_connection_string=os.getenv('MONGODB_CONNECTION_STRING'),
            mongodb_database=os.getenv('MONGODB_DATABASE', 'infinite_memory_chat'),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
        )

def create_backend(config: BackendConfig) -> VectorBackend:
    """Factory function to create the appropriate backend."""
    if config.backend_type == BackendType.OPENAI:
        from vector_backends.openai_backend import OpenAIBackend
        return OpenAIBackend()
    elif config.backend_type == BackendType.MONGODB:
        from vector_backends.mongodb_backend import MongoDBBackend
        return MongoDBBackend(
            connection_string=config.mongodb_connection_string,
            database_name=config.mongodb_database
        )
    else:
        raise ValueError(f"Unsupported backend type: {config.backend_type}")
```

### Success Criteria:

#### Automated Verification:
- [ ] MongoDB backend imports: `python -c "from vector_backends.mongodb_backend import MongoDBBackend"`
- [ ] Unit tests pass: `python -m pytest tests/test_mongodb_backend.py`
- [ ] Integration tests pass: `python -m pytest tests/test_backend_integration.py`

#### Manual Verification:
- [ ] MongoDB backend can archive and search messages
- [ ] Vector search returns relevant results
- [ ] Performance is acceptable for typical usage

**Implementation Note**: After completing this phase and automated verification passes, pause for manual confirmation before proceeding to next phase.

---

## Phase 3: Configuration & Runtime Selection

### Overview
Add configuration management and runtime backend selection with environment variable support and fallback logic.

### Changes Required:

#### 1. Enhanced Main Application
**File**: `infinite_memory_chat.py`
**Changes**: Add backend selection and configuration

```python
import os
from config import BackendConfig, create_backend

class InfiniteMemoryChat:
    def __init__(self, config: Optional[BackendConfig] = None):
        self.config = config or BackendConfig.from_env()
        self.client = OpenAI()  # Always needed for chat responses
        
        try:
            self.backend = create_backend(self.config)
            print(f"🔌 Using {self.config.backend_type.value} backend")
        except Exception as e:
            print(f"⚠️ Error initializing {self.config.backend_type.value} backend: {e}")
            print("🔄 Falling back to OpenAI backend...")
            self.backend = OpenAIBackend(self.client)
        
        self.conversation_history = []
        self.archive_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def chat(self, user_message: str) -> str:
        """Enhanced chat method with backend-aware search."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.conversation_history)
        
        # Use backend-specific search if available
        if hasattr(self.backend, 'search_archives') and self.archive_count > 0:
            # For MongoDB backend, integrate search results
            if self.config.backend_type == BackendType.MONGODB:
                search_results = self.backend.search_archives(user_message, limit=5)
                if search_results:
                    context = "\n\nRelevant conversation history:\n"
                    for result in search_results:
                        for msg in result.get('messages', []):
                            role = "User" if msg["role"] == "user" else "Assistant"
                            context += f"{role}: {msg['content']}\n"
                    messages[0]["content"] += context
        
        # Make API call (OpenAI handles file_search automatically for OpenAI backend)
        if self.config.backend_type == BackendType.OPENAI and hasattr(self.backend, 'vector_store_id') and self.archive_count > 0:
            response = self.client.responses.create(
                model=MODEL,
                input=messages,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [self.backend.vector_store_id]
                }]
            )
        else:
            response = self.client.responses.create(
                model=MODEL,
                input=messages
            )
        
        assistant_message = response.output_text
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        self.archive_messages()
        return assistant_message
```

#### 2. Environment Configuration
**File**: `.env.example`
**Changes**: Add configuration examples

```bash
# Vector Backend Configuration
VECTOR_BACKEND=openai  # or 'mongodb'

# OpenAI Configuration (always needed for chat and embeddings)
OPENAI_API_KEY=your_openai_api_key_here

# MongoDB Configuration (only needed if using mongodb backend)
MONGODB_CONNECTION_STRING=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DATABASE=infinite_memory_chat
EMBEDDING_MODEL=text-embedding-3-small

# Application Configuration
MODEL=gpt-4o-mini
MAX_MESSAGES=20
ARCHIVE_COUNT=10
MAX_ARCHIVE_FILES=100
```

#### 3. Updated Requirements
**File**: `requirements.txt`
**Changes**: Add MongoDB dependencies

```txt
openai
pymongo
python-dotenv
numpy  # for vector operations
```

### Success Criteria:

#### Automated Verification:
- [ ] Environment variables load correctly: `python -c "from config import BackendConfig; print(BackendConfig.from_env())"`
- [ ] Backend switching works: Test with both `VECTOR_BACKEND=openai` and `VECTOR_BACKEND=mongodb`
- [ ] Fallback logic works when MongoDB unavailable

#### Manual Verification:
- [ ] Chat works with both backends
- [ ] Configuration switches work correctly
- [ ] Error handling gracefully falls back to OpenAI

**Implementation Note**: After completing this phase and automated verification passes, pause for manual confirmation before proceeding to next phase.

---

## Phase 4: Migration Tools & Production Readiness

### Overview
Build utilities for migrating data between backends and add comprehensive testing, monitoring, and documentation.

### Changes Required:

#### 1. Migration Utilities
**File**: `tools/migrate_backend.py`
**Changes**: Tool for migrating between backends

```python
#!/usr/bin/env python3
"""
Migration utility for moving data between vector backends.
"""
import argparse
from datetime import datetime
from config import BackendConfig, BackendType, create_backend

class BackendMigrator:
    def __init__(self, source_config: BackendConfig, target_config: BackendConfig):
        self.source = create_backend(source_config)
        self.target = create_backend(target_config)
    
    def migrate_session(self, session_id: str) -> bool:
        """Migrate a complete session from source to target backend."""
        print(f"🔄 Migrating session {session_id}...")
        
        # Get all archives from source
        if hasattr(self.source, 'export_session_data'):
            session_data = self.source.export_session_data(session_id)
        else:
            print("❌ Source backend doesn't support data export")
            return False
        
        # Set up target store
        target_store_id = self.target.setup_store(session_id)
        
        # Import data to target
        success_count = 0
        for archive in session_data:
            if self.target.archive_messages(
                archive['messages'], 
                archive['archive_id'], 
                session_id
            ):
                success_count += 1
        
        print(f"✅ Migrated {success_count}/{len(session_data)} archives")
        return success_count == len(session_data)

def main():
    parser = argparse.ArgumentParser(description='Migrate data between vector backends')
    parser.add_argument('--source', choices=['openai', 'mongodb'], required=True)
    parser.add_argument('--target', choices=['openai', 'mongodb'], required=True)
    parser.add_argument('--session-id', required=True)
    
    args = parser.parse_args()
    
    source_config = BackendConfig(backend_type=BackendType(args.source))
    target_config = BackendConfig(backend_type=BackendType(args.target))
    
    migrator = BackendMigrator(source_config, target_config)
    migrator.migrate_session(args.session_id)

if __name__ == '__main__':
    main()
```

#### 2. Comprehensive Test Suite
**File**: `tests/test_integration.py`
**Changes**: End-to-end integration tests

```python
import pytest
import os
from unittest.mock import Mock, patch
from infinite_memory_chat import InfiniteMemoryChat
from config import BackendConfig, BackendType

class TestIntegration:
    """Integration tests for both backends."""
    
    @pytest.fixture
    def openai_config(self):
        return BackendConfig(backend_type=BackendType.OPENAI)
    
    @pytest.fixture
    def mongodb_config(self):
        return BackendConfig(
            backend_type=BackendType.MONGODB,
            mongodb_connection_string="mongodb://localhost:27017",
            mongodb_database="test_infinite_memory"
        )
    
    def test_openai_chat_flow(self, openai_config):
        """Test complete chat flow with OpenAI backend."""
        chat = InfiniteMemoryChat(openai_config)
        chat.setup_vector_store()
        
        # Test normal conversation
        response = chat.chat("Hello, my name is Alice")
        assert isinstance(response, str)
        assert len(chat.conversation_history) == 2
    
    def test_mongodb_chat_flow(self, mongodb_config):
        """Test complete chat flow with MongoDB backend."""
        chat = InfiniteMemoryChat(mongodb_config)
        chat.setup_vector_store()
        
        # Test normal conversation
        response = chat.chat("Hello, my name is Bob")
        assert isinstance(response, str)
        assert len(chat.conversation_history) == 2
    
    def test_backend_fallback(self):
        """Test fallback to OpenAI when MongoDB unavailable."""
        config = BackendConfig(
            backend_type=BackendType.MONGODB,
            mongodb_connection_string="mongodb://invalid:27017"
        )
        
        chat = InfiniteMemoryChat(config)
        # Should fallback to OpenAI
        assert chat.backend.__class__.__name__ == "OpenAIBackend"
    
    def test_archival_trigger(self, openai_config):
        """Test that archival triggers at MAX_MESSAGES."""
        chat = InfiniteMemoryChat(openai_config)
        chat.setup_vector_store()
        
        # Fill up to MAX_MESSAGES
        for i in range(21):  # Exceed MAX_MESSAGES
            chat.chat(f"Message {i}")
        
        # Should have triggered archival
        assert len(chat.conversation_history) < 21
        assert chat.archive_count > 0
```

#### 3. Production Documentation
**File**: `docs/DEPLOYMENT.md`
**Changes**: Production deployment guide

```markdown
# Production Deployment Guide

## Backend Selection

### OpenAI Vector Store (Recommended for < 10K messages/month)
```bash
export VECTOR_BACKEND=openai
export OPENAI_API_KEY=your_key_here
export MODEL=gpt-4o-mini
```

### MongoDB Atlas (Recommended for > 50K messages/month)
```bash
export VECTOR_BACKEND=mongodb
export OPENAI_API_KEY=your_key_here  # Still needed for chat and embeddings
export MONGODB_CONNECTION_STRING=mongodb+srv://user:pass@cluster.mongodb.net/
export MONGODB_DATABASE=infinite_memory_chat
```

## MongoDB Atlas Setup

1. Create Atlas cluster (M10+ recommended)
2. Add dedicated search nodes (S20+ recommended)
3. Create vector search index:
   ```json
   {
     "fields": [
       {
         "type": "vector",
         "path": "content_embedding",
         "numDimensions": 1536,
         "similarity": "cosine"
       }
     ]
   }
   ```

## Monitoring

- Monitor archive counts and search performance
- Set up alerts for backend failures
- Track embedding API usage costs
```

### Success Criteria:

#### Automated Verification:
- [ ] Migration tool works: `python tools/migrate_backend.py --help`
- [ ] Full test suite passes: `python -m pytest tests/`
- [ ] Documentation is complete and accurate

#### Manual Verification:
- [ ] Migration between backends preserves all data
- [ ] Production deployment guide works
- [ ] Error handling and monitoring work correctly

**Implementation Note**: After completing this phase and automated verification passes, the MongoDB integration is production-ready.

---

## Testing Strategy

### Unit Tests:
- Backend interface compliance for both implementations
- Vector embedding generation and storage
- Search result accuracy and ranking
- Configuration loading and validation
- Error handling and fallback logic

### Integration Tests:
- End-to-end conversation flows with both backends
- Archive and consolidation operations
- Backend switching and migration
- Performance under load
- Data consistency between backends

### Performance Tests:
- Vector search response times
- Archive operation throughput  
- Memory usage patterns
- Cost analysis at different scales
