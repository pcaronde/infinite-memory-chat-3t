"""
MongoDB Vector Search backend implementation.
"""

import os
import json
import struct
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError, OperationFailure
from bson import Binary
from openai import OpenAI

from .base import VectorBackend


class MongoDBBackend(VectorBackend):
    """MongoDB Vector Search backend implementation."""
    
    def __init__(self, 
                 connection_string: str, 
                 database_name: str = "infinite_memory_chat", 
                 embedding_client: Optional[OpenAI] = None,
                 embedding_model: str = "text-embedding-3-small"):
        self.connection_string = connection_string
        self.database_name = database_name
        self.embedding_model = embedding_model
        self.embedding_client = embedding_client or OpenAI()
        
        # Connection will be established lazily
        self.client: Optional[MongoClient] = None
        self.db = None
        self.collection_name = None
        self.collection: Optional[Collection] = None
        
        # State tracking
        self.archive_count = 0
        self.consolidated_archives = []
        
    def _connect(self):
        """Establish MongoDB connection if not already connected."""
        if self.client is None:
            try:
                self.client = MongoClient(self.connection_string)
                self.db = self.client[self.database_name]
                # Test the connection
                self.client.admin.command('hello')
                print(f"✅ Connected to MongoDB: {self.database_name}")
            except PyMongoError as e:
                print(f"❌ Failed to connect to MongoDB: {e}")
                raise
                
    def setup_store(self, session_id: str) -> str:
        """Create MongoDB collection and set up vector search index."""
        self._connect()
        
        print("📦 Setting up MongoDB collection and vector index...")
        
        self.collection_name = f"chat_memory_{session_id}"
        self.collection = self.db[self.collection_name]
        
        # Create the collection explicitly if it doesn't exist
        if self.collection_name not in self.db.list_collection_names():
            self.db.create_collection(self.collection_name)
        
        # Note: Vector search indexes need to be created manually in MongoDB Atlas UI
        # or via the Atlas Admin API. For production, this would be automated.
        # For now, we'll document the required index structure.
        
        print(f"✅ MongoDB collection created: {self.collection_name}")
        print("ℹ️  Vector search index required (create manually in Atlas):")
        print("   Index name: vector_index")
        print("   Field: content_embedding (vector, 1536 dimensions, cosine)")
        print("   Filter fields: session_id, archive_id, type")
        
        return self.collection_name
    
    def archive_messages(self, messages: List[Dict], archive_id: int, session_id: str) -> bool:
        """Archive messages to MongoDB with vector embeddings."""
        if not self.collection:
            print("❌ MongoDB collection not initialized")
            return False
            
        print(f"📁 Archiving {len(messages)} messages to MongoDB...")
        
        try:
            # Create summary text for embedding
            summary_text = f"Conversation archive #{archive_id}\n"
            summary_text += f"Timestamp: {datetime.now().isoformat()}\n"
            summary_text += f"Session: {session_id}\n"
            summary_text += f"Content:\n"
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                summary_text += f"\n{role}: {msg['content']}\n"
            
            # Generate embedding using OpenAI
            print(f"🧠 Generating embedding for archive #{archive_id}...")
            embedding_response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=summary_text
            )
            embedding_vector = embedding_response.data[0].embedding
            
            # Convert to BinData for efficient storage (optional - can use array too)
            embedding_bindata = self._vector_to_bindata(embedding_vector)
            
            # Create document with comprehensive metadata
            archive_doc = {
                "_id": f"{session_id}_{archive_id}",
                "session_id": session_id,
                "archive_id": archive_id,
                "timestamp": datetime.now(),
                "messages": messages,
                "content_text": summary_text,
                "content_embedding": embedding_bindata,  # BinData format
                "content_embedding_array": embedding_vector,  # Array format for compatibility
                "message_count": len(messages),
                "type": "archive",
                "embedding_model": self.embedding_model,
                "created_at": datetime.now()
            }
            
            # Insert document
            result = self.collection.insert_one(archive_doc)
            self.archive_count += 1
            
            print(f"✅ Archive #{archive_id} saved to MongoDB (ID: {result.inserted_id})")
            return True
            
        except Exception as e:
            print(f"❌ Error archiving to MongoDB: {e}")
            return False
    
    def search_archives(self, query_text: str, limit: int = 10) -> List[Dict]:
        """Search archived messages using MongoDB vector search."""
        if not self.collection:
            print("❌ MongoDB collection not initialized")
            return []
        
        try:
            print(f"🔍 Searching archives for: '{query_text[:50]}...'")
            
            # Generate query embedding
            embedding_response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=query_text
            )
            query_vector = embedding_response.data[0].embedding
            
            # MongoDB vector search aggregation pipeline
            # Note: This requires a vector search index to be created in Atlas
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",  # Must match the index name in Atlas
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
                        "session_id": 1,
                        "content_text": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            # Execute the search
            results = list(self.collection.aggregate(pipeline))
            
            print(f"✅ Found {len(results)} relevant archives")
            
            return results
            
        except OperationFailure as e:
            if "vector search" in str(e).lower():
                print(f"⚠️  Vector search index not found. Create 'vector_index' in MongoDB Atlas")
                print(f"   Error: {e}")
            else:
                print(f"❌ MongoDB operation failed: {e}")
            return []
        except Exception as e:
            print(f"❌ Error searching MongoDB: {e}")
            return []
    
    def consolidate_archives(self, archive_ids: List[str]) -> Optional[str]:
        """
        Consolidate multiple archives into a single document.
        Unlike file-based consolidation, this merges documents in the database.
        """
        if not self.collection or not archive_ids:
            return None
            
        try:
            print(f"📦 Consolidating {len(archive_ids)} archives in MongoDB...")
            
            # Fetch archives to consolidate
            archives_to_merge = list(self.collection.find({
                "archive_id": {"$in": [int(aid) for aid in archive_ids if aid.isdigit()]},
                "type": "archive"
            }))
            
            if not archives_to_merge:
                print("⚠️  No archives found to consolidate")
                return None
            
            # Create consolidated document
            consolidated_id = f"consolidated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Collect all messages and create combined content
            all_messages = []
            combined_content = f"Consolidated Archive: {consolidated_id}\n"
            combined_content += f"Original Archives: {[a['archive_id'] for a in archives_to_merge]}\n"
            combined_content += f"Timestamp: {datetime.now().isoformat()}\n"
            combined_content += f"Content:\n"
            
            for archive in archives_to_merge:
                all_messages.extend(archive.get('messages', []))
                combined_content += f"\n--- Archive #{archive['archive_id']} ---\n"
                for msg in archive.get('messages', []):
                    role = "User" if msg["role"] == "user" else "Assistant"
                    combined_content += f"{role}: {msg['content']}\n"
            
            # Generate new embedding for consolidated content
            print(f"🧠 Generating embedding for consolidated archive...")
            embedding_response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=combined_content
            )
            consolidated_vector = embedding_response.data[0].embedding
            consolidated_bindata = self._vector_to_bindata(consolidated_vector)
            
            # Create consolidated document
            consolidated_doc = {
                "_id": consolidated_id,
                "session_id": archives_to_merge[0]['session_id'],
                "archive_id": consolidated_id,
                "timestamp": datetime.now(),
                "messages": all_messages,
                "content_text": combined_content,
                "content_embedding": consolidated_bindata,
                "content_embedding_array": consolidated_vector,
                "message_count": len(all_messages),
                "type": "consolidated",
                "original_archive_ids": [a['archive_id'] for a in archives_to_merge],
                "consolidation_count": len(archives_to_merge),
                "embedding_model": self.embedding_model,
                "created_at": datetime.now()
            }
            
            # Insert consolidated document
            result = self.collection.insert_one(consolidated_doc)
            
            # Remove original archives
            delete_result = self.collection.delete_many({
                "archive_id": {"$in": [int(aid) for aid in archive_ids if aid.isdigit()]},
                "type": "archive"
            })
            
            print(f"✅ Consolidated {len(archives_to_merge)} archives into {consolidated_id}")
            print(f"   Deleted {delete_result.deleted_count} original archives")
            
            self.consolidated_archives.append(consolidated_id)
            return consolidated_id
            
        except Exception as e:
            print(f"❌ Error during consolidation: {e}")
            return None
    
    def get_archive_count(self) -> int:
        """Get the number of archived message groups."""
        if not self.collection:
            return 0
            
        try:
            count = self.collection.count_documents({"type": {"$in": ["archive", "consolidated"]}})
            return count
        except Exception as e:
            print(f"❌ Error getting archive count: {e}")
            return 0
    
    def cleanup_archives(self, archive_ids: List[str]) -> bool:
        """Remove specified archives from MongoDB."""
        if not self.collection:
            return False
            
        try:
            # Convert string IDs to appropriate types
            numeric_ids = [int(aid) for aid in archive_ids if aid.isdigit()]
            string_ids = [aid for aid in archive_ids if not aid.isdigit()]
            
            # Delete documents
            result = self.collection.delete_many({
                "$or": [
                    {"archive_id": {"$in": numeric_ids}},
                    {"archive_id": {"$in": string_ids}}
                ]
            })
            
            print(f"✅ Deleted {result.deleted_count} archives from MongoDB")
            return result.deleted_count > 0
            
        except Exception as e:
            print(f"❌ Error cleaning up archives: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get backend-specific status information."""
        status = {
            "backend_type": "mongodb",
            "database_name": self.database_name,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "archived_files": 0,  # For compatibility with OpenAI backend
            "consolidations": len(self.consolidated_archives),
            "consolidated_size_mb": 0,  # MongoDB doesn't have direct file sizes
            "connection_status": "disconnected"
        }
        
        if self.collection:
            try:
                # Get document counts by type
                archive_count = self.collection.count_documents({"type": "archive"})
                consolidated_count = self.collection.count_documents({"type": "consolidated"})
                
                # Get collection stats (approximate size)
                stats = self.db.command("collStats", self.collection_name)
                collection_size_mb = stats.get("size", 0) / (1024 * 1024)
                
                status.update({
                    "archived_files": archive_count,
                    "consolidated_archives": consolidated_count,
                    "total_documents": archive_count + consolidated_count,
                    "consolidated_size_mb": collection_size_mb,
                    "connection_status": "connected",
                    "index_count": len(stats.get("indexSizes", {})),
                    "storage_size_mb": stats.get("storageSize", 0) / (1024 * 1024)
                })
                
            except Exception as e:
                print(f"⚠️  Error getting detailed status: {e}")
        
        return status
    
    def should_consolidate(self, max_archives: int) -> bool:
        """Check if consolidation should be triggered."""
        if not self.collection:
            return False
            
        try:
            archive_count = self.collection.count_documents({"type": "archive"})
            return archive_count >= max_archives
        except Exception:
            return False
    
    def _vector_to_bindata(self, vector: List[float]) -> Binary:
        """Convert float vector to MongoDB BinData for efficient storage."""
        try:
            # Pack as 32-bit floats
            byte_data = struct.pack(f'{len(vector)}f', *vector)
            return Binary(byte_data, subtype=4)  # float32 subtype
        except Exception as e:
            print(f"⚠️  Error converting vector to BinData: {e}")
            # Fallback to array storage
            return vector
    
    def _bindata_to_vector(self, bindata: Binary) -> List[float]:
        """Convert MongoDB BinData back to float vector."""
        try:
            if isinstance(bindata, Binary):
                # Unpack from bytes
                float_count = len(bindata) // 4  # 4 bytes per float
                return list(struct.unpack(f'{float_count}f', bindata))
            else:
                # Already an array
                return bindata
        except Exception as e:
            print(f"⚠️  Error converting BinData to vector: {e}")
            return []
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None
            print("✅ MongoDB connection closed")
    
    def __del__(self):
        """Ensure connection is closed when object is destroyed."""
        self.close()
