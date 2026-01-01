"""
OpenAI Vector Store backend implementation.
"""

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
        self.archived_files = []  # Track file IDs in order
        self.consolidation_count = 0  # Number of times we've consolidated
        
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
        if not self.vector_store_id:
            print("❌ Vector store not initialized")
            return False
            
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
        
        try:
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
            
        except Exception as e:
            print(f"❌ Error archiving to OpenAI: {e}")
            return False
    
    def search_archives(self, query_text: str, limit: int = 10) -> List[Dict]:
        """
        OpenAI handles search through the file_search tool in responses.
        This method returns empty list as search is integrated into chat responses.
        """
        return []
    
    def consolidate_archives(self, archive_ids: List[str]) -> Optional[str]:
        """
        Merge the oldest files into a single large file.
        Preserves all data without summarization.
        """
        if len(archive_ids) == 0:
            return None
            
        # Get files to consolidate based on archive_ids
        files_to_consolidate = []
        for archive_id in archive_ids:
            for file_info in self.archived_files:
                if str(file_info["archive_id"]) == str(archive_id):
                    files_to_consolidate.append(file_info)
                    break
        
        if not files_to_consolidate:
            print("⚠️ No files found to consolidate")
            return None
        
        # Separate already consolidated files that are near the size limit
        MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500MB
        files_to_merge = []
        
        for file_info in files_to_consolidate:
            if file_info.get("is_consolidated"):
                # Check file size
                try:
                    file_size = os.path.getsize(file_info["filename"])
                    if file_size > MAX_FILE_SIZE_BYTES * 0.8:  # 80% of limit
                        print(f"⚠️ Skipping large file ({file_size / 1024 / 1024:.1f} MB)")
                        continue
                except FileNotFoundError:
                    pass
            files_to_merge.append(file_info)
        
        if not files_to_merge:
            print("⚠️ No files to consolidate (all are too large)")
            return None
        
        # Collect all content from files to merge
        all_archives = []
        for file_info in files_to_merge:
            try:
                with open(file_info["filename"], "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # If it's a consolidated file, extract its archives
                    if file_info.get("is_consolidated") and "archives" in data:
                        all_archives.extend(data["archives"])
                    else:
                        all_archives.append(data)
            except FileNotFoundError:
                print(f"⚠️ Could not read {file_info['filename']}")
                continue
        
        self.consolidation_count += 1
        
        print(f"📦 Merging {len(files_to_merge)} files into 1...")
        
        # Create merged file with all data
        consolidated_data = {
            "type": "consolidated_archive",
            "consolidation_id": self.consolidation_count,
            "timestamp": datetime.now().isoformat(),
            "original_archive_ids": [f["archive_id"] for f in files_to_merge],
            "file_count": len(all_archives),
            "archives": all_archives  # All original data preserved
        }
        
        filename = f"/tmp/consolidated_{self.consolidation_count}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(consolidated_data, f, ensure_ascii=False, indent=2)
        
        # Check resulting file size
        file_size = os.path.getsize(filename)
        file_size_mb = file_size / 1024 / 1024
        print(f"📏 Consolidated file size: {file_size_mb:.1f} MB")
        
        if file_size > MAX_FILE_SIZE_BYTES:
            print(f"❌ File too large! Aborting consolidation.")
            os.remove(filename)
            return None
        
        try:
            # Remove old files from vector store
            print(f"🗑️ Removing {len(files_to_merge)} old archive files...")
            for file_info in files_to_merge:
                try:
                    self.client.vector_stores.files.delete(
                        vector_store_id=self.vector_store_id,
                        file_id=file_info["file_id"]
                    )
                    self.client.files.delete(file_id=file_info["file_id"])
                except Exception as e:
                    print(f"⚠️ Could not delete file: {e}")
            
            # Upload consolidated file
            print(f"📤 Uploading merged file...")
            with open(filename, "rb") as f:
                file_obj = self.client.files.create(file=f, purpose="assistants")
            
            self.client.vector_stores.files.create(
                vector_store_id=self.vector_store_id,
                file_id=file_obj.id
            )
            
            self._wait_for_indexing(file_obj.id)
            
            # Update our list of archived files
            # Remove merged ones, add new one
            merged_ids = {f["archive_id"] for f in files_to_merge}
            self.archived_files = [f for f in self.archived_files if f["archive_id"] not in merged_ids]
            
            consolidated_id = f"consolidated_{self.consolidation_count}"
            self.archived_files.append({
                "file_id": file_obj.id,
                "archive_id": consolidated_id,
                "filename": filename,
                "is_consolidated": True
            })
            
            print(f"✅ Consolidation #{self.consolidation_count} complete!")
            print(f"   {len(files_to_merge)} files → 1 file (all data preserved)")
            print(f"   Archive files now: {len(self.archived_files)}")
            
            return consolidated_id
            
        except Exception as e:
            print(f"❌ Error during consolidation: {e}")
            return None
    
    def get_archive_count(self) -> int:
        """Get the number of archived message groups."""
        return len(self.archived_files)
    
    def cleanup_archives(self, archive_ids: List[str]) -> bool:
        """Remove specified archives from storage."""
        success_count = 0
        for archive_id in archive_ids:
            for file_info in self.archived_files:
                if str(file_info["archive_id"]) == str(archive_id):
                    try:
                        self.client.vector_stores.files.delete(
                            vector_store_id=self.vector_store_id,
                            file_id=file_info["file_id"]
                        )
                        self.client.files.delete(file_id=file_info["file_id"])
                        self.archived_files.remove(file_info)
                        success_count += 1
                    except Exception as e:
                        print(f"⚠️ Could not delete archive {archive_id}: {e}")
                    break
        
        return success_count == len(archive_ids)
    
    def get_status(self) -> Dict[str, Any]:
        """Get backend-specific status information."""
        # Calculate total size of consolidated files
        total_consolidated_size = 0
        for f in self.archived_files:
            if f.get("is_consolidated"):
                try:
                    total_consolidated_size += os.path.getsize(f["filename"])
                except FileNotFoundError:
                    pass
        
        return {
            "backend_type": "openai",
            "vector_store_id": self.vector_store_id,
            "archived_files": len(self.archived_files),
            "consolidations": self.consolidation_count,
            "consolidated_size_mb": total_consolidated_size / 1024 / 1024,
            "files": [
                {
                    "archive_id": f["archive_id"],
                    "is_consolidated": f.get("is_consolidated", False)
                }
                for f in self.archived_files
            ]
        }
    
    def should_consolidate(self, max_archives: int) -> bool:
        """Check if consolidation should be triggered."""
        return len(self.archived_files) >= max_archives
    
    def _wait_for_indexing(self, file_id: str, timeout: int = 30) -> bool:
        """Wait until the file is indexed in vector store."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                file_status = self.client.vector_stores.files.retrieve(
                    vector_store_id=self.vector_store_id,
                    file_id=file_id
                )
                if file_status.status == "completed":
                    return True
                time.sleep(1)
            except Exception as e:
                print(f"⚠️ Error checking indexing status: {e}")
                break
        
        print("⚠️ Timeout while indexing")
        return False
