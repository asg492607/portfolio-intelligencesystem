import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "portfolio_vectors"
VECTOR_SIZE = 768  # Matches google-generativeai models/text-embedding-004 or embedding-001

class VectorDBClient:
    def __init__(self):
        try:
            if QDRANT_URL:
                # Cloud/Remote mode
                self.client = QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY
                )
                print(f"Connected to Qdrant remote instance at {QDRANT_URL}")
            else:
                # Local disk storage mode (SQLite-equivalent for Qdrant)
                self.client = QdrantClient(path="./qdrant_local_db")
                print("Initialized local disk-backed Qdrant database")
            
            # Ensure collection exists
            self._ensure_collection()
        except Exception as e:
            print(f"Failed to connect to Qdrant: {e}. Vector search will be mocked.")
            self.client = None

    def _ensure_collection(self):
        if not self.client:
            return
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == COLLECTION_NAME for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=qdrant_models.VectorParams(
                        size=VECTOR_SIZE,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                print(f"Created Qdrant collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"Failed to verify/create Qdrant collection: {e}")

    def add_portfolio_chunk(self, job_id: str, chunk_index: int, text: str, vector: list, metadata: dict):
        """Add an embedded chunk to Qdrant."""
        if not self.client or len(vector) != VECTOR_SIZE:
            return
        
        try:
            import hashlib
            point_id = int(hashlib.md5(f"{job_id}_{chunk_index}".encode()).hexdigest(), 16) % (2**63)
            
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "job_id": job_id,
                            "chunk_index": chunk_index,
                            "text": text,
                            **metadata
                        }
                    )
                ]
            )
        except Exception as e:
            print(f"Error indexing chunk to Qdrant: {e}")

    def query_similar_portfolios(self, vector: list, limit: int = 3) -> list:
        """Search Qdrant for similar portfolio chunks for RAG context."""
        if not self.client or len(vector) != VECTOR_SIZE:
            # Fallback mock portfolio chunks if Qdrant isn't working
            return [
                {
                    "text": "Senior UX Lead: Core case study highlights key conversions, showing user flow maps, spacing tokens, and explicit test metrics.",
                    "score": 0.85
                },
                {
                    "text": "Exceptional Frontend Developer: Layout uses modular CSS grid systems, automated test suites, clean file tree organization.",
                    "score": 0.80
                }
            ]
        
        try:
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vector,
                limit=limit
            )
            return [
                {
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    **{k: v for k, v in hit.payload.items() if k not in ["text"]}
                } for hit in results
            ]
        except Exception as e:
            print(f"Error querying Qdrant: {e}")
            return []

vector_db = VectorDBClient()
