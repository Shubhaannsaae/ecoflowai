"""
Vector store implementation using Zilliz Cloud for semantic search and RAG capabilities.
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np

from app.config import (
    ZILLIZ_CLOUD_URI, ZILLIZ_CLOUD_TOKEN, 
    VECTOR_DIMENSION, VECTOR_COLLECTION_NAME,
    get_logger
)

logger = get_logger(__name__)

class ZillizVectorStore:
    """
    Vector store implementation using Zilliz Cloud for semantic search
    """
    
    def __init__(self):
        self.collection_name = VECTOR_COLLECTION_NAME
        self.dimension = VECTOR_DIMENSION
        self.collection = None
        self.connected = False
        
        try:
            self.connect_to_zilliz()
            self.setup_collection()
            self.connected = True
        except Exception as e:
            logger.error(f"Failed to initialize Zilliz vector store: {str(e)}")
            self.connected = False
    
    def connect_to_zilliz(self):
        """Connect to Zilliz Cloud"""
        try:
            from pymilvus import connections
            
            if not ZILLIZ_CLOUD_URI or not ZILLIZ_CLOUD_TOKEN:
                raise ValueError("Zilliz Cloud credentials not configured")
            
            connections.connect(
                alias="default",
                uri=ZILLIZ_CLOUD_URI,
                token=ZILLIZ_CLOUD_TOKEN
            )
            logger.info("Successfully connected to Zilliz Cloud")
        except ImportError:
            logger.error("pymilvus package not installed. Please install with 'pip install pymilvus'")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Zilliz Cloud: {str(e)}")
            raise
    
    def setup_collection(self):
        """Set up the collection schema"""
        try:
            from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            schema = CollectionSchema(fields, "Supply chain sustainability knowledge base")
            
            if not Collection.exists(self.collection_name):
                collection = Collection(self.collection_name, schema)
                
                # Create index for vector search
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "AUTOINDEX",
                    "params": {}
                }
                collection.create_index(field_name="embedding", index_params=index_params)
                logger.info(f"Created collection: {self.collection_name}")
            
            self.collection = Collection(self.collection_name)
            self.collection.load()
            
        except Exception as e:
            logger.error(f"Failed to setup collection: {str(e)}")
            raise
    
    def add_documents(self, texts: List[str], categories: List[str], metadata: List[Dict]):
        """Add documents to the vector store"""
        if not self.connected:
            logger.warning("Vector store not connected, skipping document addition")
            return
            
        try:
            # Generate embeddings
            embeddings = self._generate_embeddings(texts)
            
            # Insert data
            data = [
                texts,
                embeddings,
                categories,
                metadata
            ]
            
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Added {len(texts)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.connected:
            logger.warning("Vector store not connected, returning empty results")
            return []
            
        try:
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])
            
            # Search
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            results = self.collection.search(
                data=query_embedding,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "category", "metadata"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "text": hit.entity.get("text"),
                        "category": hit.entity.get("category"),
                        "metadata": hit.entity.get("metadata"),
                        "score": hit.score
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts using sentence transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(texts).tolist()
            
            return embeddings
            
        except ImportError:
            logger.error("sentence-transformers package not installed. Please install with 'pip install sentence-transformers'")
            # Return random embeddings as fallback
            return [[0.0] * self.dimension for _ in texts]
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            # Return random embeddings as fallback
            return [[0.0] * self.dimension for _ in texts]
    
    def populate_knowledge_base(self):
        """Populate the vector store with sustainability knowledge"""
        if not self.connected:
            logger.warning("Vector store not connected, skipping knowledge base population")
            return
            
        sustainability_docs = [
            "Scope 1 emissions are direct greenhouse gas emissions from sources owned or controlled by the company",
            "Scope 2 emissions are indirect emissions from purchased electricity, steam, heating, and cooling",
            "Scope 3 emissions include all other indirect emissions in the value chain",
            "Transportation by rail typically produces 70% fewer emissions than road transport",
            "Air freight produces 10-50 times more emissions than sea freight per tonne-kilometer",
            "Consolidating shipments can reduce logistics emissions by 15-25%",
            "Local sourcing can reduce transportation emissions by up to 80%",
            "ISO 14001 certification indicates strong environmental management systems",
            "The GHG Protocol provides standards for measuring and managing greenhouse gas emissions",
            "Carbon offsetting should be used as a last resort after reduction efforts",
            "Renewable energy procurement can significantly reduce Scope 2 emissions",
            "Supply chain mapping is essential for Scope 3 emissions calculation",
            "Science-based targets align with climate science to limit global warming",
            "Circular economy principles can reduce material consumption and waste",
            "Life cycle assessment evaluates environmental impacts throughout product lifecycle"
        ]
        
        categories = [
            "emissions", "emissions", "emissions", "transport", "transport", 
            "optimization", "optimization", "certification", "standards", "offsetting",
            "energy", "scope3", "targets", "circular_economy", "assessment"
        ]
        
        metadata = [
            {"source": "ghg_protocol", "type": "definition"},
            {"source": "ghg_protocol", "type": "definition"},
            {"source": "ghg_protocol", "type": "definition"},
            {"source": "transport_data", "type": "fact"},
            {"source": "transport_data", "type": "fact"},
            {"source": "optimization_guide", "type": "recommendation"},
            {"source": "optimization_guide", "type": "recommendation"},
            {"source": "iso_standards", "type": "certification"},
            {"source": "ghg_protocol", "type": "standard"},
            {"source": "carbon_management", "type": "strategy"},
            {"source": "renewable_energy", "type": "strategy"},
            {"source": "scope3_guide", "type": "methodology"},
            {"source": "sbti", "type": "framework"},
            {"source": "circular_economy", "type": "principle"},
            {"source": "lca_methodology", "type": "assessment"}
        ]
        
        try:
            self.add_documents(sustainability_docs, categories, metadata)
            logger.info("Successfully populated knowledge base")
        except Exception as e:
            logger.error(f"Failed to populate knowledge base: {str(e)}")
