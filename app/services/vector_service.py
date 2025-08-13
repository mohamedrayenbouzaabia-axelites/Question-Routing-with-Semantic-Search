import numpy as np
import faiss
import pickle
import os
import json
import uuid
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
from app.config import config
import time

class VectorStore(ABC):
    @abstractmethod
    def add_vector(self, pool_id: str, class_id: str, class_name: str, 
                   vector: np.ndarray, metadata: Dict) -> None:
        pass
    
    @abstractmethod
    def search(self, pool_id: str, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, str, float]]:
        pass
    
    @abstractmethod
    def get_all_classes_in_pool(self, pool_id: str) -> List[Dict]:
        pass

class FAISSVectorStore(VectorStore):
    def __init__(self):
        self.pools = {}  # pool_id -> {'index': faiss.Index, 'metadata': List[Dict]}
        self.storage_dir = "./faiss_storage"
        os.makedirs(self.storage_dir, exist_ok=True)
        self._load_indices()
    
    def _get_pool_path(self, pool_id: str) -> str:
        return os.path.join(self.storage_dir, f"{pool_id}.pkl")
    
    def _load_indices(self):
        """Load existing indices from disk"""
        for file in os.listdir(self.storage_dir):
            if file.endswith('.pkl'):
                pool_id = file[:-4]
                try:
                    with open(os.path.join(self.storage_dir, file), 'rb') as f:
                        data = pickle.load(f)
                        self.pools[pool_id] = data
                        print(f"Loaded pool {pool_id} with {len(data['metadata'])} vectors")
                except Exception as e:
                    print(f"Error loading pool {pool_id}: {e}")
    
    def _save_pool(self, pool_id: str):
        """Save pool to disk"""
        if pool_id in self.pools:
            with open(self._get_pool_path(pool_id), 'wb') as f:
                pickle.dump(self.pools[pool_id], f)
    
    def add_vector(self, pool_id: str, class_id: str, class_name: str, 
                   vector: np.ndarray, metadata: Dict) -> None:
        if pool_id not in self.pools:
            # Create new pool with FAISS index
            dimension = len(vector)
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.pools[pool_id] = {
                'index': index,
                'metadata': []
            }
        
        # Normalize vector for cosine similarity
        normalized_vector = vector / np.linalg.norm(vector)
        
        # Add to FAISS index
        self.pools[pool_id]['index'].add(normalized_vector.reshape(1, -1).astype('float32'))
        
        # Add metadata
        metadata.update({
            'class_id': class_id,
            'class_name': class_name
        })
        self.pools[pool_id]['metadata'].append(metadata)
        
        # Save to disk
        self._save_pool(pool_id)
    
    def search(self, pool_id: str, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, str, float]]:
        if pool_id not in self.pools:
            return []
        
        start_time = time.time()
        
        # Normalize query vector
        normalized_query = query_vector / np.linalg.norm(query_vector)
        
        # Search in FAISS
        pool_data = self.pools[pool_id]
        scores, indices = pool_data['index'].search(
            normalized_query.reshape(1, -1).astype('float32'), 
            min(k, len(pool_data['metadata']))
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                metadata = pool_data['metadata'][idx]
                results.append((
                    metadata['class_id'],
                    metadata['class_name'],
                    float(score)
                ))
        
        elapsed = (time.time() - start_time) * 1000
        print(f"Vector search took {elapsed:.2f}ms")
        
        return results
    
    def get_all_classes_in_pool(self, pool_id: str) -> List[Dict]:
        if pool_id not in self.pools:
            return []
        
        return [
            {
                'class_id': meta['class_id'],
                'class_name': meta['class_name']
            }
            for meta in self.pools[pool_id]['metadata']
        ]

class VectorService:
    def __init__(self):
        if config.VECTOR_DB_TYPE == "faiss":
            self.store = FAISSVectorStore()
        else:
            raise ValueError(f"Unsupported vector DB type: {config.VECTOR_DB_TYPE}")
    
    def add_class(self, pool_id: str, class_id: str, class_name: str, 
                  class_description: str, example_questions: List[str], 
                  embedding: np.ndarray) -> None:
        metadata = {
            'class_description': class_description,
            'example_questions': example_questions,
            'created_at': time.time()
        }
        
        self.store.add_vector(pool_id, class_id, class_name, embedding, metadata)
    
    def search_similar_classes(self, pool_id: str, query_embedding: np.ndarray, 
                               threshold: float = None) -> List[Tuple[str, str, float]]:
        if threshold is None:
            threshold = config.SIMILARITY_THRESHOLD
        
        results = self.store.search(pool_id, query_embedding, config.MAX_RESULTS)
        
        # Filter by threshold
        filtered_results = [
            (class_id, class_name, score) 
            for class_id, class_name, score in results 
            if score >= threshold
        ]
        
        return filtered_results
    
    def get_all_classes_in_pool(self, pool_id: str) -> List[Dict]:
        return self.store.get_all_classes_in_pool(pool_id)