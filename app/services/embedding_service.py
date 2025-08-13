import openai
import numpy as np
from typing import List, Union
import time
import asyncio
from app.config import config

class EmbeddingService:
    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set")
        openai.api_key = config.OPENAI_API_KEY
    
    async def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Get embeddings for text(s)"""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            start_time = time.time()
            response = await openai.Embedding.acreate(
                input=texts,
                model=config.EMBEDDING_MODEL
            )
            
            embeddings = np.array([item['embedding'] for item in response['data']])
            
            # Log timing for performance monitoring
            elapsed = (time.time() - start_time) * 1000
            print(f"Embedding generation took {elapsed:.2f}ms for {len(texts)} texts")
            
            return embeddings
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def create_class_embedding(self, class_name: str, class_description: str, 
                              example_questions: List[str]) -> np.ndarray:
        """Create a composite embedding for a class"""
        # Combine class info into a single text
        class_text = f"Class: {class_name}\nDescription: {class_description}\n"
        class_text += "Examples:\n" + "\n".join(example_questions)
        
        # This is a synchronous wrapper for the async method
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, we need to use run_in_executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.get_embeddings(class_text))
                embedding = future.result()
        else:
            embedding = asyncio.run(self.get_embeddings(class_text))
        
        return embedding[0]  # Return first (and only) embedding