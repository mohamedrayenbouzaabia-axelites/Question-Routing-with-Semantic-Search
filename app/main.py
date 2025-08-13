from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import asyncio
from typing import Dict, List

from app.models import *
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService
from app.config import config

app = FastAPI(
    title="Question Router API",
    description="LLM-based question routing with semantic search and caching",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
embedding_service = EmbeddingService()
vector_service = VectorService()
llm_service = LLMService()

@app.post("/seed", response_model=SeedResponse)
async def seed_classes(request: SeedRequest):
    """Seed new classes or add questions to existing classes"""
    try:
        created_classes = []
        updated_classes = []
        pools = set()
        
        for class_data in request.classes:
            # Generate pool_id if not provided
            if not class_data.pool_id:
                class_data.pool_id = str(uuid.uuid4())
            
            pools.add(class_data.pool_id)
            
            if class_data.mode == SeedMode.NEW_CLASS:
                if not class_data.class_name or not class_data.class_description:
                    raise HTTPException(
                        status_code=400, 
                        detail="class_name and class_description required for new class"
                    )
                
                # Generate new class_id
                class_id = str(uuid.uuid4())
                
                # Create embedding for the class
                embedding = embedding_service.create_class_embedding(
                    class_data.class_name,
                    class_data.class_description,
                    class_data.example_questions
                )
                
                # Add to vector store
                vector_service.add_class(
                    class_data.pool_id,
                    class_id,
                    class_data.class_name,
                    class_data.class_description,
                    class_data.example_questions,
                    embedding
                )
                
                created_classes.append({
                    "class_id": class_id,
                    "class_name": class_data.class_name,
                    "pool_id": class_data.pool_id
                })
                
            elif class_data.mode == SeedMode.EXISTING_CLASS:
                if not class_data.class_id:
                    raise HTTPException(
                        status_code=400,
                        detail="class_id required for existing class mode"
                    )
                
                # TODO: Implement adding questions to existing class
                # This would require updating the embedding with new examples
                updated_classes.append(class_data.class_id)
        
        return SeedResponse(
            success=True,
            created_classes=created_classes,
            updated_classes=updated_classes,
            pools=list(pools)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_classes(request: QueryRequest):
    """Query for the most relevant class in a pool"""
    try:
        # Generate embedding for the question
        question_embedding = await embedding_service.get_embeddings(request.question)
        question_embedding = question_embedding[0]  # Get first embedding
        
        # Search for similar classes in the pool
        similar_classes = vector_service.search_similar_classes(
            request.pool_id, 
            question_embedding,
            config.SIMILARITY_THRESHOLD
        )
        
        search_method = "vector_search"
        created_class = None
        
        if not similar_classes:
            # Vector search failed - try LLM fallback
            all_classes = vector_service.get_all_classes_in_pool(request.pool_id)
            
            if all_classes:
                # Use LLM to match
                llm_probabilities = await llm_service.match_question_to_classes(
                    request.question, all_classes
                )
                search_method = "llm_fallback"
                
                # Convert to results format
                results = []
                for cls in all_classes:
                    prob = llm_probabilities.get(cls['class_id'], 0.0)
                    results.append(ClassProbability(
                        class_id=cls['class_id'],
                        class_name=cls['class_name'],
                        probability=prob
                    ))
                
                # Sort by probability
                results.sort(key=lambda x: x.probability, reverse=True)
                
            elif request.create_class_if_missing and request.class_name and request.class_description:
                # Create new class dynamically
                class_id = str(uuid.uuid4())
                
                embedding = embedding_service.create_class_embedding(
                    request.class_name,
                    request.class_description,
                    [request.question]  # Use the question as an example
                )
                
                vector_service.add_class(
                    request.pool_id,
                    class_id,
                    request.class_name,
                    request.class_description,
                    [request.question],
                    embedding
                )
                
                created_class = {
                    "class_id": class_id,
                    "class_name": request.class_name,
                    "pool_id": request.pool_id
                }
                
                results = [ClassProbability(
                    class_id=class_id,
                    class_name=request.class_name,
                    probability=1.0
                )]
                
            else:
                results = []
        
        else:
            # Convert vector search results to probability distribution
            total_score = sum(score for _, _, score in similar_classes)
            results = []
            
            for class_id, class_name, score in similar_classes:
                probability = score / total_score if total_score > 0 else 0
                results.append(ClassProbability(
                    class_id=class_id,
                    class_name=class_name,
                    probability=probability
                ))
        
        return QueryResponse(
            success=True,
            question=request.question,
            pool_id=request.pool_id,
            results=results,
            created_class=created_class,
            search_method=search_method
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/pools/{pool_id}/classes")
async def get_pool_classes(pool_id: str):
    """Get all classes in a pool"""
    try:
        classes = vector_service.get_all_classes_in_pool(pool_id)
        return {"pool_id": pool_id, "classes": classes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)