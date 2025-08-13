from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class SeedMode(str, Enum):
    NEW_CLASS = "new_class"
    EXISTING_CLASS = "existing_class"

class ClassData(BaseModel):
    mode: SeedMode
    class_id: Optional[str] = None
    class_name: Optional[str] = None
    class_description: Optional[str] = None
    example_questions: List[str]
    pool_id: Optional[str] = None

class SeedRequest(BaseModel):
    classes: List[ClassData]

class SeedResponse(BaseModel):
    success: bool
    created_classes: List[Dict[str, str]]
    updated_classes: List[str]
    pools: List[str]

class QueryRequest(BaseModel):
    question: str
    pool_id: str
    create_class_if_missing: Optional[bool] = False
    class_name: Optional[str] = None
    class_description: Optional[str] = None

class ClassProbability(BaseModel):
    class_id: str
    class_name: str
    probability: float

class QueryResponse(BaseModel):
    success: bool
    question: str
    pool_id: str
    results: List[ClassProbability]
    created_class: Optional[Dict[str, str]] = None
    search_method: str  # "vector_search" or "llm_fallback"