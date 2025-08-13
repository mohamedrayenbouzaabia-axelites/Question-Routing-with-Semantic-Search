# Python
# main.py
#
# API FastAPI pour routage de questions par recherche sémantique (cosine) avec:
# - Seed (création de pools/classes et ajout d'exemples, y compris batch)
# - Query (distribution de probabilité sur classes d'un pool)
# - Cache des réponses dans Redis Vector
# - Repli LLM optionnel (OpenAI) si la similarité est sous un seuil
#
# Démarrage:
#   REDIS_URL=redis://localhost:6379 uvicorn main:app --reload
#
# Dépendances:
#   pip install fastapi uvicorn[standard] redis sentence-transformers numpy pydantic openai

import json
import os
import time
import uuid
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from redis import Redis
from redis.commands.search.field import TextField, TagField, VectorField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

# -----------------------------
# Configuration
# -----------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Seuils/paramètres
SIMILARITY_THRESHOLD_DEFAULT = float(os.getenv("SIMILARITY_THRESHOLD", "0.32"))  # seuil déclenchant LLM fallback
CACHE_HIT_SIM_THRESHOLD = float(os.getenv("CACHE_HIT_SIM_THRESHOLD", "0.95"))    # hit cache si cos sim >= 0.95
SOFTMAX_TEMPERATURE = float(os.getenv("SOFTMAX_TEMPERATURE", "0.5"))
TOP_K_CLASSES_DEFAULT = int(os.getenv("TOP_K_CLASSES_DEFAULT", "8"))

# -----------------------------
# Embeddings provider
# -----------------------------
class EmbeddingsProvider:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True)  # L2 normalisation -> Cosine = dot
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32)
        return vectors.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# -----------------------------
# Redis Vector Store
# -----------------------------
class RedisVectorStore:
    """
    Stocke:
      - Classes: clé "cls:{class_id}" avec champ vectoriel "embedding" (capability)
      - Exemples: clé "ex:{example_id}" (facultatif pour démo; ici surtout pour recalculer les embeddings de classe)
      - Cache: clé "cache:{cache_id}" avec champ vectoriel "embedding" + distribution JSON
    Isolation par pool avec champ pool_id (Tag).
    """
    def __init__(self, redis_url: str, embed_dim: int):
        self.r = Redis.from_url(redis_url)
        self.embed_dim = embed_dim
        self._ensure_indexes()

    @staticmethod
    def _bytes_from_vec(vec: np.ndarray) -> bytes:
        assert vec.dtype == np.float32
        return vec.tobytes()

    def _ensure_indexes(self):
        # Index classes
        try:
            self.r.ft("idx:classes").info()
        except Exception:
            schema = (
                TagField("pool_id"),
                TextField("name"),
                TextField("description"),
                TagField("is_dynamic"),
                NumericField("created_at"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.embed_dim,
                        "DISTANCE_METRIC": "COSINE",
                        "M": 16,
                        "EF_CONSTRUCTION": 200,
                    },
                ),
            )
            self.r.ft("idx:classes").create_index(
                schema=schema,
                definition=IndexDefinition(prefix=["cls:"], index_type=IndexType.HASH),
            )

        # Index examples (utile si vous voulez interroger par exemples)
        try:
            self.r.ft("idx:examples").info()
        except Exception:
            schema = (
                TagField("pool_id"),
                TextField("class_id"),
                TextField("text"),
                NumericField("created_at"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.embed_dim,
                        "DISTANCE_METRIC": "COSINE",
                        "M": 16,
                        "EF_CONSTRUCTION": 200,
                    },
                ),
            )
            self.r.ft("idx:examples").create_index(
                schema=schema,
                definition=IndexDefinition(prefix=["ex:"], index_type=IndexType.HASH),
            )

        # Index cache
        try:
            self.r.ft("idx:cache").info()
        except Exception:
            schema = (
                TagField("pool_id"),
                TextField("query_text"),
                TextField("distribution_json"),
                NumericField("created_at"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.embed_dim,
                        "DISTANCE_METRIC": "COSINE",
                        "M": 16,
                        "EF_CONSTRUCTION": 200,
                    },
                ),
            )
            self.r.ft("idx:cache").create_index(
                schema=schema,
                definition=IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH),
            )

    # -------------------------
    # Helpers - Classes
    # -------------------------
    def create_pool(self) -> str:
        return str(uuid.uuid4())

    def create_class(
        self,
        pool_id: str,
        name: str,
        description: str,
        capability_embedding: np.ndarray,
        is_dynamic: bool = False,
    ) -> str:
        class_id = str(uuid.uuid4())
        key = f"cls:{class_id}"
        self.r.hset(
            key,
            mapping={
                "pool_id": pool_id,
                "name": name,
                "description": description,
                "is_dynamic": "1" if is_dynamic else "0",
                "created_at": int(time.time()),
                "embedding": self._bytes_from_vec(capability_embedding),
            },
        )
        return class_id

    def get_class(self, class_id: str) -> Optional[Dict[str, Any]]:
        key = f"cls:{class_id}"
        data = self.r.hgetall(key)
        if not data:
            return None
        return {
            "class_id": class_id,
            "pool_id": data.get(b"pool_id", b"").decode(),
            "name": data.get(b"name", b"").decode(),
            "description": data.get(b"description", b"").decode(),
            "is_dynamic": data.get(b"is_dynamic", b"0").decode() == "1",
            "created_at": int(data.get(b"created_at", b"0").decode() or "0"),
            "embedding": data.get(b"embedding"),  # bytes
        }

    def list_classes_in_pool(self, pool_id: str) -> List[Dict[str, Any]]:
        q = Query(f'@pool_id:{{{pool_id}}}').paging(0, 10000).return_fields("name", "description", "is_dynamic", "created_at")
        res = self.r.ft("idx:classes").search(q)
        classes = []
        for doc in res.docs:
            class_id = doc.id.split("cls:")[1]
            # récupérer embedding brut si besoin
            emb_bytes = self.r.hget(doc.id, "embedding")
            classes.append(
                {
                    "class_id": class_id,
                    "pool_id": pool_id,
                    "name": getattr(doc, "name", ""),
                    "description": getattr(doc, "description", ""),
                    "is_dynamic": (getattr(doc, "is_dynamic", "0") == "1"),
                    "created_at": int(getattr(doc, "created_at", "0")),
                    "embedding": emb_bytes,
                }
            )
        return classes

    def update_class_embedding_from_parts(
        self,
        class_id: str,
        desc_embedding: Optional[np.ndarray],
        example_embeddings: List[np.ndarray],
    ):
        # moyenne (description + exemples). Si pas d'exemples, juste description
        parts = []
        if desc_embedding is not None:
            parts.append(desc_embedding)
        parts.extend(example_embeddings)
        if not parts:
            return
        mean_vec = np.mean(np.stack(parts, axis=0), axis=0).astype(np.float32)
        mean_vec /= np.linalg.norm(mean_vec) + 1e-12
        self.r.hset(f"cls:{class_id}", mapping={"embedding": self._bytes_from_vec(mean_vec)})

    # -------------------------
    # Helpers - Examples
    # -------------------------
    def add_examples(self, pool_id: str, class_id: str, examples: List[str], emb_provider: EmbeddingsProvider) -> int:
        if not examples:
            return 0
        embs = emb_provider.embed(examples)
        for text, vec in zip(examples, embs):
            ex_id = str(uuid.uuid4())
            self.r.hset(
                f"ex:{ex_id}",
                mapping={
                    "pool_id": pool_id,
                    "class_id": class_id,
                    "text": text,
                    "created_at": int(time.time()),
                    "embedding": self._bytes_from_vec(vec),
                },
            )
        # recalculer l’embedding de la classe (moyenne description + exemples)
        cls = self.get_class(class_id)
        desc_emb = None
        if cls and cls["description"]:
            desc_emb = emb_provider.embed_one(cls["description"])
        self.update_class_embedding_from_parts(
            class_id=class_id,
            desc_embedding=desc_emb,
            example_embeddings=[emb_provider.embed_one(e) for e in examples],
        )
        return len(examples)

    # -------------------------
    # Helpers - Search
    # -------------------------
    def _knn_query_vector(self, index: str, prefix: str, pool_id: str, vec: np.ndarray, top_k: int):
        # RediSearch requête KNN avec filtre pool_id
        base = f'(@pool_id:{{{pool_id}}})=>[KNN {top_k} @embedding $vec_param AS score]'
        q = (
            Query(base)
            .return_fields("name", "description", "score")
            .sort_by("score")
            .paging(0, top_k)
            .dialect(2)
        )
        params = {"vec_param": self._bytes_from_vec(vec)}
        res = self.r.ft(index).search(q, query_params=params)
        out = []
        for doc in res.docs:
            _id = doc.id.split(f"{prefix}:")[1]
            score = float(getattr(doc, "score"))
            # RediSearch HNSW "score" with COSINE is distance; convert to similarity
            similarity = 1.0 - score
            name = getattr(doc, "name", "") if hasattr(doc, "name") else ""
            description = getattr(doc, "description", "") if hasattr(doc, "description") else ""
            out.append({"id": _id, "name": name, "description": description, "similarity": similarity})
        return out

    def knn_classes(self, pool_id: str, vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        return self._knn_query_vector("idx:classes", "cls", pool_id, vec, top_k)

    def knn_cache(self, pool_id: str, vec: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        # retourne id cache + champs utiles
        base = f'(@pool_id:{{{pool_id}}})=>[KNN {top_k} @embedding $vec_param AS score]'
        q = (
            Query(base)
            .return_fields("query_text", "distribution_json", "score")
            .sort_by("score")
            .paging(0, top_k)
            .dialect(2)
        )
        params = {"vec_param": self._bytes_from_vec(vec)}
        res = self.r.ft("idx:cache").search(q, query_params=params)
        out = []
        for doc in res.docs:
            _id = doc.id.split("cache:")[1]
            similarity = 1.0 - float(getattr(doc, "score"))
            out.append(
                {
                    "id": _id,
                    "similarity": similarity,
                    "query_text": getattr(doc, "query_text", ""),
                    "distribution_json": getattr(doc, "distribution_json", ""),
                }
            )
        return out

    # -------------------------
    # Helpers - Cache
    # -------------------------
    def cache_put(self, pool_id: str, query_text: str, query_vec: np.ndarray, distribution: List[Dict[str, Any]]):
        cache_id = str(uuid.uuid4())
        self.r.hset(
            f"cache:{cache_id}",
            mapping={
                "pool_id": pool_id,
                "query_text": query_text,
                "created_at": int(time.time()),
                "distribution_json": json.dumps(distribution, ensure_ascii=False),
                "embedding": self._bytes_from_vec(query_vec),
            },
        )


# -----------------------------
# LLM fallback (optionnel)
# -----------------------------
def llm_pick_class(question: str, classes: List[Dict[str, str]]) -> Optional[Tuple[str, float]]:
    """
    Retourne (class_id, confidence) via LLM si OPENAI_API_KEY présent.
    Sinon, heuristique simple (chevauchement lexical) comme secours.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            # Construit un prompt compact
            labels = [{"class_id": c["class_id"], "name": c["name"], "description": c["description"]} for c in classes]
            system = "Tu es un routeur. Choisis la classe la plus pertinente pour la question. Réponds en JSON."
            user = {
                "question": question,
                "classes": labels,
                "instruction": "Réponds au format strict: {\"class_id\":\"...\",\"confidence\":0.0-1.0}"
            }
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
            )
            content = resp.choices[0].message.content.strip()
            data = json.loads(content)
            return data.get("class_id"), float(data.get("confidence", 0.6))
        except Exception:
            pass

    # Heuristique de secours (si pas de clé ou erreur): score TF simples
    def score(c: Dict[str, str]) -> float:
        q_tokens = set(question.lower().split())
        text = (c.get("name", "") + " " + c.get("description", "")).lower()
        return sum(1 for t in q_tokens if t in text) / (len(q_tokens) + 1e-6)

    best = None
    best_s = -1.0
    for c in classes:
        s = score(c)
        if s > best_s:
            best_s = s
            best = c["class_id"]
    conf = min(0.9, max(0.5, best_s)) if best is not None else 0.5
    return (best, conf) if best else None


# -----------------------------
# Utils
# -----------------------------
def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.array(scores, dtype=np.float32) / max(1e-6, temperature)
    x = x - np.max(x)
    e = np.exp(x)
    s = e / (np.sum(e) + 1e-12)
    return s


def bytes_to_vec(b: bytes, dim: int) -> np.ndarray:
    v = np.frombuffer(b, dtype=np.float32)
    if v.shape[0] != dim:
        raise ValueError("Embedding size mismatch")
    return v


# -----------------------------
# Schémas API
# -----------------------------
class SeedItem(BaseModel):
    pool_id: Optional[str] = None
    class_id: Optional[str] = None
    class_name: Optional[str] = None
    class_description: Optional[str] = ""
    examples: List[str] = Field(default_factory=list)


class SeedRequest(BaseModel):
    items: List[SeedItem]


class SeedResponse(BaseModel):
    pools_created: List[str]
    classes_created: List[Dict[str, Any]]
    classes_updated: List[Dict[str, Any]]
    stats: Dict[str, Any]


class DynamicClass(BaseModel):
    class_name: str
    class_description: str = ""
    examples: List[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    pool_id: str
    question: str
    dynamic_class: Optional[DynamicClass] = None
    top_k: Optional[int] = TOP_K_CLASSES_DEFAULT
    similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD_DEFAULT
    use_llm_fallback: Optional[bool] = True


class ClassProb(BaseModel):
    class_id: str
    class_name: str
    probability: float
    similarity: float


class QueryResponse(BaseModel):
    pool_id: str
    cached: bool
    method: str  # "cache" | "vector" | "llm_fallback"
    distribution: List[ClassProb]
    selected_class_id: str
    selected_class_name: str


# -----------------------------
# App & dépendances globales
# -----------------------------
app = FastAPI(title="LLM Question Router API", version="1.0.0")

_embeddings = EmbeddingsProvider()
_store = RedisVectorStore(REDIS_URL, _embeddings.dim)


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/seed", response_model=SeedResponse)
def seed(req: SeedRequest):
    pools_created = []
    classes_created = []
    classes_updated = []
    examples_added = 0

    for item in req.items:
        pool_id = item.pool_id or _store.create_pool()
        if not item.pool_id:
            pools_created.append(pool_id)

        if item.class_id:
            # Ajouter des exemples à une classe existante
            cls = _store.get_class(item.class_id)
            if not cls:
                raise HTTPException(status_code=404, detail=f"Classe introuvable: {item.class_id}")
            if cls["pool_id"] != pool_id:
                raise HTTPException(status_code=400, detail="pool_id ne correspond pas à la classe donnée")
            cnt = _store.add_examples(pool_id, item.class_id, item.examples, _embeddings)
            examples_added += cnt
            classes_updated.append({"class_id": item.class_id, "pool_id": pool_id, "examples_added": cnt})
        else:
            # Créer une nouvelle classe depuis nom/description
            if not item.class_name:
                raise HTTPException(status_code=400, detail="class_name requis si class_id absent")
            desc_vec = _embeddings.embed_one(item.class_description or item.class_name)
            example_vecs = _embeddings.embed(item.examples) if item.examples else np.zeros((0, _embeddings.dim), dtype=np.float32)
            if example_vecs.size > 0:
                parts = [desc_vec] + [v for v in example_vecs]
                capability_vec = (np.mean(np.stack(parts, axis=0), axis=0)).astype(np.float32)
                capability_vec /= np.linalg.norm(capability_vec) + 1e-12
            else:
                capability_vec = desc_vec
            class_id = _store.create_class(pool_id, item.class_name, item.class_description or "", capability_vec, is_dynamic=False)
            classes_created.append({"class_id": class_id, "pool_id": pool_id, "class_name": item.class_name})
            if item.examples:
                cnt = _store.add_examples(pool_id, class_id, item.examples, _embeddings)
                examples_added += cnt

    resp = SeedResponse(
        pools_created=list(set(pools_created)),
        classes_created=classes_created,
        classes_updated=classes_updated,
        stats={"examples_added": examples_added},
    )
    return resp


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # 1) Embedding de la question
    q_vec = _embeddings.embed_one(req.question)

    # 2) Cache lookup
    cache_hits = _store.knn_cache(req.pool_id, q_vec, top_k=3)
    for hit in cache_hits:
        if hit["similarity"] >= CACHE_HIT_SIM_THRESHOLD:
            try:
                dist = json.loads(hit["distribution_json"])
                # Retour direct depuis cache
                selected = max(dist, key=lambda d: d["probability"])
                return QueryResponse(
                    pool_id=req.pool_id,
                    cached=True,
                    method="cache",
                    distribution=[
                        ClassProb(
                            class_id=d["class_id"], class_name=d["class_name"], probability=d["probability"], similarity=d.get("similarity", 0.0)
                        )
                        for d in dist
                    ],
                    selected_class_id=selected["class_id"],
                    selected_class_name=selected["class_name"],
                )
            except Exception:
                pass  # si cache corrompu on continue

    # 3) Charger toutes les classes du pool (petits pools => OK)
    classes = _store.list_classes_in_pool(req.pool_id)

    # 3.1 Classe dynamique (optionnelle)
    dyn_class_id = None
    if req.dynamic_class:
        dyn_desc_vec = _embeddings.embed_one(req.dynamic_class.class_description or req.dynamic_class.class_name)
        dyn_ex_vecs = _embeddings.embed(req.dynamic_class.examples) if req.dynamic_class.examples else np.zeros((0, _embeddings.dim), dtype=np.float32)
        if dyn_ex_vecs.size > 0:
            parts = [dyn_desc_vec] + [v for v in dyn_ex_vecs]
            dyn_cap = (np.mean(np.stack(parts, axis=0), axis=0)).astype(np.float32)
            dyn_cap /= np.linalg.norm(dyn_cap) + 1e-12
        else:
            dyn_cap = dyn_desc_vec
        dyn_class_id = _store.create_class(
            pool_id=req.pool_id,
            name=req.dynamic_class.class_name,
            description=req.dynamic_class.class_description or "",
            capability_embedding=dyn_cap,
            is_dynamic=True,
        )
        classes.append(
            {
                "class_id": dyn_class_id,
                "pool_id": req.pool_id,
                "name": req.dynamic_class.class_name,
                "description": req.dynamic_class.class_description or "",
                "is_dynamic": True,
                "created_at": int(time.time()),
                "embedding": _store.r.hget(f"cls:{dyn_class_id}", "embedding"),
            }
        )

    if not classes:
        raise HTTPException(status_code=404, detail="Aucune classe dans ce pool")

    # 4) Similarités question -> classes
    names = []
    ids = []
    sims = []
    for cls in classes:
        emb_bytes = cls["embedding"]
        vec = bytes_to_vec(emb_bytes, _embeddings.dim)
        sim = float(np.dot(q_vec, vec))  # embeddings normalisés => dot = cosine
        sims.append(sim)
        ids.append(cls["class_id"])
        names.append(cls["name"])
    sims = np.array(sims, dtype=np.float32)

    # 5) Si top similarity < seuil => repli LLM (optionnel)
    method = "vector"
    max_sim = float(np.max(sims))
    if req.use_llm_fallback and max_sim < (req.similarity_threshold or SIMILARITY_THRESHOLD_DEFAULT):
        pick = llm_pick_class(req.question, [{"class_id": i, "name": n, "description": cl["description"]} for i, n, cl in zip(ids, names, classes)])
        if pick:
            method = "llm_fallback"
            picked_id, confidence = pick
            # Pondération: 70% LLM, 30% similarité vectorielle (softmax)
            sim_probs = softmax(sims, temperature=SOFTMAX_TEMPERATURE)
            llm_prior = np.zeros_like(sim_probs)
            idx = ids.index(picked_id) if picked_id in ids else int(np.argmax(sim_probs))
            llm_prior[idx] = 1.0
            probs = 0.7 * (confidence * llm_prior + (1 - confidence) * sim_probs) + 0.3 * sim_probs
            probs = probs / (probs.sum() + 1e-12)
        else:
            # Pas de sortie LLM -> softmax classique
            probs = softmax(sims, temperature=SOFTMAX_TEMPERATURE)
    else:
        probs = softmax(sims, temperature=SOFTMAX_TEMPERATURE)

    # 6) Limiter à top_k pour la réponse (distribution somme = 1.0)
    top_k = req.top_k or TOP_K_CLASSES_DEFAULT
    order = np.argsort(-probs)
    top_idx = order[:top_k]
    top_probs = probs[top_idx]
    top_probs = top_probs / (top_probs.sum() + 1e-12)

    distribution = []
    for i, p in zip(top_idx, top_probs):
        distribution.append(
            {
                "class_id": ids[i],
                "class_name": names[i],
                "probability": float(p),
                "similarity": float(sims[i]),
            }
        )
    selected = max(distribution, key=lambda d: d["probability"])

    # 7) Cache write
    _store.cache_put(req.pool_id, req.question, q_vec, distribution)

    return QueryResponse(
        pool_id=req.pool_id,
        cached=False,
        method=method,
        distribution=[ClassProb(**d) for d in distribution],
        selected_class_id=selected["class_id"],
        selected_class_name=selected["class_name"],
    )


# -----------------------------
# Exemple d’utilisation rapide (pour tests)
# -----------------------------
# 1) Lancer le serveur:
#    REDIS_URL=redis://localhost:6379 uvicorn main:app --reload
#
# 2) Seed - création batch (mélange de nouvelles classes et ajout à une classe existante):
#
# curl -X POST http://localhost:8000/seed -H "Content-Type: application/json" -d @- <<'JSON'
# {
#   "items": [
#     {
#       "class_name": "FAQ Facturation",
#       "class_description": "Questions sur factures, paiements, remboursements.",
#       "examples": [
#         "Où trouver ma facture ?",
#         "Comment mettre à jour ma carte de crédit ?",
#         "Puis-je obtenir un remboursement ?"
#       ]
#     },
#     {
#       "class_name": "Support Technique",
#       "class_description": "Problèmes techniques, erreurs, performances.",
#       "examples": [
#         "L'application plante au démarrage",
#         "Erreur 500 sur le tableau de bord",
#         "Lenteur lors du chargement des données"
#       ]
#     }
#   ]
# }
# JSON
#
# Note: La réponse renverra les class_id et le(s) pool_id créé(s).
# Vous pouvez ensuite réutiliser un class_id pour ajouter des exemples:
#
# curl -X POST http://localhost:8000/seed -H "Content-Type: application/json" -d @- <<'JSON'
# {
#   "items": [
#     {
#       "pool_id": "<POOL_ID_RECU>",
#       "class_id": "<CLASS_ID_SUPPORT_TECHNIQUE>",
#       "examples": [
#         "Le bouton sauvegarder ne répond pas",
#         "Impossible de me connecter"
#       ]
#     }
#   ]
# }
# JSON
#
# 3) Query - distribution de probabilité
#
# curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d @- <<'JSON'
# {
#   "pool_id": "<POOL_ID_RECU>",
#   "question": "Comment obtenir une copie de ma dernière facture ?",
#   "top_k": 5,
#   "similarity_threshold": 0.35,
#   "use_llm_fallback": true
# }
# JSON
#
# 4) Query avec classe dynamique (créée à la volée et incluse dans la distribution):
#
# curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d @- <<'JSON'
# {
#   "pool_id": "<POOL_ID_RECU>",
#   "question": "Comment migrer mes données vers la nouvelle version ?",
#   "dynamic_class": {
#     "class_name": "Migration & Mises à niveau",
#     "class_description": "Questions sur migrations de versions, compatibilité, guides.",
#     "examples": ["Migrer vers la v2", "Compatibilité des sauvegardes"]
#   }
# }
# JSON
#
# 5) Cache: répétez exactement la même question; la réponse sera servie depuis le cache si la similarité est suffisante.


# -----------------------------
# Notes de conception
# -----------------------------
# - Les embeddings et le cache sont intégralement stockés dans Redis Vector:
#   * Classes: champ vectoriel "embedding" (capability) + métadonnées
#   * Cache: champ vectoriel "embedding" + distribution sérialisée
# - Isolation par pool via le champ Tag pool_id, évitant le mélange de projets.
# - L’indice HNSW en mémoire minimise la latence de recherche; l’encodage (embedding) reste la partie la plus coûteuse.
# - Le repli LLM est invoqué si la similarité maxi est sous un seuil; si aucune clé n’est fournie, un heuristique léger est appliqué.
# - Conçu de manière modulaire (EmbeddingsProvider, RedisVectorStore) pour pouvoir échanger de backend vectoriel si souhaité.