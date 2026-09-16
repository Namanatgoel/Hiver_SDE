# DESIGN RATIONALE
# [@B2_wu2026evidence] Hybrid BM25+dense+weighted RRF+cross-encoder — the deployed-system
#   paper showing reranker took top-1 56.8%->75.7%; bigger LLM moved it <1pt. Ranking, not
#   the LLM, is the bottleneck. This is the single highest-leverage change.
# [@B1_xu2024ragkg]    Report MRR/recall@K/nDCG@K separately from answer quality — LinkedIn's
#   RAG+KG paper is the closest published analogue to grounded-in-brand-history retrieval.

"""
src/retriever.py — hybrid BM25 + dense + weighted RRF + cross-encoder rerank.

Device: RTX 5050 (embeddings + reranker)
Phase:  6
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.6"))
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "20"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))


class HybridRetriever:
    """BM25 + dense embedding retrieval with weighted RRF fusion and cross-encoder rerank."""

    def __init__(self, threads: list[dict]) -> None:
        self.threads = threads
        self.docs = [t["first_customer_message"] for t in threads]
        self._bm25 = None
        self._embeddings: np.ndarray | None = None
        self._embed_model = None
        self._reranker = None

    # ------------------------------------------------------------------
    # Build index
    # ------------------------------------------------------------------

    def build_index(self, device: str = "cpu") -> float:
        """Tokenize BM25 and compute dense embeddings. Returns peak VRAM (MB)."""
        import torch
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        # BM25 (CPU)
        tokenized = [doc.lower().split() for doc in self.docs]
        self._bm25 = BM25Okapi(tokenized)
        print(f"  BM25 index built: {len(self.docs):,} docs")

        # Dense embeddings
        print(f"  Loading embedding model {EMBEDDING_MODEL!r} on {device} ...")
        self._embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        self._embeddings = self._embed_model.encode(
            self.docs,
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vram = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0
        print(f"  Dense embeddings: {self._embeddings.shape} | VRAM: {vram:.0f} MB")
        return vram

    def free_embed_model(self) -> None:
        """Release embedding model to free VRAM before loading reranker."""
        import torch
        del self._embed_model
        self._embed_model = None
        if os.getenv("DEVICE", "cpu") == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _bm25_scores(self, query: str) -> np.ndarray:
        tokenized_q = query.lower().split()
        scores = np.array(self._bm25.get_scores(tokenized_q))
        # Normalise to [0, 1]
        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return scores

    def _dense_scores(self, query: str) -> np.ndarray:
        import torch
        device = os.getenv("DEVICE", "cpu")
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        q_emb = self._embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return (self._embeddings @ q_emb)  # cosine sim (already normalised)

    def _rrf_fusion(self, bm25_scores: np.ndarray, dense_scores: np.ndarray, k: int = 60) -> np.ndarray:
        """Weighted Reciprocal Rank Fusion."""
        bm25_ranks = np.argsort(-bm25_scores)
        dense_ranks = np.argsort(-dense_scores)

        bm25_rrf = np.zeros(len(bm25_scores))
        dense_rrf = np.zeros(len(dense_scores))
        for rank, idx in enumerate(bm25_ranks):
            bm25_rrf[idx] = 1.0 / (k + rank + 1)
        for rank, idx in enumerate(dense_ranks):
            dense_rrf[idx] = 1.0 / (k + rank + 1)

        return BM25_WEIGHT * bm25_rrf + DENSE_WEIGHT * dense_rrf

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
        """Retrieve top_k candidates via weighted RRF."""
        bm25 = self._bm25_scores(query)
        dense = self._dense_scores(query)
        fused = self._rrf_fusion(bm25, dense)
        top_idx = np.argsort(-fused)[:top_k]
        return [
            {**self.threads[i], "_retrieval_score": float(fused[i]), "_retrieval_rank": rank}
            for rank, i in enumerate(top_idx)
        ]

    def rerank(self, query: str, candidates: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
        """Cross-encoder rerank; loads/unloads model per call (load-use-del pattern)."""
        import torch
        from sentence_transformers import CrossEncoder

        device = os.getenv("DEVICE", "cpu")
        print(f"    Loading reranker {RERANKER_MODEL!r} on {device} ...")
        reranker = CrossEncoder(RERANKER_MODEL, device=device)
        pairs = [(query, c["first_customer_message"]) for c in candidates]
        scores = reranker.predict(pairs)
        del reranker
        if device == "cuda":
            torch.cuda.empty_cache()

        ranked = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )
        return [
            {**cand, "_rerank_score": float(score), "_rerank_rank": i}
            for i, (cand, score) in enumerate(ranked[:top_k])
        ]

    def query(self, query: str, top_k: int = TOP_K_RERANK) -> list[dict]:
        """Full pipeline: retrieve -> rerank."""
        candidates = self.retrieve(query, top_k=TOP_K_RETRIEVE)
        return self.rerank(query, candidates, top_k=top_k)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, queries: list[str], relevant_thread_ids: list[str]) -> dict:
        """Compute MRR, recall@5, nDCG@5 against known-relevant threads."""
        mrr_sum = 0.0
        recall_sum = 0.0
        ndcg_sum = 0.0

        for query, rel_id in zip(queries, relevant_thread_ids):
            candidates = self.retrieve(query, top_k=TOP_K_RETRIEVE)
            reranked = self.rerank(query, candidates, top_k=5)
            reranked_ids = [c["thread_id"] for c in reranked]

            # MRR
            for rank, tid in enumerate(reranked_ids, 1):
                if tid == rel_id:
                    mrr_sum += 1.0 / rank
                    break

            # Recall@5
            recall_sum += 1.0 if rel_id in reranked_ids else 0.0

            # nDCG@5
            dcg = sum(
                1.0 / np.log2(rank + 2)
                for rank, tid in enumerate(reranked_ids)
                if tid == rel_id
            )
            ndcg_sum += dcg  # ideal DCG = 1.0 (single relevant doc)

        n = len(queries)
        return {
            "MRR": mrr_sum / n,
            "recall@5": recall_sum / n,
            "nDCG@5": ndcg_sum / n,
            "n": n,
        }
