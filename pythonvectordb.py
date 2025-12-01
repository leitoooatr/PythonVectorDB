#!/usr/bin/env python3
"""
PythonVectorDB – Pure Python vector database (int8 quantized)
Version: 1.0.0
License: MIT
"""

import sys
import numpy as np
import time
import threading
from typing import List, Dict, Any, Tuple, Optional, Callable
from collections import deque
from numba import njit, prange
from datetime import datetime

__version__ = "1.0.0"
__all__ = ["PythonVectorDB", "PythonVectorDBConfig"]

# Configuration constants
SEARCH_HISTORY_SIZE = 100
DELETED_THRESHOLD = 1000  # Trigger cleanup after 1000 deletions


@njit(fastmath=True, cache=True, parallel=True, nogil=True)
def cosine_similarity_int8(query: np.ndarray, vectors_int8: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query vector and Int8 vectors.
    
    Args:
        query: Query vector (float32)
        vectors_int8: Int8 quantized vectors
        
    Returns:
        Similarity scores for each vector
    """
    n = vectors_int8.shape[0]
    dim = vectors_int8.shape[1]
    similarities = np.empty(n, dtype=np.float32)
    
    query_norm_sq = 0.0
    for j in range(dim):
        query_norm_sq += query[j] * query[j]
    query_norm = np.sqrt(query_norm_sq)
    
    if query_norm < 1e-10:
        similarities.fill(0.0)
        return similarities
    
    scale = 1.0 / 127.0
    
    for i in prange(n):
        dot = 0.0
        vec_norm_sq = 0.0
        
        for j in range(dim):
            v = vectors_int8[i, j] * scale
            dot += query[j] * v
            vec_norm_sq += v * v
        
        vec_norm = np.sqrt(vec_norm_sq)
        if vec_norm > 1e-10:
            similarities[i] = dot / (query_norm * vec_norm)
        else:
            similarities[i] = 0.0
    
    return similarities


@njit(fastmath=True, cache=True, parallel=True, nogil=True)
def normalize_batch(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length.
    
    Args:
        vectors: Input vectors (float32)
        
    Returns:
        Normalized vectors
    """
    n = vectors.shape[0]
    dim = vectors.shape[1]
    normalized = np.empty_like(vectors)
    
    for i in prange(n):
        norm_sq = 0.0
        for j in range(dim):
            norm_sq += vectors[i, j] * vectors[i, j]
        
        norm = np.sqrt(norm_sq)
        if norm > 1e-10:
            inv_norm = 1.0 / norm
            for j in range(dim):
                normalized[i, j] = vectors[i, j] * inv_norm
        else:
            for j in range(dim):
                normalized[i, j] = 0.0
    
    return normalized


@njit(fastmath=True, cache=True, parallel=True, nogil=True)
def quantize_batch(normalized: np.ndarray) -> np.ndarray:
    """
    Quantize normalized vectors to Int8.
    
    Args:
        normalized: Normalized vectors (float32)
        
    Returns:
        Quantized vectors (int8)
    """
    n = normalized.shape[0]
    dim = normalized.shape[1]
    quantized = np.empty((n, dim), dtype=np.int8)
    
    for i in prange(n):
        for j in range(dim):
            val = normalized[i, j] * 127.0
            if val > 127.0:
                quantized[i, j] = 127
            elif val < -128.0:
                quantized[i, j] = -128
            else:
                quantized[i, j] = np.int8(val)
    
    return quantized


@njit(fastmath=True, cache=True)
def top_k_selection(similarities: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select top k indices with highest similarity scores.
    
    Args:
        similarities: Similarity scores
        k: Number of top results to return
        
    Returns:
        Tuple of (indices, scores) sorted by descending score
    """
    n = len(similarities)
    if k >= n:
        idx = np.argsort(similarities)[::-1]
        return idx, similarities[idx]
    
    idx = np.argpartition(similarities, -k)[-k:]
    top_sims = similarities[idx]
    sorted_order = np.argsort(top_sims)[::-1]
    
    return idx[sorted_order], top_sims[sorted_order]


class PythonVectorDBConfig:
    """
    Configuration for PythonVectorDB.
    
    Attributes:
        dimension: Vector dimension
        initial_capacity: Initial capacity for vector storage
    """
    
    def __init__(
        self,
        dimension: int = 128,
        initial_capacity: int = 10000
    ):
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")
        if initial_capacity <= 0:
            raise ValueError(f"Initial capacity must be positive, got {initial_capacity}")
            
        self.dimension = dimension
        self.initial_capacity = initial_capacity


class PythonVectorDB:
    """
    Production vector database with Int8 quantization and lazy deletion.
    
    This database stores vectors in quantized Int8 format for memory efficiency
    while maintaining high search accuracy through cosine similarity.
    
    Features:
    - Lazy deletion with threshold-based compaction
    - Binary save/load with NumPy compression
    - Thread-safe operations
    
    Example:
        db = PythonVectorDB(dimension=128)
        vectors = np.random.randn(1000, 128).astype(np.float32)
        db.add_vectors(vectors)
        query = np.random.randn(128).astype(np.float32)
        results = db.search(query, k=10)
    """
    
    def __init__(
        self, 
        dimension: int = 128, 
        initial_capacity: int = 10000,
        config: Optional[PythonVectorDBConfig] = None
    ):
        """
        Initialize PythonVectorDB.
        
        Args:
            dimension: Vector dimension (must be positive)
            initial_capacity: Initial storage capacity (must be positive)
            config: Optional configuration object (overrides dimension and initial_capacity)
            
        Raises:
            ValueError: If dimension or capacity is invalid
        """
        if config:
            self.dimension = config.dimension
            self.capacity = config.initial_capacity
        else:
            if dimension <= 0:
                raise ValueError(f"Dimension must be positive, got {dimension}")
            if initial_capacity <= 0:
                raise ValueError(f"Initial capacity must be positive, got {initial_capacity}")
            self.dimension = dimension
            self.capacity = initial_capacity
        
        self.vectors = np.empty((self.capacity, self.dimension), dtype=np.int8, order='C')
        self.vector_count = 0
        self.deleted_count = 0
        
        self.vector_ids: List[str] = []
        self.id_to_index: Dict[str, int] = {}
        self.metadata: Dict[str, Dict] = {}
        
        self.lock = threading.RLock()
        self.search_times = deque(maxlen=SEARCH_HISTORY_SIZE)
        self.created_at = datetime.now()
    
    def _ensure_capacity(self, needed: int) -> None:
        """
        Grow storage capacity if needed.
        
        Args:
            needed: Required capacity
        """
        if needed <= self.capacity:
            return
        
        new_capacity = max(needed, int(self.capacity * 1.5))
        new_vectors = np.empty((new_capacity, self.dimension), dtype=np.int8, order='C')
        
        if self.vector_count > 0:
            new_vectors[:self.vector_count] = self.vectors[:self.vector_count]
        
        self.vectors = new_vectors
        self.capacity = new_capacity
    
    def add_vectors(
        self, 
        vectors: np.ndarray, 
        vector_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Add vectors to the database.
        
        Args:
            vectors: Array of shape (n, dimension) or (dimension,)
            vector_ids: Optional list of unique vector IDs
            metadata: Optional list of metadata dictionaries
            
        Raises:
            ValueError: If dimension mismatch, invalid values, or duplicate IDs
        """
        vectors = np.asarray(vectors, dtype=np.float32, order='C')
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        n_new = len(vectors)
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}"
            )
        
        if not np.isfinite(vectors).all():
            raise ValueError("Vectors contain NaN or infinite values")
        
        if vector_ids is None:
            start = len(self.vector_ids)
            vector_ids = [f"vec_{start+i}" for i in range(n_new)]
        
        if len(vector_ids) != n_new:
            raise ValueError(
                f"vector_ids length ({len(vector_ids)}) != vectors length ({n_new})"
            )
        
        if len(vector_ids) != len(set(vector_ids)):
            raise ValueError("Duplicate vector IDs in input")
        
        with self.lock:
            existing = set(self.id_to_index.keys())
            duplicates = set(vector_ids) & existing
            if duplicates:
                raise ValueError(f"Vector IDs already exist: {duplicates}")
            
            self._ensure_capacity(self.vector_count + n_new)
            
            normalized = normalize_batch(vectors)
            quantized = quantize_batch(normalized)
            
            start_idx = self.vector_count
            self.vectors[start_idx:start_idx + n_new] = quantized
            
            self.vector_ids.extend(vector_ids)
            for i, vid in enumerate(vector_ids):
                self.id_to_index[vid] = start_idx + i
            
            if metadata:
                if len(metadata) != n_new:
                    raise ValueError(
                        f"metadata length ({len(metadata)}) != vectors length ({n_new})"
                    )
                for vid, meta in zip(vector_ids, metadata):
                    self.metadata[vid] = meta
            
            self.vector_count += n_new
    
    def search(
        self, 
        query: np.ndarray, 
        k: int = 10,
        filter_fn: Optional[Callable[[str, Dict], bool]] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector of shape (dimension,)
            k: Number of results to return (must be positive)
            filter_fn: Optional filter function(vector_id, metadata) -> bool
            
        Returns:
            List of (vector_id, score, metadata) tuples sorted by descending score
            
        Raises:
            ValueError: If query dimension mismatch or invalid k
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        start = time.perf_counter()
        
        query = np.asarray(query, dtype=np.float32).flatten()
        
        if len(query) != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, got {len(query)}"
            )
        
        if not np.isfinite(query).all():
            raise ValueError("Query contains NaN or infinite values")
        
        with self.lock:
            if self.vector_count == 0:
                return []
            
            k_actual = min(k, self.vector_count)
            
            if filter_fn:
                valid_mapping = []
                for i in range(self.vector_count):
                    vid = self.vector_ids[i]
                    if filter_fn(vid, self.metadata.get(vid, {})):
                        valid_mapping.append((i, vid))
                
                if not valid_mapping:
                    return []
                
                valid_indices = [idx for idx, _ in valid_mapping]
                active = self.vectors[valid_indices]
                sims = cosine_similarity_int8(query, active)
                
                indices, scores = top_k_selection(sims, min(k_actual, len(sims)))
                
                results = []
                for filtered_idx, score in zip(indices, scores):
                    original_idx, vid = valid_mapping[filtered_idx]
                    results.append((vid, float(score), self.metadata.get(vid, {})))
            else:
                active = self.vectors[:self.vector_count]
                sims = cosine_similarity_int8(query, active)
                
                indices, scores = top_k_selection(sims, k_actual)
                
                results = []
                for idx, score in zip(indices, scores):
                    vid = self.vector_ids[idx]
                    results.append((vid, float(score), self.metadata.get(vid, {})))
            
            elapsed = time.perf_counter() - start
            self.search_times.append(elapsed)
            
            return results
    
    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """
        Retrieve a vector by ID (dequantized to Float32).
        
        Args:
            vector_id: Vector ID to retrieve
            
        Returns:
            Vector as Float32 array, or None if not found
        """
        with self.lock:
            idx = self.id_to_index.get(vector_id)
            if idx is None:
                return None
            
            if idx >= self.vector_count or idx < 0:
                return None
            
            quantized = self.vectors[idx]
            return quantized.astype(np.float32) / 127.0
    
    def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector by ID using lazy deletion.
        
        This operation marks the vector as deleted by zero-filling it.
        The actual storage compaction happens only when the deleted
        count exceeds DELETED_THRESHOLD.
        
        Args:
            vector_id: Vector ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self.lock:
            if vector_id not in self.id_to_index:
                return False
            
            idx = self.id_to_index[vector_id]
            
            # Zero-fill the vector data (lazy delete)
            self.vectors[idx].fill(0)
            
            # Remove from mappings
            del self.id_to_index[vector_id]
            self.metadata.pop(vector_id, None)
            
            try:
                self.vector_ids.remove(vector_id)
            except ValueError:
                pass
            
            self.deleted_count += 1
            
            # Trigger cleanup if threshold exceeded
            if self.deleted_count >= DELETED_THRESHOLD:
                self._compact_storage()
            
            return True
    
    def _compact_storage(self) -> None:
        """
        Compact storage by removing deleted vectors.
        
        This rebuilds the vector array, removing any gaps from deletions.
        """
        if self.vector_count == 0:
            return
        
        active_count = len(self.id_to_index)
        
        if active_count == 0:
            self.vectors.fill(0)
            self.vector_ids.clear()
            self.id_to_index.clear()
            self.metadata.clear()
            self.vector_count = 0
            self.deleted_count = 0
            return
        
        if active_count == self.vector_count:
            self.deleted_count = 0
            return
        
        write_pos = 0
        active_ids = []
        new_id_to_index = {}
        
        for vid in self.vector_ids:
            if vid in self.id_to_index:
                old_idx = self.id_to_index[vid]
                
                if old_idx != write_pos:
                    self.vectors[write_pos] = self.vectors[old_idx]
                
                active_ids.append(vid)
                new_id_to_index[vid] = write_pos
                write_pos += 1
        
        if write_pos < self.vector_count:
            self.vectors[write_pos:self.vector_count].fill(0)
        
        self.vector_ids = active_ids
        self.id_to_index = new_id_to_index
        self.vector_count = active_count
        self.deleted_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics and performance metrics.
        
        Returns:
            Dictionary containing version, counts, memory usage, and performance stats
        """
        with self.lock:
            vector_memory = self.vector_count * self.dimension
            
            ids_memory = sys.getsizeof(self.vector_ids)
            for vid in self.vector_ids:
                ids_memory += sys.getsizeof(vid)
            
            metadata_memory = sys.getsizeof(self.metadata)
            for k, v in self.metadata.items():
                metadata_memory += sys.getsizeof(k) + sys.getsizeof(v)
            
            total_memory_bytes = vector_memory + ids_memory + metadata_memory
            
            base_stats = {
                'version': __version__,
                'vectors': self.vector_count,
                'capacity': self.capacity,
                'dimension': self.dimension,
                'deleted_count': self.deleted_count,
                'memory_mb': f"{total_memory_bytes / 1e6:.3f}",
                'utilization': f"{100 * self.vector_count / self.capacity:.1f}%"
            }
            
            if self.search_times:
                times_ms = np.array(self.search_times) * 1000
                avg_ms = np.mean(times_ms)
                qps = 1000 / avg_ms if avg_ms > 0 else 0
                
                base_stats.update({
                    'avg_ms': f"{avg_ms:.2f}",
                    'p50_ms': f"{np.percentile(times_ms, 50):.2f}",
                    'p95_ms': f"{np.percentile(times_ms, 95):.2f}",
                    'p99_ms': f"{np.percentile(times_ms, 99):.2f}",
                    'qps': f"{qps:.0f}"
                })
            
            return base_stats
    
    def save(self, filepath: str) -> None:
        """
        Save database to disk using compressed NumPy format.
        
        This is much faster than JSON and produces smaller files.
        
        Args:
            filepath: Path to save file (.npz recommended)
        """
        with self.lock:
            # Compact before saving for efficiency
            if self.deleted_count > 0:
                self._compact_storage()
            
            # Save using compressed NumPy format
            np.savez_compressed(
                filepath,
                version=__version__,
                dimension=self.dimension,
                vectors=self.vectors[:self.vector_count],
                vector_ids=np.array(self.vector_ids),
                id_to_index={k: v for k, v in self.id_to_index.items()},
                metadata=self.metadata,
                vector_count=self.vector_count,
                deleted_count=self.deleted_count
            )
    
    @classmethod
    def load(cls, filepath: str) -> 'PythonVectorDB':
        """
        Load database from disk using NumPy format.
        
        Args:
            filepath: Path to database file
            
        Returns:
            Loaded PythonVectorDB instance
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is invalid or data is corrupted
        """
        try:
            data = np.load(filepath, allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Database file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Invalid database file: {e}")
        
        # Extract data
        version = str(data['version'])
        dimension = int(data['dimension'])
        vectors = data['vectors']
        vector_ids = data['vector_ids'].tolist()
        id_to_index = data['id_to_index'].item()
        metadata = data['metadata'].item()
        vector_count = int(data['vector_count'])
        deleted_count = int(data['deleted_count'])
        
        # Validate data
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"Invalid dimension: {dimension}")
        
        if not isinstance(vector_count, int) or vector_count < 0:
            raise ValueError(f"Invalid vector_count: {vector_count}")
        
        if vectors.shape[0] != vector_count:
            raise ValueError("Vector count mismatch")
        
        if vectors.shape[1] != dimension:
            raise ValueError("Vector dimension mismatch")
        
        if len(vector_ids) != vector_count:
            raise ValueError("Vector IDs count mismatch")
        
        # Create instance
        db = cls(dimension=dimension, initial_capacity=vector_count)
        
        if vector_count > db.capacity:
            db._ensure_capacity(vector_count)
        
        # Load data
        db.vectors[:vector_count] = vectors
        db.vector_ids = vector_ids
        db.id_to_index = {str(k): int(v) for k, v in id_to_index.items()}
        db.metadata = metadata
        db.vector_count = vector_count
        db.deleted_count = deleted_count
        
        return db
    
    def __len__(self) -> int:
        """Return number of vectors in database."""
        return self.vector_count
    
    def __repr__(self) -> str:
        """String representation of database."""
        return (
            f"PythonVectorDB(vectors={self.vector_count:,}, "
            f"dimension={self.dimension}, "
            f"capacity={self.capacity:,}, "
            f"deleted={self.deleted_count})"
        )
