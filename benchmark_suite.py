#!/usr/bin/env python3
"""
PythonVectorDB Complete Benchmark Suite
=================================

Comprehensive performance testing including:
- Insert performance (various batch sizes)
- Search performance (various database sizes)
- Memory usage analysis
- Concurrent operations
- Lazy deletion performance
- Binary save/load performance
"""

import numpy as np
import time
import threading
import psutil
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Any

from pythonvectordb import PythonVectorDB


class BenchmarkSuite:
    """Comprehensive benchmark suite for PythonVectorDB."""
    
    def __init__(self):
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "numpy_version": np.__version__,
                "cpu_count": os.cpu_count(),
                "system": os.name
            },
            "tests": {}
        }
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def benchmark_insert_performance(self) -> Dict[str, Any]:
        """Benchmark insert performance across different batch sizes."""
        print("📊 Benchmark: Insert Performance")
        print("-" * 40)
        
        results = {}
        dimension = 128
        
        for batch_size in [1, 10, 100, 1000, 10000]:
            db = PythonVectorDB(dimension=dimension)
            
            # Generate test vectors
            vectors = np.random.randn(batch_size, dimension).astype(np.float32)
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            
            # Benchmark insertion
            start_memory = self.get_memory_usage()
            start_time = time.perf_counter()
            
            db.add_vectors(vectors)
            
            end_time = time.perf_counter()
            end_memory = self.get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            vectors_per_sec = batch_size / duration if duration > 0 else float('inf')
            memory_per_vector = (memory_delta * 1024) / batch_size if batch_size > 0 else 0  # KB
            
            results[f"batch_{batch_size}"] = {
                "duration_ms": duration * 1000,
                "vectors_per_sec": vectors_per_sec,
                "memory_per_vector_kb": memory_per_vector,
                "total_memory_mb": end_memory
            }
            
            print(f"  Batch {batch_size:5d}: {duration*1000:6.2f}ms, {vectors_per_sec:8.0f} vec/s, {memory_per_vector:6.2f}KB/vec")
        
        return results
    
    def benchmark_search_performance(self) -> Dict[str, Any]:
        """Benchmark search performance across different database sizes."""
        print("\n🔍 Benchmark: Search Performance")
        print("-" * 40)
        
        results = {}
        dimension = 128
        
        for db_size in [1000, 5000, 10000, 50000, 100000]:
            # Create and populate database
            db = PythonVectorDB(dimension=dimension, initial_capacity=db_size)
            
            vectors = np.random.randn(db_size, dimension).astype(np.float32)
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            db.add_vectors(vectors)
            
            # Test search performance
            query = np.random.randn(dimension).astype(np.float32)
            query = query / np.linalg.norm(query)
            
            # Warm up
            for _ in range(10):
                db.search(query, k=10)
            
            # Benchmark
            num_queries = 100
            start_time = time.perf_counter()
            
            for _ in range(num_queries):
                db.search(query, k=10)
            
            end_time = time.perf_counter()
            avg_time = (end_time - start_time) / num_queries
            qps = 1 / avg_time if avg_time > 0 else 0
            
            results[f"db_size_{db_size}"] = {
                "avg_search_ms": avg_time * 1000,
                "qps": qps,
                "vectors": db_size
            }
            
            print(f"  DB {db_size:6d}: {avg_time*1000:6.3f}ms avg, {qps:6.0f} QPS")
        
        return results
    
    def benchmark_memory_scaling(self) -> Dict[str, Any]:
        """Benchmark memory usage scaling."""
        print("\n💾 Benchmark: Memory Scaling")
        print("-" * 40)
        
        results = {}
        dimension = 128
        
        for size in [10000, 50000, 100000, 500000]:
            db = PythonVectorDB(dimension=dimension, initial_capacity=size)
            
            start_memory = self.get_memory_usage()
            
            vectors = np.random.randn(size, dimension).astype(np.float32)
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            db.add_vectors(vectors)
            
            end_memory = self.get_memory_usage()
            memory_used = end_memory - start_memory
            memory_per_vector = (memory_used * 1024 * 1024) / size  # bytes per vector
            
            results[f"size_{size}"] = {
                "total_memory_mb": end_memory,
                "memory_delta_mb": memory_used,
                "memory_per_vector_bytes": memory_per_vector
            }
            
            print(f"  {size:6d} vectors: {memory_used:6.1f}MB, {memory_per_vector:6.1f} bytes/vec")
        
        return results
    
    def benchmark_concurrent_operations(self) -> Dict[str, Any]:
        """Benchmark concurrent operations."""
        print("\n🔄 Benchmark: Concurrent Operations")
        print("-" * 40)
        
        results = {}
        dimension = 128
        
        # Setup database
        db = PythonVectorDB(dimension=dimension)
        vectors = np.random.randn(10000, dimension).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        db.add_vectors(vectors)
        
        # Concurrent search test
        def search_worker(worker_id: int, num_queries: int) -> Dict:
            query = np.random.randn(dimension).astype(np.float32)
            query = query / np.linalg.norm(query)
            
            times = []
            for _ in range(num_queries):
                start = time.perf_counter()
                db.search(query, k=10)
                times.append(time.perf_counter() - start)
            
            return {
                "worker_id": worker_id,
                "queries": num_queries,
                "avg_time": np.mean(times),
                "total_time": sum(times)
            }
        
        # Run concurrent searches
        num_workers = 10
        queries_per_worker = 50
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(search_worker, i, queries_per_worker)
                for i in range(num_workers)
            ]
            
            worker_results = []
            for future in as_completed(futures):
                worker_results.append(future.result())
        
        total_time = time.perf_counter() - start_time
        total_queries = sum(r["queries"] for r in worker_results)
        overall_qps = total_queries / total_time
        
        results["concurrent_search"] = {
            "workers": num_workers,
            "total_queries": total_queries,
            "total_time_sec": total_time,
            "overall_qps": overall_qps,
            "worker_results": worker_results
        }
        
        print(f"  {num_workers} workers: {overall_qps:.0f} QPS ({total_queries} queries in {total_time:.2f}s)")
        
        return results
    
    def benchmark_lazy_deletion(self) -> Dict[str, Any]:
        """Benchmark lazy deletion performance."""
        print("\n🗑️  Benchmark: Lazy Deletion")
        print("-" * 40)
        
        results = {}
        dimension = 128
        
        # Setup database
        db = PythonVectorDB(dimension=dimension)
        vectors = np.random.randn(10000, dimension).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        db.add_vectors(vectors)
        
        # Test lazy deletion (before compaction threshold)
        lazy_deletes = []
        start_time = time.perf_counter()
        
        for i in range(500):  # Less than threshold
            db.delete_vector(f"vec_{i}")
        
        lazy_time = time.perf_counter() - start_time
        lazy_deletes.append({
            "num_deletes": 500,
            "total_time_sec": lazy_time,
            "avg_time_ms": (lazy_time / 500) * 1000
        })
        
        print(f"  Lazy delete (500): {lazy_time*1000:.2f}ms total, {(lazy_time/500)*1000:.3f}ms avg")
        
        # Test deletion with compaction (trigger threshold)
        start_time = time.perf_counter()
        
        for i in range(500, 1500):  # Trigger compaction
            db.delete_vector(f"vec_{i}")
        
        compact_time = time.perf_counter() - start_time
        compact_deletes = {
            "num_deletes": 1000,
            "total_time_sec": compact_time,
            "avg_time_ms": (compact_time / 1000) * 1000,
            "triggered_compaction": True
        }
        
        print(f"  Delete with compaction (1000): {compact_time*1000:.2f}ms total, {(compact_time/1000)*1000:.3f}ms avg")
        
        results["lazy_deletion"] = {
            "lazy_deletes": lazy_deletes,
            "compact_deletes": compact_deletes,
            "final_stats": db.get_stats()
        }
        
        return results
    
    def benchmark_save_load(self) -> Dict[str, Any]:
        """Benchmark binary save/load performance."""
        print("\n💾 Benchmark: Save/Load Performance")
        print("-" * 40)
        
        results = {}
        dimension = 128
        
        for size in [1000, 10000, 50000]:
            # Create database
            db = PythonVectorDB(dimension=dimension)
            vectors = np.random.randn(size, dimension).astype(np.float32)
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            vector_ids = [f"vec_{i}" for i in range(size)]
            metadata = [{"index": i} for i in range(size)]
            db.add_vectors(vectors, vector_ids, metadata)
            
            # Benchmark save
            test_file = f"benchmark_{size}.npz"
            start_time = time.perf_counter()
            db.save(test_file)
            save_time = time.perf_counter() - start_time
            
            # Get file size
            file_size_mb = os.path.getsize(test_file) / 1024 / 1024
            
            # Benchmark load
            start_time = time.perf_counter()
            loaded_db = PythonVectorDB.load(test_file)
            load_time = time.perf_counter() - start_time
            
            # Verify integrity
            query = np.random.randn(dimension).astype(np.float32)
            query = query / np.linalg.norm(query)
            original_results = db.search(query, k=10)
            loaded_results = loaded_db.search(query, k=10)
            
            integrity_check = len(original_results) == len(loaded_results)
            
            results[f"size_{size}"] = {
                "save_time_sec": save_time,
                "load_time_sec": load_time,
                "file_size_mb": file_size_mb,
                "integrity_check": integrity_check,
                "vectors": size
            }
            
            print(f"  {size:5d} vectors: save {save_time:.3f}s, load {load_time:.3f}s, {file_size_mb:.1f}MB")
            
            # Cleanup
            os.remove(test_file)
        
        return results
    
    def run_complete_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        print("🚀 PythonVectorDB Complete Benchmark Suite")
        print("=" * 50)
        
        # Run all benchmarks
        self.results["tests"]["insert_performance"] = self.benchmark_insert_performance()
        self.results["tests"]["search_performance"] = self.benchmark_search_performance()
        self.results["tests"]["memory_scaling"] = self.benchmark_memory_scaling()
        self.results["tests"]["concurrent_operations"] = self.benchmark_concurrent_operations()
        self.results["tests"]["lazy_deletion"] = self.benchmark_lazy_deletion()
        self.results["tests"]["save_load_performance"] = self.benchmark_save_load()
        
        # Add summary
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self):
        """Generate benchmark summary."""
        print("\n📊 BENCHMARK SUMMARY")
        print("=" * 50)
        
        # Insert performance summary
        insert_results = self.results["tests"]["insert_performance"]
        best_insert = max(insert_results.values(), key=lambda x: x["vectors_per_sec"])
        print(f"📈 Best Insert Performance: {best_insert['vectors_per_sec']:.0f} vectors/sec")
        
        # Search performance summary
        search_results = self.results["tests"]["search_performance"]
        best_search = max(search_results.values(), key=lambda x: x["qps"])
        print(f"🔍 Best Search QPS: {best_search['qps']:.0f}")
        
        # Memory efficiency summary
        memory_results = self.results["tests"]["memory_scaling"]
        best_memory = min(memory_results.values(), key=lambda x: x["memory_per_vector_bytes"])
        print(f"💾 Best Memory Efficiency: {best_memory['memory_per_vector_bytes']:.1f} bytes/vector")
        
        # Lazy deletion summary
        if "lazy_deletion" in self.results["tests"] and "lazy_deletes" in self.results["tests"]["lazy_deletion"]:
            lazy_results = self.results["tests"]["lazy_deletion"]["lazy_deletes"][0]
            print(f"🗑️  Lazy Delete Speed: {lazy_results['avg_time_ms']:.3f}ms per delete")
        
        # Save/load summary
        if "save_load_performance" in self.results["tests"]:
            save_load_results = self.results["tests"]["save_load_performance"]
            fastest_save = min(save_load_results.values(), key=lambda x: x["save_time_sec"])
            print(f"💾 Fastest Save: {fastest_save['save_time_sec']:.3f}s for {fastest_save['vectors']} vectors")
        
        # Concurrent performance
        if "concurrent_operations" in self.results["tests"]:
            concurrent_results = self.results["tests"]["concurrent_operations"]["concurrent_search"]
            print(f"🔄 Concurrent QPS: {concurrent_results['overall_qps']:.0f}")
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to {filename}")


def main():
    """Run the complete benchmark suite."""
    suite = BenchmarkSuite()
    results = suite.run_complete_benchmark()
    suite.save_results()
    return results


if __name__ == "__main__":
    main()
