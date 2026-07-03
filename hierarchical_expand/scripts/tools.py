import os
from typing import List, Dict, Any, Optional

try:
    from hierarchical_index import HierarchicalIndex, Node
    from caching import KVCache, SearchCache
except ModuleNotFoundError:
    try:
        from .hierarchical_index import HierarchicalIndex, Node
        from .caching import KVCache, SearchCache
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from hierarchical_index import HierarchicalIndex, Node
        from caching import KVCache, SearchCache

_INDEX_INSTANCE: Optional[HierarchicalIndex] = None
_KV_CACHE_INSTANCE: Optional[KVCache] = None
_SEARCH_CACHE_INSTANCE: Optional[SearchCache] = None

def get_task_scratch_dir() -> Optional[str]:
    # 1. Check environment variable override
    env_dir = os.environ.get("TASK_SCRATCH_DIR")
    if env_dir:
        return env_dir

    # 2. Check sys.argv[0] to see if a pipeline script is being executed
    import sys
    if sys.argv:
        main_script = os.path.abspath(sys.argv[0])
        if "system_scratch" in main_script:
            return os.path.dirname(main_script).replace("\\", "/")

    # 3. Inspect stack frames for SUT runner (serve_query)
    import inspect
    try:
        for frame_info in inspect.stack():
            if frame_info.function == "serve_query":
                locals_dict = frame_info.frame.f_locals
                self_obj = locals_dict.get("self")
                if self_obj:
                    for attr in ["question_output_dir", "question_intermediate_dir", "output_dir"]:
                        val = getattr(self_obj, attr, None)
                        if val and isinstance(val, str):
                            if attr == "output_dir":
                                name = getattr(self_obj, "name", "")
                                query_id = locals_dict.get("query_id", "default-0")
                                val = os.path.join(val, name, query_id)
                            return os.path.abspath(val).replace("\\", "/")
                
                # Stack fallback using SUT class name and query_id
                query_id = locals_dict.get("query_id")
                if query_id and self_obj:
                    sut_name = self_obj.__class__.__name__
                    fallback_path = os.path.join("system_scratch", sut_name, query_id)
                    return os.path.abspath(fallback_path).replace("\\", "/")
    except Exception:
        pass

    return None

def get_hierarchical_index(dataset_dir: str) -> HierarchicalIndex:
    global _INDEX_INSTANCE
    normalized_dir = os.path.abspath(os.path.normpath(dataset_dir))
    if _INDEX_INSTANCE is None or _INDEX_INSTANCE.dataset_dir != normalized_dir:
        dataset_parent = os.path.dirname(normalized_dir)
        dataset_cache = os.path.join(dataset_parent, "hierarchical_index_cache.json")
        if os.path.exists(dataset_cache):
            cache_file = dataset_cache
        else:
            scratch_dir = get_task_scratch_dir()
            if scratch_dir and os.path.exists(scratch_dir):
                cache_file = os.path.join(scratch_dir, "hierarchical_index_cache.json")
            else:
                skills_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                cache_file = os.path.join(skills_root, "hierarchical_index", "hierarchical_index_cache.json")
        _INDEX_INSTANCE = HierarchicalIndex(dataset_dir=normalized_dir, cache_file=cache_file)
    return _INDEX_INSTANCE

def get_kv_cache_store() -> KVCache:
    global _KV_CACHE_INSTANCE
    if _KV_CACHE_INSTANCE is None:
        scratch_dir = get_task_scratch_dir()
        if scratch_dir and os.path.exists(scratch_dir):
            cache_file = os.path.join(scratch_dir, "kv_cache.json")
        else:
            skills_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_file = os.path.join(skills_root, "hierarchical_index", "kv_cache.json")
        _KV_CACHE_INSTANCE = KVCache(cache_file=cache_file)
    return _KV_CACHE_INSTANCE

def get_search_cache_store() -> SearchCache:
    global _SEARCH_CACHE_INSTANCE
    if _SEARCH_CACHE_INSTANCE is None:
        scratch_dir = get_task_scratch_dir()
        if scratch_dir and os.path.exists(scratch_dir):
            cache_file = os.path.join(scratch_dir, "search_cache.json")
        else:
            skills_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_file = os.path.join(skills_root, "hierarchical_index", "search_cache.json")
        _SEARCH_CACHE_INSTANCE = SearchCache(cache_file=cache_file)
    return _SEARCH_CACHE_INSTANCE

def search(query: str, dataset_dir: str, scope: str = "data", granularity: int = 1, top_k: int = 5) -> List[str]:
    """
    Search matches within a dataset hierarchy scope at a given granularity.
    """
    index = get_hierarchical_index(dataset_dir)
    nodes = index.search(query=query, scope=scope, granularity=granularity, top_k=top_k)
    return [node.path for node in nodes]

def summarize(scope: str, dataset_dir: str, top_k: int = 5) -> List[str]:
    """
    Summarizes structure of a file/table node or summarizes values of a column/field node.
    """
    index = get_hierarchical_index(dataset_dir)
    return index.summarize(scope=scope, top_k=top_k)

def expand(scope: str, dataset_dir: str, granularity: int = 1, sample_size: int = 5) -> List[str]:
    """
    Lists child nodes under a scope.
    """
    index = get_hierarchical_index(dataset_dir)
    nodes = index.expand(scope=scope, granularity=granularity, sample_size=sample_size)
    return [node.path for node in nodes]

def get_kv(messages: List[Dict[str, str]]) -> Optional[str]:
    """
    Retrieves cached assistant response matching conversational prefix messages.
    """
    kv_store = get_kv_cache_store()
    return kv_store.get(messages)

def set_kv(messages: List[Dict[str, str]], response: str):
    """
    Saves generated assistant response for the corresponding conversational history.
    """
    kv_store = get_kv_cache_store()
    kv_store.set(messages, response)

def search_with_cache(query: str, dataset_dir: str, scope: str = "data", granularity: int = 1, top_k: int = 5) -> List[str]:
    """
    Search helper that uses standard search caching.
    """
    search_cache = get_search_cache_store()
    canonical_query = search_cache.canonicalize(query)
    
    cached_results = search_cache.get(canonical_query)
    if cached_results is not None:
        return cached_results
        
    results = search(query=canonical_query, dataset_dir=dataset_dir, scope=scope, granularity=granularity, top_k=top_k)
    search_cache.set(canonical_query, results)
    return results

export_tools = [search, summarize, expand, get_kv, set_kv, search_with_cache]
