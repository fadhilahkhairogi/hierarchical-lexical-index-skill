import os
from typing import List, Dict, Any, Optional
from systems.system_1.scripts.hierarchical_index import HierarchicalIndex, Node
from systems.system_1.scripts.caching import KVCache, SearchCache

_INDEX_INSTANCE: Optional[HierarchicalIndex] = None
_KV_CACHE_INSTANCE: Optional[KVCache] = None
_SEARCH_CACHE_INSTANCE: Optional[SearchCache] = None

def get_hierarchical_index(dataset_dir: str) -> HierarchicalIndex:
    global _INDEX_INSTANCE
    normalized_dir = os.path.normpath(dataset_dir)
    if _INDEX_INSTANCE is None or _INDEX_INSTANCE.dataset_dir != normalized_dir:
        cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hierarchical_index_cache.json")
        _INDEX_INSTANCE = HierarchicalIndex(dataset_dir=normalized_dir, cache_file=cache_file)
    return _INDEX_INSTANCE

def get_kv_cache_store() -> KVCache:
    global _KV_CACHE_INSTANCE
    if _KV_CACHE_INSTANCE is None:
        cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kv_cache.json")
        _KV_CACHE_INSTANCE = KVCache(cache_file=cache_file)
    return _KV_CACHE_INSTANCE

def get_search_cache_store() -> SearchCache:
    global _SEARCH_CACHE_INSTANCE
    if _SEARCH_CACHE_INSTANCE is None:
        cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "search_cache.json")
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
