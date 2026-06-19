---
name: hierarchical_index
description: A document-centric hierarchical search index and caching skill. Provides search (grep-like), summarize (cat-like), and expand (ls-like) functionality over a data lake of CSV and Excel tables, along with conversational KV caching and query canonicalization search caching.
license: MIT
metadata:
  author: Fadhilah
  version: "1.0"
---

# Hierarchical Index & Search Caching Skill

This skill implements a document-centric hierarchical indexing engine (DCI) and advanced search caching mechanisms to optimize agentic workflows over large data lakes.

## Key Features

1. **Hierarchical Indexing Engine**:
   - Organizes raw spreadsheets (CSV/Excel) into a logical hierarchy: `data` (root) $\rightarrow$ `data::[table_name]` (file node) $\rightarrow$ `data::[table_name]::[column_name]` (field node).
   - Exposes three Unix-inspired API methods:
     - `search(query, scope, granularity, top_k)`: Performs semantic/lexical search over matching hierarchy nodes.
     - `summarize(scope, top_k)`: Generates column/cell value summaries.
     - `expand(scope, granularity, sample_size)`: Lists child nodes of a scope.

2. **Advanced Caching**:
   - **KV Cache**: Caches conversation turns to reuse token/state generations.
   - **Search Cache**: Maps variations of user queries to a single canonical query representation, caching the search results to avoid repeated searches.

## Usage

```python
from systems.system_1.scripts.hierarchical_index import HierarchicalIndex
from systems.system_1.scripts.caching import KVCache, SearchCache

# Initialize
index = HierarchicalIndex(dataset_dir="path/to/data")
search_cache = SearchCache(model="openrouter/deepseek/deepseek-v4-flash")

# Query canonicalization + Hierarchical Search
canonical_query = search_cache.canonicalize(query="Arizona places to visit")
cached_results = search_cache.get(canonical_query)

if not cached_results:
    # Run hierarchical search
    nodes = index.search(query=canonical_query, scope="data", granularity=1, top_k=3)
    results = [str(node) for node in nodes]
    search_cache.set(canonical_query, results)
```
