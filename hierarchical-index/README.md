# Document-Centric Hierarchical Search Index and Caching Skill

This repository implements agent skill for OpenCode.

## Overview

1. **Hierarchical Indexing Engine**:
   - Organizes raw spreadsheets (CSV/Excel) into a logical hierarchy: `data` (root) -> `data::[table_name]` (file node) -> `data::[table_name]::[column_name]` (field node).
   - API methods:
     - `search(query, dataset_dir, scope, granularity, top_k)`: Performs search over matched hierarchy nodes.
     - `search_without_cache(query, dataset_dir, scope, granularity, top_k)`: Performs search over matched hierarchy nodes.
     - `summarize(scope, dataset_dir, top_k)`: Generates column/cell value summaries.
     - `expand(scope, dataset_dir, granularity, sample_size)`: Lists child nodes of a scope.

2. **Advanced Caching**:
   - **Search Cache**: Maps variations of queries to standard canonical queries using semantic similarity (`embeddinggemma`), caching retrieved search results. On Cache Hit, prepends formal metadata header `"[Cache Hit: canonical_query='...']"`.

## Usage

Exposed functions can be imported locally:

```python
from tools import search, summarize, expand, search_without_cache
```
