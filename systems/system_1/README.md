# Document-Centric Hierarchical Search Index and Caching Skill (system_1)

This repository implements `system_1`, structured as a self-contained agent skill for OpenCode.

## Overview

1. **Hierarchical Indexing Engine**:
   - Organizes raw spreadsheets (CSV/Excel) into a logical hierarchy: `data` (root) -> `data::[table_name]` (file node) -> `data::[table_name]::[column_name]` (field node).
   - API methods:
     - `search(query, dataset_dir, scope, granularity, top_k)`: Performs semantic/lexical search over matched hierarchy nodes.
     - `summarize(scope, dataset_dir, top_k)`: Generates column/cell value summaries.
     - `expand(scope, dataset_dir, granularity, sample_size)`: Lists child nodes of a scope.

2. **Advanced Caching**:
   - **KV Cache**: Caches conversational histories using prefix matching.
   - **Search Cache**: Maps variations of queries to standard canonical queries using semantic similarity and standardizes query representation via `deepseek-v4-flash`, caching retrieved search results.

## Usage

Exposed functions can be imported from `systems.system_1.scripts.tools`:

```python
from systems.system_1.scripts.tools import search, summarize, expand, search_with_cache, get_kv, set_kv
```
