---
name: hierarchical_search
description: A document-centric hierarchical search and caching skill. Provides hybrid (BM25S lexical + vector similarity) semantic search over a data lake of CSV and Excel tables, along with conversational KV caching and query canonicalization search caching.
license: MIT
metadata:
  author: Fadhilah
  version: "1.0"
---

# Hierarchical Search

A document-centric hierarchical search and caching skill. Provides hybrid (BM25S lexical + vector similarity) semantic search over a data lake of CSV and Excel tables, along with conversational KV caching and query canonicalization search caching.

## Usage

```python
import sys
sys.path.insert(0, ".opencode/skills/hierarchical_search/scripts")

from hierarchical_index import HierarchicalIndex

index = HierarchicalIndex(dataset_dir="data/archeology/input")

# Search across tables and columns
results = index.search(query="find relevant columns", scope="data", granularity=2, top_k=10)
for r in results:
    print(r)
```
