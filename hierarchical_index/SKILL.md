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
import sys

# Add this skill's scripts directory to the Python path.
# The path below matches the skill's installed location.
sys.path.insert(0, ".opencode/skills/hierarchical_index/scripts")

from hierarchical_index import HierarchicalIndex

# Initialize the index (builds or loads from cache)
index = HierarchicalIndex(dataset_dir="data/archeology/input")

# Expand: list tables (like ls)
tables = index.expand(scope="data", granularity=1)
for t in tables:
    print(t)

# Search: find relevant columns (hybrid BM25 + semantic)
results = index.search(query="column description keywords", scope="data", granularity=2, top_k=10)
for r in results:
    print(r)

# Summarize: inspect column values (like cat)
values = index.summarize(scope="data::tablename::columnname", top_k=5)
for v in values:
    print(v)
```
