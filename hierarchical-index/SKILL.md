---
name: hierarchical-index
description: A document-centric hierarchical search index and caching skill. Provides search (grep-like, BM25S lexical + vector similarity), summarize (cat-like schema/value frequencies), and expand (ls-like hierarchy navigation) functionality over a data lake of CSV and Excel tables, along with query canonicalization search caching.
license: MIT
metadata:
  author: Fadhilah
  version: "1.0"
---

# Hierarchical Index & Search Caching Skill

This skill implements a document-centric hierarchical indexing engine (DCI) and advanced search caching mechanisms to optimize agentic workflows over large data lakes.

## Key Features

1. **Hierarchical Indexing Engine**:
   - Organizes raw spreadsheets (CSV/Excel) into a logical hierarchy: `data` (root) -> `data::[table_name]` (file node) -> `data::[table_name]::[column_name]` (field node).
   - Exposes three Unix-inspired API methods:
     - `search(query, scope, granularity, top_k)`: Performs hybrid (BM25S lexical + vector similarity) semantic search over matching hierarchy nodes.
     - `summarize(scope, top_k)`: Generates schema summaries for file/table nodes or cell value occurrences/frequencies for column/field nodes.
     - `expand(scope, granularity, sample_size)`: Lists child nodes of a scope (navigation).

2. **Advanced Caching**:
   - **Search Cache**: Maps variations of user queries to a single canonical query representation, caching the search results to avoid repeated searches.

## API Usage

```python
from tools import search, summarize, expand, search_without_cache

# 1. Expand node level (ls-like navigation)
# Immediate child tables/files under root directory (depth = 1)
tables = expand(scope="data", dataset_dir="path/to/data", granularity=1, sample_size=50)

# List columns inside a specific table (depth = 2)
columns = expand(scope="data::treasury_bulletin_1941_01", dataset_dir="path/to/data", granularity=1, sample_size=50)

# List all columns across all tables directly from root (depth = 2)
all_columns = expand(scope="data", dataset_dir="path/to/data", granularity=2, sample_size=50)


# 2. Search index (grep-like semantic & lexical search)
# Search for a keyword with cache re-use
cached_results = search(query="national defense", dataset_dir="path/to/data", scope="data", granularity=2, top_k=10)

# Search for a keyword without cache
matching_tables = search_without_cache(query="national defense", dataset_dir="path/to/data", scope="data", granularity=1, top_k=7)


# 3. Summarize (cat-like schema & column value analysis)
# Get schema of a table node (lists table name, columns, and path)
table_summary = summarize(scope="data::treasury_bulletin_1941_01", dataset_dir="path/to/data", top_k=10)

# Get top-20 value occurrences in a column node
column_summary = summarize(scope="data::treasury_bulletin_1941_01::expenditures", dataset_dir="path/to/data", top_k=20)
```

## Parameters

* `query` (str): Natural language search terms or keywords (for `search` and `search_without_cache`).
* `scope` (str, default: `"data"`): Target node prefix path, relative file path, or subfolder path (e.g. `"data"`, `"data::treasury_bulletin_1941_01"`, `"treasury_bulletins_parsed/transformed/treasury_bulletin_1941_01.txt"`, or `"treasury_bulletins_parsed/transformed"`). Supports Universal Path Resolution across Node IDs, relative paths, and folder paths.
* `dataset_dir` (str): Absolute path to the dataset folder.
* `granularity` (int, default: `1`): Depth level relative to scope.
  * `0`: Same-level sibling lookup (returns scope node itself).
  * `1` (Default): Immediate child nodes (tables under root, or columns under table).
  * `2`: Grandchild nodes (columns directly under root, skipping table level).
  * `>=3`: Deeper descendant nodes (e.g. depth 3 or 4) if the dataset hierarchy supports it.
* `sample_size` (int, default: `50`): Maximum number of nodes to return for `expand`. If the returned list ends with a truncation warning node (`... [Truncated: X more items exist. Pass a larger sample_size to view all]`), you should call `expand` again with a larger `sample_size` (e.g., `sample_size=100`) to retrieve the remaining nodes.
* `top_k` (int, default: `10`): Maximum number of items to return for `search` or column value frequencies in `summarize`. If the column value count ends with a truncation warning (`... [Truncated: X other unique values exist. Pass a larger top_k to view all]`), you should call `summarize` again with a larger `top_k` (e.g., `top_k=20`) to inspect the remaining unique values.

## Returns

* `expand`: `List[str]` — Alphabetically sorted list of child node paths.
* `search` / `search_without_cache`: `List[str]` — Sorted list of matching node paths (highest score first).
* `summarize`: `List[str]` — Text strings summarizing table metadata (columns) or top-value frequencies for columns.
