---
name: hierarchical_summarize
description: A document-centric hierarchical indexing summarization skill. Provides summarize (cat-like) functionality to retrieve schemas, column metadata, and cell value frequencies over a data lake of CSV and Excel tables.
license: MIT
metadata:
  author: Fadhilah
  version: "1.0"
---

# Hierarchical Summarize

A document-centric hierarchical indexing summarization skill. Provides summarize (cat-like) functionality to retrieve schemas, column metadata, and cell value frequencies over a data lake of CSV and Excel tables.

## Usage

```python
import sys
sys.path.insert(0, ".opencode/skills/hierarchical_summarize/scripts")

from hierarchical_index import HierarchicalIndex

index = HierarchicalIndex(dataset_dir="data/archeology/input")

# Summarize a table: shows column names and schema
schema = index.summarize(scope="data::tablename")
for s in schema:
    print(s)

# Summarize a column: shows top value frequencies
values = index.summarize(scope="data::tablename::columnname", top_k=5)
for v in values:
    print(v)
```
