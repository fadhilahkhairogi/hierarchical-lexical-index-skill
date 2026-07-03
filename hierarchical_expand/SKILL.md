---
name: hierarchical_expand
description: A document-centric hierarchical indexing navigation skill. Provides expand (ls-like) functionality to list children of a scope over a data lake of CSV and Excel tables.
license: MIT
metadata:
  author: Fadhilah
  version: "1.0"
---

# Hierarchical Expand

A document-centric hierarchical indexing navigation skill. Provides expand (ls-like) functionality to list children of a scope over a data lake of CSV and Excel tables.

## Usage

```python
import sys
sys.path.insert(0, ".opencode/skills/hierarchical_expand/scripts")

from hierarchical_index import HierarchicalIndex

index = HierarchicalIndex(dataset_dir="data/archeology/input")

# List files/tables at the top level
tables = index.expand(scope="data", granularity=1)
for t in tables:
    print(t)

# List columns within a specific table
columns = index.expand(scope="data::tablename", granularity=1)
for c in columns:
    print(c)
```
