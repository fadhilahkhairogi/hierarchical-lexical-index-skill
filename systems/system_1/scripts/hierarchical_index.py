import os
import re
import json
import math
import hashlib
import random
import urllib.request
import pandas as pd
import bm25s
from typing import List, Dict, Any, Optional

class Node:
    def __init__(self, path: str):
        self.path = path

    def __repr__(self):
        return f"Node('{self.path}')"

    def __str__(self):
        return self.path

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.path == other.path
        return False

    def __hash__(self):
        return hash(self.path)

# --- Embedding helper with robust deterministic fallback & online check ---
_OLLAMA_ONLINE: Optional[bool] = None

def get_embedding(text: str) -> List[float]:
    """
    Computes a text embedding using local Ollama (nomic-embed-text) and normalizes it.
    Uses a global check to avoid trying Ollama if it is known to be offline (prevents timeouts).
    Falls back to a deterministic vector derived from hashing the text.
    """
    global _OLLAMA_ONLINE
    if not isinstance(text, str):
        text = str(text)
    
    def get_fallback():
        emb_dim = 768
        hasher = hashlib.md5(text.encode('utf-8', errors='ignore'))
        seed = int(hasher.hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        emb = [rng.gauss(0.0, 1.0) for _ in range(emb_dim)]
        magnitude = math.sqrt(sum(val ** 2 for val in emb))
        if magnitude > 0:
            emb = [val / magnitude for val in emb]
        return emb

    if _OLLAMA_ONLINE is False:
        return get_fallback()

    try:
        ollama_url = os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434").rstrip('/')
        if '/v1' in ollama_url:
            ollama_url = ollama_url.replace('/v1', '')
            
        model = os.environ.get("R3_EMBED_MODEL", "nomic-embed-text")
        
        if _OLLAMA_ONLINE is None:
            try:
                with urllib.request.urlopen(ollama_url, timeout=0.5) as resp:
                    if resp.status == 200:
                        _OLLAMA_ONLINE = True
            except Exception:
                _OLLAMA_ONLINE = False
                print("[Embedding] Ollama is offline or unreachable. Using deterministic fallback embeddings.")
                return get_fallback()
        
        if len(text) > 8000:
            text = text[:8000]
            
        payload = json.dumps({"model": model, "prompt": text}).encode('utf-8')
        req = urllib.request.Request(
            f"{ollama_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        
        emb = data["embedding"]
        magnitude = math.sqrt(sum(val ** 2 for val in emb))
        if magnitude > 0:
            emb = [val / magnitude for val in emb]
        return emb
    except Exception:
        _OLLAMA_ONLINE = False
        return get_fallback()

class HierarchicalIndex:
    def __init__(self, dataset_dir: str, cache_file: str = "hierarchical_index_cache.json"):
        self.dataset_dir = os.path.normpath(dataset_dir)
        self.cache_file = cache_file
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.corpus_paths: List[str] = []
        self.corpus_texts: List[str] = []
        self.bm25: Optional[bm25s.BM25] = None
        
        self._build_index()
        self._initialize_bm25()

    def _normalize_name(self, name: str) -> str:
        name_no_ext = os.path.splitext(name)[0]
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", name_no_ext)
        return clean_name

    def _split_identifier(self, name: str) -> str:
        s = re.sub(r"[^a-zA-Z0-9]", " ", name)
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
        return s.strip()

    def _parse_semi_structured_file(self, filepath: str) -> Dict[str, Dict[str, Any]]:
        """
        Parses a raw text file, skips metadata headers, and splits vertically stacked sub-tables.
        """
        sub_tables = {}
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                
            blocks = re.split(r'\n\s*\n', content)
            table_counter = 1
            
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                    
                lines = [line.strip() for line in block.split('\n') if line.strip()]
                
                data_lines = []
                for line in lines:
                    if not line.startswith(('#', ':', ';')):
                        data_lines.append(line)
                        
                if not data_lines:
                    continue
                    
                parsed_rows = []
                for line in data_lines:
                    row_cols = [c.strip() for c in re.split(r'\s{2,}|\t', line) if c.strip()]
                    if len(row_cols) > 1:
                        parsed_rows.append(row_cols)
                        
                if len(parsed_rows) < 2:
                    continue
                    
                header = parsed_rows[0]
                rows_data = parsed_rows[1:]
                clean_header = [re.sub(r'[^a-zA-Z0-9_UT\-]', '_', col) for col in header]
                
                seen_headers = {}
                final_header = []
                for h in clean_header:
                    if h in seen_headers:
                        seen_headers[h] += 1
                        final_header.append(f"{h}_{seen_headers[h]}")
                    else:
                        seen_headers[h] = 0
                        final_header.append(h)
                
                if data_lines[0] != " ".join(header) and len(re.split(r'\s{2,}|\t', data_lines[0])) == 1:
                    block_title = re.sub(r'[^a-zA-Z0-9_]', '_', data_lines[0].lower()).strip('_')
                else:
                    block_title = f"table_{table_counter}"
                    table_counter += 1
                    
                max_cols = len(final_header)
                padded_rows = [row[:max_cols] + [""] * (max_cols - len(row)) for row in rows_data]
                
                df = pd.DataFrame(padded_rows, columns=final_header)
                columns_meta = {}
                
                for col in df.columns:
                    unique_vals = df[col].dropna().unique()
                    sample_vals = [str(v) for v in unique_vals if str(v).strip()][:5]
                    columns_meta[col] = {
                        "sample_values": sample_vals
                    }
                    
                sub_tables[block_title] = {
                    "columns": final_header,
                    "columns_meta": columns_meta,
                    "row_count": len(df),
                    "dataframe": df
                }
        except Exception as e:
            print(f"[SemiStructuredParser] Error reading {filepath}: {e}")
        return sub_tables

    def _build_index(self):
        # Look for cached index first
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    if cached_data.get("dataset_dir") == self.dataset_dir:
                        self.nodes = cached_data.get("nodes", {})
                        print(f"[HierarchicalIndex] Loaded {len(self.nodes)} nodes from cache.")
                        return
            except Exception as e:
                print(f"[HierarchicalIndex] Cache load error: {e}")

        print(f"[HierarchicalIndex] Building index for dataset: {self.dataset_dir}")
        self.nodes = {}
        
        # 1. Root Node
        root_path = "data"
        self.nodes[root_path] = {
            "path": root_path,
            "type": "root",
            "name": "data",
            "children": [],
            "representation": "Root data lake directory containing all table and document files.",
            "embedding": get_embedding("Root data lake directory containing all table and document files.")
        }

        if not os.path.exists(self.dataset_dir):
            return

        # 2. File Nodes and Column Nodes
        for root, _, filenames in os.walk(self.dataset_dir):
            for filename in filenames:
                if filename.startswith(".") or filename.startswith("~"):
                    continue
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, self.dataset_dir).replace("\\", "/")
                norm_filename = self._normalize_name(filename)
                
                is_excel = filename.endswith((".xlsx", ".xls"))
                is_csv = filename.endswith(".csv")
                is_txt = filename.endswith(".txt")
                
                if is_csv or is_excel:
                    file_node_path = f"data::{norm_filename}"
                    columns = []
                    row_count = 0
                    df = None
                    
                    try:
                        if is_csv:
                            skip = 0
                            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                                for line in f:
                                    if line.strip().startswith(('#', ':', ';')):
                                        skip += 1
                                    else:
                                        break
                            df = pd.read_csv(filepath, skiprows=skip, nrows=10)
                            columns = list(df.columns)
                            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                                row_count = sum(1 for _ in f) - 1 - skip
                        else:
                            xl = pd.ExcelFile(filepath)
                            sheet_name = xl.sheet_names[0]
                            df = pd.read_excel(filepath, sheet_name=sheet_name, nrows=10)
                            columns = list(df.columns)
                            row_count = len(pd.read_excel(filepath, sheet_name=sheet_name))
                    except Exception as e:
                        print(f"[HierarchicalIndex] Pandas read failed for {filepath}: {e}. Falling back to text parser.")
                        is_txt = True
                    
                    if not is_txt:
                        split_filename = self._split_identifier(norm_filename)
                        split_cols = ", ".join([f"{col} ({self._split_identifier(col)})" if self._split_identifier(col) != col else col for col in columns])
                        file_repr = f"Table: {norm_filename} ({split_filename}), Columns: {split_cols}, File Path: {rel_path}, Total Rows: {row_count}"
                        
                        self.nodes[file_node_path] = {
                            "path": file_node_path,
                            "type": "file",
                            "name": norm_filename,
                            "filepath": filepath,
                            "rel_filepath": rel_path,
                            "columns": columns,
                            "children": [],
                            "representation": file_repr,
                            "embedding": get_embedding(file_repr)
                        }
                        
                        if file_node_path not in self.nodes[root_path]["children"]:
                            self.nodes[root_path]["children"].append(file_node_path)
                        
                        for col in columns:
                            col_node_path = f"{file_node_path}::{col}"
                            sample_vals = []
                            try:
                                unique_vals = df[col].dropna().unique()[:5]
                                sample_vals = [str(v) for v in unique_vals]
                            except Exception:
                                pass
                            
                            sample_vals_str = ", ".join(sample_vals) if sample_vals else "None"
                            col_repr = f"Column: {col} ({self._split_identifier(col)}) in Table: {norm_filename} ({split_filename}), Sample Values: {sample_vals_str}"
                            
                            self.nodes[col_node_path] = {
                                "path": col_node_path,
                                "type": "column",
                                "name": col,
                                "parent": file_node_path,
                                "representation": col_repr,
                                "embedding": get_embedding(col_repr)
                            }
                            self.nodes[file_node_path]["children"].append(col_node_path)

                if is_txt:
                    sub_tables = self._parse_semi_structured_file(filepath)
                    file_node_path = f"data::{norm_filename}"
                    
                    if not sub_tables:
                        split_filename = self._split_identifier(norm_filename)
                        file_repr = f"Document: {norm_filename} ({split_filename}), Type: Plain Text, File Path: {rel_path}"
                        self.nodes[file_node_path] = {
                            "path": file_node_path,
                            "type": "file",
                            "name": norm_filename,
                            "filepath": filepath,
                            "rel_filepath": rel_path,
                            "columns": [],
                            "children": [],
                            "representation": file_repr,
                            "embedding": get_embedding(file_repr)
                        }
                        if file_node_path not in self.nodes[root_path]["children"]:
                            self.nodes[root_path]["children"].append(file_node_path)
                    else:
                        # Register the parent file node at depth 1
                        split_filename = self._split_identifier(norm_filename)
                        file_repr = f"Document containing sub-tables: {norm_filename} ({split_filename}), File Path: {rel_path}"
                        self.nodes[file_node_path] = {
                            "path": file_node_path,
                            "type": "file",
                            "name": norm_filename,
                            "filepath": filepath,
                            "rel_filepath": rel_path,
                            "columns": [],
                            "children": [],
                            "representation": file_repr,
                            "embedding": get_embedding(file_repr)
                        }
                        if file_node_path not in self.nodes[root_path]["children"]:
                            self.nodes[root_path]["children"].append(file_node_path)
                            
                        # Add each sub-table as a child table node under the file node
                        for sub_title, sub_info in sub_tables.items():
                            sub_node_path = f"{file_node_path}::{sub_title}"
                            cols = sub_info["columns"]
                            row_count = sub_info["row_count"]
                            split_subtitle = self._split_identifier(sub_title)
                            split_cols = ", ".join([f"{c} ({self._split_identifier(c)})" if self._split_identifier(c) != c else c for c in cols])
                            
                            sub_repr = f"Sub-Table: {sub_title} ({split_subtitle}) inside {norm_filename} ({split_filename}), Columns: {split_cols}, File Path: {rel_path}, Total Rows: {row_count}"
                            
                            self.nodes[sub_node_path] = {
                                "path": sub_node_path,
                                "type": "file",
                                "name": f"{norm_filename}::{sub_title}",
                                "filepath": filepath,
                                "rel_filepath": rel_path,
                                "columns": cols,
                                "children": [],
                                "representation": sub_repr,
                                "embedding": get_embedding(sub_repr)
                            }
                            self.nodes[file_node_path]["children"].append(sub_node_path)
                            
                            for col in cols:
                                col_node_path = f"{sub_node_path}::{col}"
                                sample_vals = sub_info["columns_meta"][col]["sample_values"]
                                sample_vals_str = ", ".join(sample_vals) if sample_vals else "None"
                                
                                col_repr = f"Column: {col} ({self._split_identifier(col)}) in Sub-Table: {sub_title} ({split_subtitle}) (file: {norm_filename} ({split_filename})), Sample Values: {sample_vals_str}"
                                
                                self.nodes[col_node_path] = {
                                    "path": col_node_path,
                                    "type": "column",
                                    "name": col,
                                    "parent": sub_node_path,
                                    "representation": col_repr,
                                    "embedding": get_embedding(col_repr)
                                }
                                self.nodes[sub_node_path]["children"].append(col_node_path)

        # Save to cache
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump({
                    "dataset_dir": self.dataset_dir,
                    "nodes": self.nodes
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[HierarchicalIndex] Warning: Failed to save index cache: {e}")

    def _initialize_bm25(self):
        self.corpus_paths = []
        self.corpus_texts = []
        
        for path, info in self.nodes.items():
            self.corpus_paths.append(path)
            self.corpus_texts.append(info.get("representation", ""))
            
        if self.corpus_texts:
            corpus_tokens = bm25s.tokenize(self.corpus_texts, stopwords=None, show_progress=False)
            self.bm25 = bm25s.BM25(k1=1.5, b=0.75)
            self.bm25.index(corpus_tokens, show_progress=False)

    def _get_candidates(self, scope: str, granularity: int) -> List[str]:
        if scope not in self.nodes:
            return []
            
        scope_sep = scope.count("::")
        target_sep = scope_sep + granularity
        
        candidates = []
        for node_path in self.nodes.keys():
            if node_path == scope:
                continue
            starts = False
            if scope == "data":
                starts = node_path.startswith("data::")
            else:
                starts = node_path.startswith(scope + "::")
                
            if starts:
                node_sep = node_path.count("::")
                if node_sep == target_sep:
                    candidates.append(node_path)
                    
        return candidates

    def search(self, query: str, scope: str = "data", granularity: int = 1, top_k: int = 5) -> List[Node]:
        candidates = self._get_candidates(scope, granularity)
        if not candidates:
            return []

        bm25_scores = {}
        if self.bm25 and self.corpus_paths:
            query_tokens = bm25s.tokenize(query, stopwords=None, show_progress=False)
            results, scores = self.bm25.retrieve(query_tokens, k=len(self.corpus_paths), show_progress=False)
            for idx, score in zip(results[0], scores[0]):
                path = self.corpus_paths[int(idx)]
                bm25_scores[path] = float(score)

        query_emb = get_embedding(query)
        scored_candidates = []
        max_bm25 = max([bm25_scores.get(p, 0.0) for p in candidates] + [1.0])
        
        for cand_path in candidates:
            cand_info = self.nodes[cand_path]
            cand_emb = cand_info.get("embedding")
            
            sem_score = 0.0
            if cand_emb and query_emb:
                sem_score = sum(q * c for q, c in zip(query_emb, cand_emb))
                
            raw_bm25 = bm25_scores.get(cand_path, 0.0)
            norm_bm25 = raw_bm25 / max_bm25
            
            total_score = 0.6 * sem_score + 0.4 * norm_bm25
            scored_candidates.append((total_score, cand_path))
            
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        results = [Node(path) for _, path in scored_candidates[:top_k]]
        return results

    def summarize(self, scope: str, top_k: int = 5) -> List[str]:
        if scope not in self.nodes:
            return [f"Error: Scope path {scope} not found in index."]
            
        node_info = self.nodes[scope]
        node_type = node_info.get("type")
        
        if node_type == "root":
            files_list = ", ".join(node_info.get("children", []))
            return [f"Root node 'data'. Registered files: {files_list}"]
            
        elif node_type == "file":
            cols = node_info.get("columns", [])
            col_str = ", ".join(cols)
            summary = [
                f"File: {node_info.get('name')}",
                f"Columns: {col_str}",
                f"RelPath: {node_info.get('rel_filepath')}"
            ]
            return summary
            
        elif node_type == "column":
            parent = node_info.get("parent")
            col_name = node_info.get("name")
            parent_info = self.nodes.get(parent)
            if not parent_info:
                return [f"Column: {col_name} (Orphaned)"]
                
            filepath = parent_info.get("filepath")
            
            try:
                parent_name = parent_info.get("name", "")
                if parent_name and "::" in parent_name:
                    sub_tables = self._parse_semi_structured_file(filepath)
                    sub_title = parent_name.split("::")[-1]
                    df = sub_tables[sub_title]["dataframe"]
                else:
                    if filepath.endswith(".csv"):
                        df = pd.read_csv(filepath, usecols=[col_name])
                    else:
                        df = pd.read_excel(filepath, usecols=[col_name])
                    
                val_counts = df[col_name].dropna().value_counts().head(top_k)
                summary_vals = []
                for val, count in val_counts.items():
                    summary_vals.append(f"{val} ({count} occurrences)")
                    
                if not summary_vals:
                    return ["No non-null values found in column."]
                return summary_vals
            except Exception as e:
                return [f"Error reading column data: {e}"]
                
        return ["Unknown node type."]

    def expand(self, scope: str, granularity: int = 1, sample_size: int = 5) -> List[Node]:
        candidates = self._get_candidates(scope, granularity)
        candidates.sort()
        results = [Node(path) for path in candidates[:sample_size]]
        return results
