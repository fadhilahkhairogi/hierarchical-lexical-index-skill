import os
import re
import json
import math
import hashlib
import random
import urllib.request
import pandas as pd
import bm25s
import time
import threading
import queue
import signal
import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

_IS_WRITING_DISK = False

def _sigint_handler(signum, frame):
    global _IS_WRITING_DISK
    if _IS_WRITING_DISK:
        print("\n[HierarchicalIndex] Currently committing batch to disk. Waiting briefly for write to finish...")
        while _IS_WRITING_DISK:
            time.sleep(0.05)
    print("\n[HierarchicalIndex] Interrupted by user (Ctrl+C). Batch safely saved to disk. Exiting instantly.")
    os._exit(1)

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

# --- Embedding helper with thread-safe integration ---
_EMBEDDER_ONLINE: Optional[bool] = None
_EMBEDDER_LOCK = threading.Lock()

def get_embedding(text: str) -> List[float]:
    global _EMBEDDER_ONLINE
    
    embed_url = os.environ.get('EMBED_API_BASE', os.environ.get('OLLAMA_API_BASE', os.environ.get('R3_EMBED_URL', 'http://localhost:11434'))).rstrip('/')
    if '/v1' in embed_url:
        embed_url = embed_url.replace('/v1', '')
        
    model = os.environ.get('R3_EMBED_MODEL', 'embeddinggemma:latest')
    
    with _EMBEDDER_LOCK:
        if _EMBEDDER_ONLINE is False:
            raise RuntimeError('Embedder is offline or model is unreachable. Embedding generation failed.')

    if len(text) > 5000:
        text = text[:5000]
        
    payload = json.dumps({'model': model, 'prompt': text}).encode('utf-8')
    req = urllib.request.Request(
        f'{embed_url}/api/embeddings',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    for attempt in range(5):
        try:
            with _EMBEDDER_LOCK:
                if _EMBEDDER_ONLINE is None or _EMBEDDER_ONLINE is False:
                    try:
                        with urllib.request.urlopen(embed_url, timeout=3.0) as resp:
                            if resp.status == 200:
                                _EMBEDDER_ONLINE = True
                    except Exception:
                        _EMBEDDER_ONLINE = False
                        if attempt == 4:
                            raise RuntimeError('Embedder connection timed out or is unreachable.')
                        time.sleep(2 * (attempt + 1))
                        continue

            with urllib.request.urlopen(req, timeout=30.0) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            
            emb = data['embedding']
            magnitude = math.sqrt(sum(val ** 2 for val in emb))
            if magnitude > 0:
                emb = [val / magnitude for val in emb]
            
            time.sleep(0.02)
            return emb
        except Exception as e:
            if attempt == 4:
                with _EMBEDDER_LOCK:
                    _EMBEDDER_ONLINE = False
                raise RuntimeError(f'Embedding API call failed after 5 attempts: {e}')
            cool_down = 2 * attempt + 1
            print(f"[Embedding] Attempt {attempt+1} failed: {e}. Retrying in {cool_down}s...")
            time.sleep(cool_down)

class HierarchicalIndex:
    def __init__(self, dataset_dir: str, cache_file: str = "hierarchical_index_cache.json", max_workers: int = 50):
        self.dataset_dir = os.path.abspath(os.path.normpath(dataset_dir))
        self.max_workers = max_workers
        
        # Setup ChromaDB persistent directory
        if cache_file.endswith("_chroma_db"):
            self.chroma_dir = cache_file
        elif cache_file.endswith(".json"):
            self.chroma_dir = os.path.splitext(cache_file)[0] + "_chroma_db"
        else:
            self.chroma_dir = cache_file + "_chroma_db"
            
            
        self.cache_file = cache_file
        os.makedirs(self.chroma_dir, exist_ok=True)
        
        # Initialize Persistent ChromaDB Client
        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name="hierarchical_nodes",
            metadata={"hnsw:space": "cosine", "hnsw:search_ef": 400}
        )
        
        self.corpus_paths: List[str] = []
        self.corpus_texts: List[str] = []
        self.bm25: Optional[bm25s.BM25] = None
        self._db_lock = threading.Lock()
        
        try:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, _sigint_handler)
        except Exception:
            pass
            
        self._build_index()
        self._initialize_bm25()

    @property
    def nodes(self) -> Dict[str, Any]:
        try:
            data = self.collection.get(include=["metadatas"])
            ids = data.get("ids", [])
            metas = data.get("metadatas", [])
            return {id_: meta for id_, meta in zip(ids, metas)}
        except Exception:
            return {}

    def __len__(self):
        return self.collection.count()

    def _normalize_name(self, name: str) -> str:
        name_no_ext = os.path.splitext(name)[0]
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", name_no_ext)
        return clean_name

    def _split_identifier(self, name: str) -> str:
        s = re.sub(r"[^a-zA-Z0-9]", " ", name)
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
        return s.strip()

    def _parse_semi_structured_file(self, filepath: str) -> Dict[str, Dict[str, Any]]:
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
                    if not line.startswith('#'):
                        data_lines.append(line)
                        
                if not data_lines:
                    continue
                    
                parsed_rows = []
                has_pipe = any('|' in line for line in data_lines)
                
                for line in data_lines:
                    if has_pipe:
                        parts = [c.strip() for c in line.split('|')]
                        if parts and parts[0] == '':
                            parts = parts[1:]
                        if parts and parts[-1] == '':
                            parts = parts[:-1]
                        if not parts:
                            continue
                        if all(re.match(r'^\s*:?-+:?\s*$', c) for c in parts):
                            continue
                        parsed_rows.append(parts)
                    else:
                        row_cols = [c.strip() for c in re.split(r'\s{2,}|\t', line) if c.strip()]
                        if len(row_cols) > 1:
                            parsed_rows.append(row_cols)
                        
                if len(parsed_rows) < 2:
                    continue
                    
                header = parsed_rows[0]
                rows_data = parsed_rows[1:]
                clean_header = [re.sub(r'[^\w\-\s]', '_', str(col)).strip() for col in header]
                
                seen_headers = {}
                final_header = []
                for h in clean_header:
                    if not h:
                        h = "unnamed"
                    if h in seen_headers:
                        seen_headers[h] += 1
                        final_header.append(f"{h}_{seen_headers[h]}")
                    else:
                        seen_headers[h] = 0
                        final_header.append(h)
                
                if data_lines[0] != " ".join(header) and len(re.split(r'\s{2,}|\t', data_lines[0])) == 1 and not has_pipe:
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

    def _parse_json_file(self, filepath: str) -> Dict[str, Dict[str, Any]]:
        import io
        sub_tables = {}
        table_counter = 1
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)

            elements = []
            if isinstance(data, dict):
                if 'document' in data and isinstance(data['document'], dict):
                    elements = data['document'].get('elements', [])
                elif 'elements' in data and isinstance(data['elements'], list):
                    elements = data['elements']

            for elem in elements:
                if isinstance(elem, dict) and (elem.get('type') == 'table' or elem.get('category') == 'table'):
                    content = elem.get('content') or elem.get('text') or ''
                    if content and ('<table' in content or '|' in content):
                        try:
                            if '<table' in content:
                                dfs = pd.read_html(io.StringIO(content))
                                if dfs:
                                    df = dfs[0]
                                    clean_header = [re.sub(r'[^a-zA-Z0-9_]', '_', str(c)) for c in df.columns]
                                    seen = {}
                                    final_header = []
                                    for h in clean_header:
                                        if not h: h = 'unnamed'
                                        if h in seen:
                                            seen[h] += 1
                                            final_header.append(f'{h}_{seen[h]}')
                                        else:
                                            seen[h] = 0
                                            final_header.append(h)
                                    df.columns = final_header
                                    sub_title = f'table_{table_counter}'
                                    table_counter += 1
                                    cols_meta = {}
                                    for col in df.columns:
                                        sample_vals = [str(v) for v in df[col].dropna().unique() if str(v).strip()][:5]
                                        cols_meta[str(col)] = {'sample_values': sample_vals}
                                    sub_tables[sub_title] = {
                                        'columns': final_header,
                                        'columns_meta': cols_meta,
                                        'row_count': len(df),
                                        'dataframe': df
                                    }
                        except Exception:
                            pass

            if not sub_tables:
                df = None
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    df = pd.DataFrame(data[:50])
                elif isinstance(data, dict):
                    rows = []
                    for k, v in data.items():
                        if isinstance(v, (dict, list)):
                            rows.append({'key': k, 'value': str(v)[:100]})
                    if rows:
                        df = pd.DataFrame(rows)
                if df is not None and not df.empty:
                    clean_header = [re.sub(r'[^a-zA-Z0-9_]', '_', str(c)) for c in df.columns]
                    df.columns = clean_header
                    cols_meta = {}
                    for col in df.columns:
                        sample_vals = [str(v) for v in df[col].dropna().unique() if str(v).strip()][:5]
                        cols_meta[str(col)] = {'sample_values': sample_vals}
                    sub_tables['table_1'] = {
                        'columns': clean_header,
                        'columns_meta': cols_meta,
                        'row_count': len(df),
                        'dataframe': df
                    }
        except Exception as e:
            print(f"[JSONParser] Error reading {filepath}: {e}")
        return sub_tables

    def _build_index(self):
        try:
            total_db_count = self.collection.count()
            if total_db_count > 0:
                print(f"[HierarchicalIndex] Found existing ChromaDB cache with {total_db_count} nodes at {self.chroma_dir}.")
                return
        except Exception as e:
            pass
        print(f"[HierarchicalIndex] Parsing dataset (ChromaDB + Multithreaded n={self.max_workers}): {self.dataset_dir}")
        raw_nodes_specs: Dict[str, Dict[str, Any]] = {}
        
        root_path = "data"
        raw_nodes_specs[root_path] = {
            "path": root_path,
            "type": "root",
            "name": "data",
            "children": [],
            "representation": "Root data lake directory containing all table and document files."
        }

        if not os.path.exists(self.dataset_dir):
            return

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
                is_json = filename.endswith(".json")
                is_other = not (is_csv or is_excel or is_txt or is_json)
                
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
                                    if line.strip().startswith('#'):
                                        skip += 1
                                    else:
                                        break
                            for encoding in ["utf-8", "latin1", "cp1252", "utf-8-sig"]:
                                try:
                                    try:
                                        df = pd.read_csv(filepath, skiprows=skip, nrows=10, encoding=encoding, on_bad_lines='skip')
                                    except TypeError:
                                        df = pd.read_csv(filepath, skiprows=skip, nrows=10, encoding=encoding, error_bad_lines=False, warn_bad_lines=False)
                                    columns = list(df.columns)
                                    with open(filepath, "r", encoding=encoding, errors="ignore") as f:
                                        row_count = sum(1 for _ in f) - 1 - skip
                                    break
                                except Exception:
                                    continue
                            else:
                                raise ValueError("CSV parse error")
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
                        
                        raw_nodes_specs[file_node_path] = {
                            "path": file_node_path,
                            "type": "file",
                            "name": norm_filename,
                            "filepath": filepath,
                            "rel_filepath": rel_path,
                            "columns": columns,
                            "children": [],
                            "representation": file_repr
                        }
                        
                        if file_node_path not in raw_nodes_specs[root_path]["children"]:
                            raw_nodes_specs[root_path]["children"].append(file_node_path)
                        
                        for col in columns:
                            col_node_path = f"{file_node_path}::{col}"
                            sample_vals = []
                            try:
                                if df is not None:
                                    unique_vals = df[col].dropna().unique()[:5]
                                    sample_vals = [str(v) for v in unique_vals]
                            except Exception:
                                pass
                            
                            sample_vals_str = ", ".join(sample_vals) if sample_vals else "None"
                            col_repr = f"Column: {col} ({self._split_identifier(col)}) in Table: {norm_filename} ({split_filename}), Sample Values: {sample_vals_str}"
                            
                            raw_nodes_specs[col_node_path] = {
                                "path": col_node_path,
                                "type": "column",
                                "name": col,
                                "parent": file_node_path,
                                "representation": col_repr
                            }
                            raw_nodes_specs[file_node_path]["children"].append(col_node_path)

                if is_json:
                    sub_tables = self._parse_json_file(filepath)
                    is_txt = True

                if is_txt:
                    if 'sub_tables' not in locals():
                        sub_tables = self._parse_semi_structured_file(filepath)
                    file_node_path = f"data::{norm_filename}"
                    
                    if not sub_tables:
                        split_filename = self._split_identifier(norm_filename)
                        file_repr = f"Document: {norm_filename} ({split_filename}), Type: Plain Text, File Path: {rel_path}"
                        raw_nodes_specs[file_node_path] = {
                            "path": file_node_path,
                            "type": "file",
                            "name": norm_filename,
                            "filepath": filepath,
                            "rel_filepath": rel_path,
                            "columns": [],
                            "children": [],
                            "representation": file_repr
                        }
                        if file_node_path not in raw_nodes_specs[root_path]["children"]:
                            raw_nodes_specs[root_path]["children"].append(file_node_path)
                    else:
                        split_filename = self._split_identifier(norm_filename)
                        file_repr = f"Document containing sub-tables: {norm_filename} ({split_filename}), File Path: {rel_path}"
                        raw_nodes_specs[file_node_path] = {
                            "path": file_node_path,
                            "type": "file",
                            "name": norm_filename,
                            "filepath": filepath,
                            "rel_filepath": rel_path,
                            "columns": [],
                            "children": [],
                            "representation": file_repr
                        }
                        if file_node_path not in raw_nodes_specs[root_path]["children"]:
                            raw_nodes_specs[root_path]["children"].append(file_node_path)
                            
                        for sub_title, sub_info in sub_tables.items():
                            sub_node_path = f"{file_node_path}::{sub_title}"
                            cols = sub_info["columns"]
                            row_count = sub_info["row_count"]
                            split_subtitle = self._split_identifier(sub_title)
                            split_cols = ", ".join([f"{c} ({self._split_identifier(c)})" if self._split_identifier(c) != c else c for c in cols])
                            
                            sub_repr = f"Sub-Table: {sub_title} ({split_subtitle}) inside {norm_filename} ({split_filename}), Columns: {split_cols}, File Path: {rel_path}, Total Rows: {row_count}"
                            
                            raw_nodes_specs[sub_node_path] = {
                                "path": sub_node_path,
                                "type": "file",
                                "name": f"{norm_filename}::{sub_title}",
                                "filepath": filepath,
                                "rel_filepath": rel_path,
                                "columns": cols,
                                "children": [],
                                "representation": sub_repr
                            }
                            raw_nodes_specs[file_node_path]["children"].append(sub_node_path)
                            
                            for col in cols:
                                col_node_path = f"{sub_node_path}::{col}"
                                sample_vals = sub_info.get("columns_meta", {}).get(col, {}).get("sample_values", [])
                                sample_vals_str = ", ".join(sample_vals[:5]) if sample_vals else "None"
                                
                                col_repr = f"Column: {col} ({self._split_identifier(col)}) in Sub-Table: {sub_title} ({split_subtitle}) (file: {norm_filename} ({split_filename})), Sample Values: {sample_vals_str}"
                                
                                raw_nodes_specs[col_node_path] = {
                                    "path": col_node_path,
                                    "type": "column",
                                    "name": col,
                                    "parent": sub_node_path,
                                    "representation": col_repr
                                }
                                raw_nodes_specs[sub_node_path]["children"].append(col_node_path)

                if is_other:
                    file_node_path = f"data::{norm_filename}"
                    split_filename = self._split_identifier(norm_filename)
                    file_repr = f"Document: {norm_filename} ({split_filename}), Type: Plain Text, File Path: {rel_path}"
                    raw_nodes_specs[file_node_path] = {
                        "path": file_node_path,
                        "type": "file",
                        "name": norm_filename,
                        "filepath": filepath,
                        "rel_filepath": rel_path,
                        "columns": [],
                        "children": [],
                        "representation": file_repr
                    }
                    if file_node_path not in raw_nodes_specs[root_path]["children"]:
                        raw_nodes_specs[root_path]["children"].append(file_node_path)

        total_nodes = len(raw_nodes_specs)
        
        # #SE
        total_db_count = self.collection.count()
        fetch_limit = max(total_db_count, 1000000) if total_db_count > 0 else 1000000
        existing_data = self.collection.get(include=[], limit=fetch_limit)
        # #EE
        existing_ids = set(existing_data.get("ids", []))
        
        nodes_to_embed = {p: spec for p, spec in raw_nodes_specs.items() if p not in existing_ids}
        already_cached_count = total_nodes - len(nodes_to_embed)
        
        if already_cached_count > 0:
            print(f"[HierarchicalIndex] {already_cached_count}/{total_nodes} nodes already cached in ChromaDB. {len(nodes_to_embed)} remaining to embed.")

        if not nodes_to_embed:
            print(f"[HierarchicalIndex] All {total_nodes} nodes are fully cached in ChromaDB.")
            return

        print(f"[HierarchicalIndex] Computing embeddings for {len(nodes_to_embed)} nodes (workers={self.max_workers})...")
        
        # Thread-Safe Queue Batch Writer Engine (Eliminates Database Lock Contention)
        write_queue = queue.Queue()
        checkpoint_interval = 10000
        
        def _batch_db_writer():
            global _IS_WRITING_DISK
            buffered_ids = []
            buffered_embeddings = []
            buffered_metadatas = []
            buffered_documents = []
            written_total = already_cached_count
            
            while True:
                item = write_queue.get()
                if item is None:
                    # Flush remaining buffer on exit
                    if buffered_ids:
                        _IS_WRITING_DISK = True
                        try:
                            with self._db_lock:
                                self.collection.add(
                                    ids=buffered_ids,
                                    embeddings=buffered_embeddings,
                                    metadatas=buffered_metadatas,
                                    documents=buffered_documents
                                )
                        finally:
                            _IS_WRITING_DISK = False
                        written_total += len(buffered_ids)
                        print(f"  [Checkpoint] Saved {written_total}/{total_nodes} nodes to ChromaDB.")
                    write_queue.task_done()
                    break
                    
                p, node_data, emb = item
                meta = {
                    "type": node_data.get("type", ""),
                    "name": node_data.get("name", ""),
                    "filepath": node_data.get("filepath", ""),
                    "rel_filepath": node_data.get("rel_filepath", ""),
                    "parent": node_data.get("parent", ""),
                    "columns": json.dumps(node_data.get("columns", []))
                }
                
                buffered_ids.append(p)
                buffered_embeddings.append(emb)
                buffered_metadatas.append(meta)
                buffered_documents.append(node_data.get("representation", ""))
                
                if len(buffered_ids) >= 1000:
                    _IS_WRITING_DISK = True
                    try:
                        with self._db_lock:
                            self.collection.add(
                                ids=buffered_ids,
                                embeddings=buffered_embeddings,
                                metadatas=buffered_metadatas,
                                documents=buffered_documents
                            )
                    finally:
                        _IS_WRITING_DISK = False
                    written_total += len(buffered_ids)
                    if written_total % checkpoint_interval == 0:
                        print(f"  [Checkpoint] Saved {written_total}/{total_nodes} nodes to ChromaDB.")
                    buffered_ids, buffered_embeddings, buffered_metadatas, buffered_documents = [], [], [], []
                
                write_queue.task_done()

        writer_thread = threading.Thread(target=_batch_db_writer, daemon=True)
        writer_thread.start()

        def _process_node_embedding(path: str, node_data: Dict[str, Any]):
            repr_text = node_data["representation"]
            emb = get_embedding(repr_text)
            write_queue.put((path, node_data, emb))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_process_node_embedding, p, spec): p
                for p, spec in nodes_to_embed.items()
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"[HierarchicalIndex Error] Embedding failed for {path}: {exc}")

        # Signal writer thread to flush and finish
        write_queue.put(None)
        write_queue.join()
        writer_thread.join()

        print(f"[HierarchicalIndex] Completed indexing. Total active nodes in ChromaDB: {self.collection.count()}")

    def _initialize_bm25(self):
        self.corpus_paths = []
        self.corpus_texts = []
        
        bm25_dir = self.chroma_dir.replace("_chroma_db", "_bm25")
        if os.path.exists(bm25_dir) and os.path.exists(os.path.join(bm25_dir, "corpus_paths.json")):
            try:
                import time
                t_start = time.time()
                self.bm25 = bm25s.BM25.load(bm25_dir, load_corpus=True)
                with open(os.path.join(bm25_dir, "corpus_paths.json"), "r", encoding="utf-8") as f:
                    self.corpus_paths = json.load(f)
                print(f"[HierarchicalIndex] Loaded BM25 index with {len(self.corpus_paths)} docs from disk cache in {time.time() - t_start:.2f}s")
                return
            except Exception as e:
                print(f"[HierarchicalIndex Warning] Failed to load BM25 disk cache: {e}")
                
        try:
            total_count = self.collection.count()
            fetch_limit = max(total_count, 1000000) if total_count > 0 else 1000000
            data = self.collection.get(include=["documents"], limit=fetch_limit)
            self.corpus_paths = data.get("ids", [])
            self.corpus_texts = data.get("documents", [])
        except Exception as e:
            print(f"[HierarchicalIndex Warning] BM25 load error: {e}")
            
        if self.corpus_texts:
            corpus_tokens = bm25s.tokenize(self.corpus_texts, stopwords=None, show_progress=False)
            self.bm25 = bm25s.BM25(k1=1.5, b=0.75)
            self.bm25.index(corpus_tokens, show_progress=False)
            
            try:
                os.makedirs(bm25_dir, exist_ok=True)
                self.bm25.save(bm25_dir, show_progress=False)
                with open(os.path.join(bm25_dir, "corpus_paths.json"), "w", encoding="utf-8") as f:
                    json.dump(self.corpus_paths, f)
                print(f"[HierarchicalIndex] Saved BM25 index to disk cache at {bm25_dir}")
            except Exception as e:
                print(f"[HierarchicalIndex Warning] Failed to save BM25 disk cache: {e}")

    def _get_candidates(self, scope: str, granularity: int) -> List[str]:
        # #SE
        # Universal Path Resolver: resolve relative path or folder path to node ID if scope is not data::
        resolved_scope = scope
        if scope != "data" and not scope.startswith("data::"):
            norm_scope = scope.replace("\\", "/").strip("/")
            all_files = self.collection.get(where={"type": "file"}, include=["metadatas"], limit=30000)
            metas = all_files.get("metadatas", [])
            ids = all_files.get("ids", [])
            for nid, m in zip(ids, metas):
                rel_p = m.get("rel_filepath", "").replace("\\", "/").strip("/")
                if norm_scope == rel_p or rel_p.endswith("/" + norm_scope) or norm_scope in rel_p or m.get("name") == norm_scope:
                    resolved_scope = nid
                    break
        # #EE
        
        scope_sep = resolved_scope.count("::")
        target_sep = scope_sep + granularity
        
        candidates = []
        all_ids = self.corpus_paths
        
        for node_path in all_ids:
            if node_path == resolved_scope:
                continue
            starts = False
            if resolved_scope == "data":
                starts = node_path.startswith("data::")
            else:
                starts = node_path.startswith(resolved_scope + "::")
                
            if starts:
                node_sep = node_path.count("::")
                if node_sep == target_sep:
                    candidates.append(node_path)
                    
        return candidates

    def search(self, query: str, scope: str = "data", granularity: int = 1, top_k: int = 10) -> List[Node]:
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
        
        # #SE
        # Chunked retrieval for candidates to prevent SQLite too many SQL variables error (>32766 candidates)
        cand_emb_map = {}
        chunk_size = 1000
        for i in range(0, len(candidates), chunk_size):
            chunk = candidates[i:i + chunk_size]
            c_data = self.collection.get(ids=chunk, include=["embeddings"])
            c_ids = c_data.get("ids", [])
            c_embs = c_data.get("embeddings", [])
            for cid, emb in zip(c_ids, c_embs):
                cand_emb_map[cid] = emb
        # #EE

        max_bm25 = max([bm25_scores.get(p, 0.0) for p in candidates] + [1.0])
        
        scored_candidates = []
        for cand_path in candidates:
            cand_emb = cand_emb_map.get(cand_path)
            sem_score = 0.0
            if cand_emb is not None and query_emb is not None:
                sem_score = sum(q * c for q, c in zip(query_emb, cand_emb))
                
            raw_bm25 = bm25_scores.get(cand_path, 0.0)
            norm_bm25 = raw_bm25 / max_bm25
            
            total_score = 0.6 * sem_score + 0.4 * norm_bm25
            scored_candidates.append((total_score, cand_path))
            
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        results = [Node(path) for _, path in scored_candidates[:top_k]]
        return results

    def summarize(self, scope: str, top_k: int = 10) -> List[str]:
        # #SE
        # Universal Path Resolver for summarize
        resolved_scope = scope
        if scope != "data" and not scope.startswith("data::"):
            norm_scope = scope.replace("\\", "/").strip("/")
            all_files = self.collection.get(where={"type": "file"}, include=["metadatas"], limit=30000)
            metas_list = all_files.get("metadatas", [])
            ids_list = all_files.get("ids", [])
            
            # Check folder match first
            folder_matches = [m.get("name") for m in metas_list if norm_scope in m.get("rel_filepath", "").replace("\\", "/")]
            if folder_matches and len(folder_matches) > 1:
                return [f"Folder scope '{scope}'. Registered files inside ({len(folder_matches)} items): {', '.join(folder_matches[:top_k])}"]
                
            for nid, m in zip(ids_list, metas_list):
                rel_p = m.get("rel_filepath", "").replace("\\", "/").strip("/")
                if norm_scope == rel_p or rel_p.endswith("/" + norm_scope) or norm_scope in rel_p or m.get("name") == norm_scope:
                    resolved_scope = nid
                    break
        # #EE

        node_data = self.collection.get(ids=[resolved_scope], include=["metadatas"])
        metas = node_data.get("metadatas", [])
        if not metas or not metas[0]:
            return [f"Error: Scope path '{scope}' not found in index."]
            
        node_info = metas[0]
        node_type = node_info.get("type")
        
        if node_type == "root":
            # #SE
            # Retrieve all 697 file nodes using limit=2000 (covers all dataset files safely without SQLite parameter limit)
            all_data = self.collection.get(where={"type": "file"}, include=["metadatas"], limit=30000)
            # #EE
            files_list = [m.get("name") for m in all_data.get("metadatas", []) if m.get("type") == "file"]
            return [f"Root node 'data'. Registered files: {', '.join(files_list[:50])}"]
            
        elif node_type == "file":
            cols_json = node_info.get("columns", "[]")
            try:
                cols = json.loads(cols_json)
            except Exception:
                cols = []
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
            
            parent_data = self.collection.get(ids=[parent], include=["metadatas"])
            parent_metas = parent_data.get("metadatas", [])
            if not parent_metas or not parent_metas[0]:
                return [f"Column: {col_name} (Orphaned)"]
                
            parent_info = parent_metas[0]
            filepath = parent_info.get("filepath")
            
            try:
                parent_name = parent_info.get("name", "")
                if parent_name and "::" in parent_name:
                    sub_tables = self._parse_semi_structured_file(filepath)
                    sub_title = parent_name.split("::")[-1]
                    df = sub_tables[sub_title]["dataframe"]
                else:
                    if filepath.endswith(".csv"):
                        for encoding in ["utf-8", "latin1", "cp1252", "utf-8-sig"]:
                            try:
                                try:
                                    df = pd.read_csv(filepath, usecols=[col_name], encoding=encoding, on_bad_lines='skip')
                                except TypeError:
                                    df = pd.read_csv(filepath, usecols=[col_name], encoding=encoding, error_bad_lines=False, warn_bad_lines=False)
                                break
                            except Exception:
                                continue
                        else:
                            raise ValueError("CSV parse error")
                    else:
                        df = pd.read_excel(filepath, usecols=[col_name])
                    
                val_counts = df[col_name].dropna().value_counts()
                total_unique = len(val_counts)
                
                if total_unique == 0:
                    return ["No non-null values found in column."]
                    
                val_counts_head = val_counts.head(top_k)
                summary_vals = [
                    f"Column: {col_name}",
                    f"Total Unique Values: {total_unique}",
                    f"Value Frequencies (Top {len(val_counts_head)}):"
                ]
                for val, count in val_counts_head.items():
                    summary_vals.append(f"- {val} ({count} occurrences)")
                    
                if total_unique > top_k:
                    remaining = total_unique - top_k
                    summary_vals.append(f"... [Truncated: {remaining} other unique values exist. Pass a larger top_k to view all]")
                    
                return summary_vals
            except Exception as e:
                return [f"Error reading column data: {e}"]
                
        return ["Unknown node type."]

    def expand(self, scope: str, granularity: int = 1, sample_size: int = 50) -> List[Node]:
        candidates = self._get_candidates(scope, granularity)
        candidates.sort()
        
        total_candidates = len(candidates)
        if total_candidates > sample_size:
            results = [Node(path) for path in candidates[:sample_size]]
            remaining = total_candidates - sample_size
            results.append(Node(f"... [Truncated: {remaining} more items exist. Pass a larger sample_size to view all]"))
        else:
            results = [Node(path) for path in candidates]
        return results
