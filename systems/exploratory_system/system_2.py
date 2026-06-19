import typing_extensions
import os
import sys
import re
import json
import time
import math
import random
import ast
import pandas as pd
import numpy as np
import litellm
import bm25s
import fnmatch
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.benchmark_api import System
from systems.smolagents.smolagents_lib import Tool, CodeAgent, AgentLogger, tool
from systems.smolagents.tools import write_file, list_input_filepaths
from systems.smolagents.smolagents_utils import parse_token_counts

# --- Embedding Helper ---
def get_embedding(text: str) -> list[float]:
    """
    Computes a text embedding using local Ollama (nomic-embed-text) and normalizes it.
    Does NOT fall back to remote APIs to prevent accidental API costs.
    Override model via R3_EMBED_MODEL env var.
    """
    try:
        import urllib.request
        import json as _json
        import math
        
        ollama_url = os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434").rstrip('/')
        if '/v1' in ollama_url:
            ollama_url = ollama_url.replace('/v1', '')
            
        model = os.environ.get("R3_EMBED_MODEL", "nomic-embed-text")
        
        # Truncate text to a maximum of 8000 
        if isinstance(text, str) and len(text) > 8000:
            text = text[:8000]
            
        payload = _json.dumps({"model": model, "prompt": text}).encode('utf-8')
        req = urllib.request.Request(
            f"{ollama_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = _json.loads(resp.read().decode('utf-8'))
        
        emb = data["embedding"]
        # Normalize the embedding so that dot product equals cosine similarity
        magnitude = math.sqrt(sum(val ** 2 for val in emb))
        if magnitude > 0:
            emb = [val / magnitude for val in emb]
        return emb
    except Exception as e:
        error_msg = f"Ollama is offline or model is missing: {e}. Please start Ollama locally (run 'ollama serve' and ensure nomic-embed-text or bge-m3 is pulled)."
        print(f"[Embedding Error] {error_msg}")
        raise RuntimeError(error_msg)

# --- LLM Helper ---
def call_llm(messages: list, model: str = "openrouter/deepseek/deepseek-v4-flash") -> str:
    """
    Calls LiteLLM completion with the specified model and OpenRouter base.
    """
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = "https://openrouter.ai/api/v1"
        response = litellm.completion(
            model=model,
            messages=messages,
            api_base=api_base,
            api_key=api_key,
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[LLM Error] {e}")
        raise e

# SE
# --- Three-Valued Logic Evaluation (Kleene Logic via SymPy) ---
import sympy
from sympy.parsing.sympy_parser import parse_expr

def evaluate_three_valued_logic(expression: str, proposition_values: dict[str, Optional[bool]]) -> Optional[bool]:
    """
    Evaluates a boolean logic expression with 3-valued logic (True, False, None/Null)
    using Kleene logic semantics, using SymPy's native symbolic evaluation.
    Supports operators: AND (&&, &), OR (||, |), NOT (!, ~), IMPLIES (->, >>), (, )
    """
    # 1. Standardize operators
    expr = expression.replace("&&", " & ").replace("||", " | ").replace("!", " ~ ")
    expr = expr.replace("and", " & ").replace("or", " | ").replace("not", " ~ ")
    expr = expr.replace("AND", " & ").replace("OR", " | ").replace("NOT", " ~ ")
    expr = expr.replace("->", " >> ").replace("implies", " >> ").replace("IMPLIES", " >> ")
    
    try:
        parsed_expr = parse_expr(expr)
    except Exception as e:
        print(f"[Logic Parser] SymPy parse error for expression '{expr}': {e}")
        return None
   
    # 2. Evaluate using symbolic substitution
    # Map proposition_values to SymPy booleans, keeping None values as unresolved symbols.
    subs_dict = {k: v for k, v in proposition_values.items() if v is not None}
    
    try:
        evaluated = sympy.simplify(parsed_expr.subs(subs_dict))
        if evaluated == sympy.true:
            return True
        elif evaluated == sympy.false:
            return False
        return None
    except Exception as e:
        print(f"[Logic Evaluator] SymPy evaluation error: {e}")
        return None


# --- Step Callback Helper: extract tool calls from CodeAgent steps ---
_R3_TOOL_NAMES = {
    "bm25_search": "bm25_search",
    "rag_search": "rag_search",
    "hybrid_semantic_then_keyword_search": "hybrid_semantic_then_keyword_search",
    "llm_keyword_search": "llm_keyword_search",
    "dci_find": "dci_find",
    "dci_grep": "dci_grep",
    "dci_cat": "dci_cat",
    "consult_explorer": "consult_explorer",
    "consult_supervisor": "consult_supervisor",
    "write_file": "write_file",
    "list_input_filepaths": "list_input_filepaths",
    "final_answer": "final_answer",
}

_PYTHON_ACTION_ALIASES = {
    "pd.read_csv": "pandas.read_csv",
    "pandas.read_csv": "pandas.read_csv",
    "read_csv": "pandas.read_csv",
    "pd.read_excel": "pandas.read_excel",
    "pandas.read_excel": "pandas.read_excel",
    "read_excel": "pandas.read_excel",
    "pd.read_json": "pandas.read_json",
    "pandas.read_json": "pandas.read_json",
    "read_json": "pandas.read_json",
    "open": "file_open",
    "os.listdir": "os.listdir",
    "listdir": "os.listdir",
    "os.walk": "os.walk",
    "walk": "os.walk",
    "glob.glob": "glob.glob",
    "glob": "glob.glob",
}

_R3_TOOL_PATTERN = re.compile(r"\b(" + "|".join(map(re.escape, _R3_TOOL_NAMES)) + r")\s*\(", re.IGNORECASE)
_PYTHON_ACTION_PATTERN = re.compile(
    r"\b(pd\.read_csv|pandas\.read_csv|read_csv|pd\.read_excel|pandas\.read_excel|read_excel|"
    r"pd\.read_json|pandas\.read_json|read_json|open|os\.listdir|listdir|os\.walk|walk|glob\.glob|glob)\s*\(",
    re.IGNORECASE,
)

def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)

def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""

def _normalize_call_name(name: str) -> str | None:
    lower = name.lower()
    base = lower.rsplit(".", 1)[-1]
    if base in _R3_TOOL_NAMES:
        return _R3_TOOL_NAMES[base]
    if lower in _PYTHON_ACTION_ALIASES:
        return _PYTHON_ACTION_ALIASES[lower]
    if base in _PYTHON_ACTION_ALIASES:
        return _PYTHON_ACTION_ALIASES[base]
    return None

def _extract_calls_from_python(code: str) -> list[str]:
    calls = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            normalized = _normalize_call_name(_call_name(node.func))
            if normalized:
                _append_unique(calls, normalized)
    return calls

def _extract_calls_from_text(text: str) -> list[str]:
    calls = []
    if not text:
        return calls

    for match in _R3_TOOL_PATTERN.findall(text):
        _append_unique(calls, _R3_TOOL_NAMES[match.lower()])
    for match in _PYTHON_ACTION_PATTERN.findall(text):
        normalized = _normalize_call_name(match)
        if normalized:
            _append_unique(calls, normalized)
    for call in _extract_calls_from_python(text):
        _append_unique(calls, call)
    return calls

def _extract_tools_from_code(model_output: str) -> str:
    """Parse generated code or model output and return concrete tool/action names."""
    calls = _extract_calls_from_text(model_output or "")
    return ", ".join(calls) if calls else "python_interpreter"

def _extract_tools_from_step(step_info) -> str:
    """Extract concrete R3 tool names from a CodeAgent ActionStep.

    CodeAgent records the outer call as python_interpreter. The actual R3 tools
    are inside that call's Python code argument, so inspect both model_output and
    tool_call.arguments.
    """
    snippets = []
    model_output = getattr(step_info, "model_output", None)
    if model_output:
        snippets.append(model_output)

    tool_names = []
    for tool_call in getattr(step_info, "tool_calls", None) or []:
        name = getattr(tool_call, "name", "")
        if name and name != "python_interpreter":
            _append_unique(tool_names, name)
        arguments = getattr(tool_call, "arguments", None)
        if isinstance(arguments, str):
            snippets.append(arguments)
        elif arguments is not None:
            try:
                snippets.append(json.dumps(arguments, ensure_ascii=False))
            except TypeError:
                snippets.append(str(arguments))

    calls = []
    for name in tool_names:
        normalized = _normalize_call_name(name) or name
        _append_unique(calls, normalized)
    for snippet in snippets:
        for call in _extract_calls_from_text(snippet):
            _append_unique(calls, call)

    return ", ".join(calls) if calls else "python_interpreter"
# Key: (dataset_dir, tuple(sorted(subset_files))) -> (corpus: List[Dict], emb_array: np.ndarray)
_RAG_CACHE: dict = {}

# --- Corpus Loading Helper ---
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    if not text:
        return chunks
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def load_dataset_corpus(dataset_dir: str, subset_files: List[str] = []) -> List[Dict[str, Any]]:
    corpus_chunks = []
    if not dataset_dir or not os.path.exists(dataset_dir):
        return corpus_chunks
    
    files = []
    for root, _, filenames in os.walk(dataset_dir):
        for fname in filenames:
            if fname.startswith("."):
                continue
            full_path = os.path.join(root, fname)
            if subset_files:
                if not any(f in fname or f in full_path for f in subset_files):
                    continue
            files.append(full_path)
            
    for filepath in files:
        content = ""
        try:
            if filepath.endswith(".csv"):
                df = pd.read_csv(filepath)
                content = df.to_string()
            elif filepath.endswith(".xlsx"):
                df = pd.read_excel(filepath)
                content = df.to_string()
            else:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
        except Exception as e:
            print(f"[Corpus Loader] Error reading {filepath}: {e}")
            continue
            
        chunks = chunk_text(content, chunk_size=1000, overlap=200)
        for i, chunk in enumerate(chunks):
            corpus_chunks.append({
                "filepath": filepath,
                "chunk_idx": i,
                "content": f"File: {os.path.basename(filepath)}\nChunk: {i}\n---\n{chunk}"
            })
            
    return corpus_chunks

# --- Retrieval Tools ---
class BM25SearchTool(Tool):
    name = "bm25_search"
    description = "Traditional keyword search using BM25. Finds the most relevant document chunks matching the query keywords."
    inputs = {
        "query": {
            "description": "The search query containing keywords.",
            "type": "string"
        }
    }
    output_type = "string"

    def __init__(self, dataset_dir: str, subset_files: List[str] = []):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.subset_files = subset_files
        self.corpus = None
        self.bm25 = None
        self._corpus_tokens = None

    def _initialize_bm25(self):
        if self.corpus is None:
            import bm25s
            self.corpus = load_dataset_corpus(self.dataset_dir, self.subset_files)
            texts = [c["content"] for c in self.corpus]
            # bm25s: tokenize corpus, build sparse inverted index (much faster than linear scan)
            self._corpus_tokens = bm25s.tokenize(texts, stopwords=None, show_progress=False)
            self.bm25 = bm25s.BM25(k1=1.5, b=0.75)
            self.bm25.index(self._corpus_tokens, show_progress=False)

    def forward(self, query: str) -> str:
        import bm25s
        self._initialize_bm25()
        if not self.corpus:
            return "No files found to search."
        query_tokens = bm25s.tokenize(query, stopwords=None, show_progress=False)
        results, scores = self.bm25.retrieve(query_tokens, k=min(3, len(self.corpus)), show_progress=False)
        output = []
        # results shape: (1, k), scores shape: (1, k)
        for idx, score in zip(results[0], scores[0]):
            chunk = self.corpus[int(idx)]
            output.append(f"### Score: {score:.4f} | Source: {chunk['filepath']}\n{chunk['content']}\n")
        return "\n".join(output)

class RAGSearchTool(Tool):
    name = "rag_search"
    description = "Vector semantic search. Finds the most semantically relevant document chunks matching the query."
    inputs = {
        "query": {
            "description": "The semantic query.",
            "type": "string"
        }
    }
    output_type = "string"

    def __init__(self, dataset_dir: str, subset_files: List[str] = []):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.subset_files = subset_files
        self.corpus = None
        self.embeddings = None  # np.ndarray shape (N, D)

    def _initialize_rag(self):
        """Load corpus + embeddings, using module-level RAM cache if already built."""
        if self.corpus is not None:
            return
        import numpy as np
        import pickle
        cache_key = (self.dataset_dir, tuple(sorted(self.subset_files)))
        if cache_key in _RAG_CACHE:
            self.corpus, self.embeddings = _RAG_CACHE[cache_key]
            print(f"[RAG Cache] HIT — {len(self.corpus)} chunks loaded from RAM cache")
            return
            
        # Determine workload name from dataset_dir
        workload_name = None
        normalized_dir = self.dataset_dir.replace("\\", "/")
        parts = normalized_dir.split("/")
        for marker in ["data", "workload", "dr-input"]:
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts):
                    workload_name = parts[idx + 1]
                    break
        if not workload_name and len(parts) >= 2:
            workload_name = parts[-2]
            
        if workload_name:
            root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            index_dir = os.path.join(root_path, "repro_program", "data_preprocess", "indexes", workload_name)
            emb_path = os.path.join(index_dir, "rag_embeddings.npy")
            corpus_path = os.path.join(index_dir, "rag_corpus.pkl")
            
            if os.path.exists(emb_path) and os.path.exists(corpus_path):
                print(f"[RAG] Loading pre-built index for workload '{workload_name}' from: {index_dir}")
                try:
                    with open(corpus_path, "rb") as f:
                        full_corpus = pickle.load(f)
                    embeddings = np.load(emb_path)
                    
                    # Filter by subset_files if specified
                    if self.subset_files:
                        filtered_corpus = []
                        filtered_indices = []
                        for idx, chunk in enumerate(full_corpus):
                            fname = chunk["filepath"]
                            if any(f in fname or f in os.path.basename(fname) for f in self.subset_files):
                                filtered_corpus.append(chunk)
                                filtered_indices.append(idx)
                        if filtered_corpus:
                            self.corpus = filtered_corpus
                            self.embeddings = embeddings[filtered_indices]
                        else:
                            self.corpus = []
                            self.embeddings = np.empty((0, embeddings.shape[1]), dtype=np.float32)
                    else:
                        self.corpus = full_corpus
                        self.embeddings = embeddings
                    
                    _RAG_CACHE[cache_key] = (self.corpus, self.embeddings)
                    print(f"[RAG Cache] LOADED PRE-BUILT — {len(self.corpus)} chunks loaded")
                    return
                except Exception as e:
                    print(f"[RAG Loader] Error loading pre-built index: {e}. Falling back to build from scratch.")
                    
        # Build from scratch
        self.corpus = load_dataset_corpus(self.dataset_dir, self.subset_files)
        raw_embeddings = []
        for i, chunk in enumerate(self.corpus):
            emb = get_embedding(chunk["content"])
            raw_embeddings.append(emb)
            if (i + 1) % 50 == 0:
                print(f"[RAG] Embedded {i+1}/{len(self.corpus)} chunks ...")
        self.embeddings = np.array(raw_embeddings, dtype=np.float32)  # (N, D)
        _RAG_CACHE[cache_key] = (self.corpus, self.embeddings)
        print(f"[RAG Cache] STORED — {len(self.corpus)} chunks cached in RAM")

    def forward(self, query: str) -> str:
        import numpy as np
        self._initialize_rag()
        if not self.corpus:
            return "No files found to search."
        query_emb = get_embedding(query)
        if not query_emb:
            return "Failed to compute embedding for query."
        # Vectorized cosine similarity via numpy 
        q = np.array(query_emb, dtype=np.float32)  # (D,)
        sims = self.embeddings @ q                   # (N,)
        top_idx = np.argsort(sims)[::-1][:3]
        output = []
        for idx in top_idx:
            sim = float(sims[idx])
            chunk = self.corpus[int(idx)]
            output.append(f"### Similarity: {sim:.4f} | Source: {chunk['filepath']}\n{chunk['content']}\n")
        return "\n".join(output)

class HybridSemanticThenKeywordSearchTool(Tool):
    name = "hybrid_semantic_then_keyword_search"
    description = "Runs semantic RAG search first; if similarity score is below 0.35, falls back to BM25 keyword search."
    inputs = {
        "query": {
            "description": "The search query.",
            "type": "string"
        }
    }
    output_type = "string"

    def __init__(self, dataset_dir: str, subset_files: List[str] = []):
        super().__init__()
        self.rag = RAGSearchTool(dataset_dir, subset_files)
        self.bm25 = BM25SearchTool(dataset_dir, subset_files)

    def forward(self, query: str) -> str:
        import numpy as np
        self.rag._initialize_rag()
        if not self.rag.corpus:
            return "No files found to search."
        query_emb = get_embedding(query)
        if not query_emb:
            return self.bm25.forward(query)
            
        # Vectorized cosine similarity via numpy
        q = np.array(query_emb, dtype=np.float32)  # (D,)
        sims = self.rag.embeddings @ q               # (N,)
        
        if len(sims) == 0:
            return self.bm25.forward(query)
            
        # Get sorted indices descending
        sorted_indices = np.argsort(sims)[::-1]
        best_idx = int(sorted_indices[0])
        best_sim = float(sims[best_idx])
        
        if best_sim >= 0.35:
            output = []
            for idx in sorted_indices[:3]:
                sim = float(sims[idx])
                chunk = self.rag.corpus[int(idx)]
                output.append(f"### [RAG Search] Similarity: {sim:.4f} | Source: {chunk['filepath']}\n{chunk['content']}\n")
            return "\n".join(output)
        else:
            bm25_res = self.bm25.forward(query)
            return f"### [Fallback to BM25 Search due to low similarity {best_sim:.4f}]\n{bm25_res}"

class LLMKeywordSearchTool(Tool):
    name = "llm_keyword_search"
    description = "Uses an LLM to extract keywords from the query and performs exact keyword matching over document chunks."
    inputs = {
        "query": {
            "description": "The query to extract keywords from and search.",
            "type": "string"
        }
    }
    output_type = "string"

    def __init__(self, dataset_dir: str, subset_files: List[str] = []):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.subset_files = subset_files
        self.corpus = None

    def _initialize_corpus(self):
        if self.corpus is None:
            self.corpus = load_dataset_corpus(self.dataset_dir, self.subset_files)

    def forward(self, query: str) -> str:
        self._initialize_corpus()
        if not self.corpus:
            return "No files found to search."
            
        prompt = f"Extract 1 to 3 important keyword words from this query for exact match search: '{query}'. Output only the keywords separated by spaces, no formatting or extra text."
        messages = [
            {"role": "system", "content": "You are a keyword extraction assistant."},
            {"role": "user", "content": prompt}
        ]
        try:
            keywords_str = call_llm(messages)
            keywords = [k.strip().lower() for k in keywords_str.split() if len(k.strip()) > 2]
        except Exception:
            keywords = [q.lower() for q in query.split() if len(q) > 3]
            
        if not keywords:
            return "No valid search keywords extracted."
            
        matches = []
        for chunk in self.corpus:
            content_lower = chunk["content"].lower()
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                matches.append((score, chunk))
                
        matches.sort(key=lambda x: x[0], reverse=True)
        if not matches:
            return f"No matches found for keywords: {', '.join(keywords)}"
            
        output = []
        for score, chunk in matches[:3]:
            output.append(f"### Matches: {score} | Source: {chunk['filepath']}\n{chunk['content']}\n")
        return "\n".join(output)

# --- Knowledge Mind Map & Hybrid KV ---
class KnowledgeMindMap:
    def __init__(self, filepath: str = None):
        self.filepath = filepath
        self.entries = []
        self.load()

    def load(self):
        if self.filepath and os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.entries = data.get("entries", [])
            except Exception as e:
                print(f"[KnowledgeMindMap] Error loading mind map: {e}")

    def save(self):
        if self.filepath:
            try:
                os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
                with open(self.filepath, "w", encoding="utf-8") as f:
                    json.dump({"entries": self.entries}, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[KnowledgeMindMap] Error saving mind map: {e}")

    def add_entry(self, expression: str, nl_logic: str, result: str, reason: str):
        embedding = get_embedding(nl_logic)
        self.entries.append({
            "expression": expression,
            "nl_logic": nl_logic,
            "result": result,
            "reason": reason,
            "embedding": embedding
        })
        self.save()

    def search(self, query: str, top_k: int = 3, threshold: float = 0.5) -> list:
        if not self.entries:
            return []
        query_emb = get_embedding(query)
        if not query_emb:
            return []
        
        results = []
        for entry in self.entries:
            emb = entry.get("embedding")
            if not emb:
                continue
            sim = sum(q * e for q, e in zip(query_emb, emb))
            if sim >= threshold:
                results.append((sim, entry))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in results[:top_k]]

class HybridKVStore:
    def __init__(self, filepath: str = None):
        self.filepath = filepath
        self.entries = []
        self.load()

    def load(self):
        if self.filepath and os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.entries = data.get("entries", [])
            except Exception as e:
                print(f"[HybridKVStore] Error loading Hybrid KV: {e}")

    def save(self):
        if self.filepath:
            try:
                os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
                with open(self.filepath, "w", encoding="utf-8") as f:
                    json.dump({"entries": self.entries}, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[HybridKVStore] Error saving Hybrid KV: {e}")

    def add_entry(self, query: str, quality: str, success: bool):
        embedding = get_embedding(query)
        self.entries.append({
            "query": query,
            "quality": quality,
            "success": success,
            "embedding": embedding
        })
        self.save()

    def search(self, query: str, top_k: int = 3, threshold: float = 0.5) -> list:
        if not self.entries:
            return []
        query_emb = get_embedding(query)
        if not query_emb:
            return []
        
        results = []
        for entry in self.entries:
            emb = entry.get("embedding")
            if not emb:
                continue
            sim = sum(q * e for q, e in zip(query_emb, emb))
            if sim >= threshold:
                results.append((sim, entry))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in results[:top_k]]

# --- Explorer Agent ---
class Explorer:
    def __init__(self, dataset_dir: str, subset_files: List[str] = [], mind_map_fp: str = None, hybrid_kv_fp: str = None, model: str = "openrouter/deepseek/deepseek-v4-flash"):
        self.dataset_dir = dataset_dir
        self.subset_files = subset_files
        self.mind_map = KnowledgeMindMap(mind_map_fp)
        self.hybrid_kv = HybridKVStore(hybrid_kv_fp)
        self.model = model
        self.rag = RAGSearchTool(dataset_dir, subset_files)

    def parse_logic_to_nl(self, expression: str) -> str:
        prompt = f"Translate the following formal logic expression into a clear, natural language human-readable statement:\nFormal Logic: {expression}\nOutput only the natural language statement."
        messages = [{"role": "user", "content": prompt}]
        try:
            return call_llm(messages, model=self.model).strip()
        except Exception:
            return expression

    def parse_nl_to_logic(self, nl_statement: str) -> str:
        prompt = f"Translate the following natural language logic statement into a formal logic expression (using variables and logical operators like AND, OR, NOT, IF-THEN):\nNatural Language: {nl_statement}\nOutput only the formal logic expression."
        messages = [{"role": "user", "content": prompt}]
        try:
            return call_llm(messages, model=self.model).strip()
        except Exception:
            return nl_statement

    def decompose_to_propositions(self, logic_query: str) -> Tuple[str, dict[str, str]]:
        """
        Decomposes a complex logic query into atomic propositions and a boolean formula.
        """
        prompt = f"""Given a complex logical statement, break it down into atomic propositions (claims that can be evaluated as True or False) and write a formal boolean logic formula representing their combination using AND, OR, NOT, and parentheses.
Example:
Statement: "If the temperature was above 30 degrees and the humidity was high, then it was hot."
Output:
{{
  "formula": "P1 AND P2",
  "propositions": {{
    "P1": "temperature was above 30 degrees",
    "P2": "humidity was high"
  }}
}}

Statement: "{logic_query}"
Respond ONLY with a valid JSON block containing the "formula" and "propositions" keys. Do not include any other text.
"""
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(3):
            res_content = ""
            try:
                res_content = call_llm(messages, model=self.model)
                clean_content = res_content.strip()
                if clean_content.startswith("```"):
                    clean_content = re.sub(r"^```(?:json)?", "", clean_content)
                    clean_content = re.sub(r"```$", "", clean_content)
                clean_content = clean_content.strip()
                
                # Find the JSON block
                start_idx = clean_content.find("{")
                end_idx = clean_content.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    json_str = clean_content[start_idx:end_idx+1]
                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        fixed_str = re.sub(r"'\s*:", '" :', json_str)
                        fixed_str = re.sub(r":\s*'", ': "', fixed_str)
                        fixed_str = re.sub(r"'\s*,", '",', fixed_str)
                        fixed_str = re.sub(r",\s*'", ',"', fixed_str)
                        fixed_str = re.sub(r"'\s*\}", '"}', fixed_str)
                        fixed_str = re.sub(r"\{\s*'", '{"', fixed_str)
                        data = json.loads(fixed_str)
                    
                    formula = data.get("formula", "")
                    propositions = data.get("propositions", {})
                    if formula and propositions:
                        return formula, propositions
            except Exception as e:
                print(f"[Explorer] Decompose attempt {attempt} error: {e}")
                messages.append({"role": "assistant", "content": res_content})
                messages.append({"role": "user", "content": f"The JSON was invalid: {e}. Please output ONLY a valid JSON object."})
                
        return "P1", {"P1": logic_query}

    def evaluate_proposition_semantically(self, prop_text: str) -> Optional[bool]:
        """
        Evaluates a single atomic proposition against retrieved context data.
        Returns True, False, or None (Null/Not enough info).
        """
        # Search Knowledge Mind Map first
        cache_results = self.mind_map.search(prop_text, top_k=1, threshold=0.8)
        if cache_results:
            entry = cache_results[0]
            if entry["result"] == "True":
                return True
            elif entry["result"] == "False":
                return False
            return None

        # Check Hybrid KV
        kv_results = self.hybrid_kv.search(prop_text, top_k=1, threshold=0.7)
        search_query = prop_text
        if kv_results:
            bad_entry = kv_results[0]
            if not bad_entry.get("success", True):
                prompt = f"Reformulate the following search query to make it different and more successful for document retrieval, avoiding keywords that failed in: '{bad_entry['query']}':\nOriginal Query: {prop_text}\nOutput only the reformulated search query."
                messages = [{"role": "user", "content": prompt}]
                try:
                    search_query = call_llm(messages, model=self.model).strip()
                except Exception:
                    pass

        # Execute retrieval
        retrieved_data = self.rag.forward(search_query)

        # Evaluate logic against retrieved data
        prompt = f"""You are a logical evaluation agent. Given the following context data and an atomic proposition, evaluate whether it is True, False, or there is Not enough info to determine.
Context Data:
{retrieved_data}

Atomic Proposition:
{prop_text}

Analyze step-by-step and output your final verdict in the following JSON format:
{{
  "result": "True" | "False" | "Not enough info",
  "reason": "Detailed explanation of your reasoning"
}}
"""
        messages = [{"role": "user", "content": prompt}]
        try:
            res_content = call_llm(messages, model=self.model)
            start = res_content.find("{")
            end = res_content.rfind("}")
            if start != -1 and end != -1 and end > start:
                eval_res = json.loads(res_content[start:end+1])
                verdict = eval_res.get("result")
                reason = eval_res.get("reason", "")
                
                # Log success status to Hybrid KV Store
                success = verdict != "Not enough info"
                quality = "good" if success else "bad"
                self.hybrid_kv.add_entry(search_query, quality, success)
                
                # Update Mind Map
                self.mind_map.add_entry(prop_text, prop_text, verdict, reason)
                
                if verdict == "True":
                    return True
                elif verdict == "False":
                    return False
        except Exception as e:
            print(f"[Explorer] Proposition eval error: {e}")
            
        return None

    def explore_and_evaluate(self, query: str) -> dict:
        # Decompose query into formal logic / propositions
        formula, propositions = self.decompose_to_propositions(query)
        print(f"[Explorer] Decomposed query into Formula: '{formula}' | Propositions: {propositions}")
        
        # Evaluate each atomic proposition
        prop_values = {}
        for key, prop_text in propositions.items():
            val = self.evaluate_proposition_semantically(prop_text)
            prop_values[key] = val
            print(f"[Explorer] Evaluated '{key}' ({prop_text}) -> {val}")
            
        # Evaluate three-valued logic formula
        verdict_val = evaluate_three_valued_logic(formula, prop_values)
        
        # Format the final result
        if verdict_val is True:
            result = "True"
        elif verdict_val is False:
            result = "False"
        else:
            result = "Not enough info"
            
        reason = f"Evaluated boolean formula '{formula}' with Kleene logic values: {prop_values}."
        
        return {
            "result": result,
            "reason": reason,
            "propositions": prop_values
        }

# --- Supervisor Agent ---
class Supervisor:
    def __init__(self, filepath: str = None, model: str = "openrouter/deepseek/deepseek-v4-flash"):
        self.filepath = filepath
        self.model = model
        self.tot_path = []
        self.actor = None
        self.load()

    def attach_actor(self, actor):
        self.actor = actor

    def load(self):
        if self.filepath and os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.tot_path = json.load(f)
            except Exception as e:
                print(f"[Supervisor] Error loading ToT path: {e}")

    def save(self):
        if self.filepath:
            try:
                os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
                with open(self.filepath, "w", encoding="utf-8") as f:
                    json.dump(self.tot_path, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[Supervisor] Error saving ToT path: {e}")

    def record_step(self, step: int, action: str, tool_used: str, tool_input: str, tool_output_summary: str, quality: str, success: bool):
        self.tot_path.append({
            "step": step,
            "action": action,
            "tool_used": tool_used,
            "tool_input": tool_input,
            "tool_output_summary": tool_output_summary[:200] if tool_output_summary else "",
            "quality": quality,
            "success": success
        })
        self.save()

    def judge_path_quality(self) -> dict:
        tot_str = json.dumps(self.tot_path, indent=2)
        prompt = f"""You are a supervisor agent (LLM-as-a-judge). Review the following linear execution path of a coding/reasoning agent:
{tot_str}

Evaluate:
1. Is the agent repeating unproductive loops or making no progress?
2. Is the path quality high or low?
3. Should the agent backtrack to a previous step/state or try a different path?

Respond in the following JSON format:
{{
  "quality": "high" | "low",
  "backtrack_recommended": true | false,
  "recommendation": "Explanation of what the agent should do next (e.g. change search query, try different tool, backtrack to step X)"
}}
"""
        messages = [
            {"role": "system", "content": "You are a path supervisor and judge for autonomous reasoning agents."},
            {"role": "user", "content": prompt}
        ]
        try:
            res_content = call_llm(messages, model=self.model)
            start = res_content.find("{")
            end = res_content.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(res_content[start:end+1])
            return {"quality": "high", "backtrack_recommended": False, "recommendation": "No recommendation"}
        except Exception as e:
            print(f"[Supervisor] Judging error: {e}")
            return {"quality": "high", "backtrack_recommended": False, "recommendation": "No recommendation"}

    def backtrack(self, max_n_steps: int = 3):
        max_n_steps = max(0, min(int(max_n_steps), 3))
        if max_n_steps == 0:
            return

        removed_actor_steps = 0
        if self.actor is not None and hasattr(self.actor, "memory"):
            steps = getattr(self.actor.memory, "steps", [])
            removable = [
                idx for idx, step in enumerate(steps)
                if step.__class__.__name__ in {"ActionStep", "PlanningStep"}
            ]
            for idx in reversed(removable[-max_n_steps:]):
                del steps[idx]
                removed_actor_steps += 1

        removed_tot_steps = min(max_n_steps, len(self.tot_path))
        if removed_tot_steps:
            self.tot_path = self.tot_path[:-removed_tot_steps]

        marker_step = len(self.tot_path) + 1
        self.tot_path.append({
            "step": marker_step,
            "action": "Supervisor backtrack",
            "tool_used": "consult_supervisor",
            "tool_input": f"max_n_steps={max_n_steps}",
            "tool_output_summary": (
                f"Backtracked {removed_actor_steps} actor memory step(s) "
                f"and {removed_tot_steps} supervisor path step(s)."
            ),
            "quality": "backtrack",
            "success": True
        })
        self.save()
        print(
            "[Supervisor] Backtrack applied "
            f"(max_n_steps={max_n_steps}, actor_steps_removed={removed_actor_steps}, "
            f"tot_steps_removed={removed_tot_steps})"
        )





# class Supervisor:
#     def __init__(self, filepath: str = None, model: str = "openrouter/deepseek/deepseek-v4-flash"):
#         self.filepath = filepath

# --- Custom Agent Tools ---
class ExplorerProbingTool(Tool):
    name = "consult_explorer"
    description = (
        "Consult the Explorer agent to verify facts, retrieve/store factual knowledge, and guarantee logical consistency. "
        "Use this tool when you need to confirm facts or save facts to memory (Knowledge Mind Map/file) to keep your reasoning logically sound. "
        "Do NOT use this tool to evaluate execution plans, code steps, or programming flows. "
        "For execution plan review, use 'consult_supervisor' instead."
    )
    inputs = {
        "logic_query": {
            "description": "Factual query about dataset content to check (e.g., 'The dataset contains X entries').",
            "type": "string"
        }
    }
    output_type = "string"

    def __init__(self, explorer_agent: Explorer):
        super().__init__()
        self.explorer = explorer_agent

    def forward(self, logic_query: str) -> str:
        res = self.explorer.explore_and_evaluate(logic_query)
        return json.dumps(res, indent=2)

class SupervisorConsultTool(Tool):
    name = "consult_supervisor"
    description = (
        "Consult the Supervisor agent to audit execution trajectory, path quality, or python errors/tracebacks. "
        "Do NOT use this tool for evaluating factual queries about the dataset (use 'consult_explorer' instead)."
    )
    inputs = {
        "status_summary": {
            "description": "Summary of current execution path or the plan to review.",
            "type": "string"
        }
    }
    output_type = "string"

    def __init__(self, supervisor_agent: Supervisor):
        super().__init__()
        self.supervisor = supervisor_agent

    def forward(self, status_summary: str) -> str:
        step = len(self.supervisor.tot_path) + 1
        self.supervisor.record_step(
            step=step,
            action="Consulting Supervisor",
            tool_used="consult_supervisor",
            tool_input=status_summary,
            tool_output_summary="Consulted",
            quality="good",
            success=True
        )
        res = self.supervisor.judge_path_quality()
        if res.get("backtrack_recommended", False):
            self.supervisor.backtrack()
        return json.dumps(res, indent=2)

# class AIMv2(AIM):
#     def check_facts(self, query: str, M: int = 3) -> str:
#         self.load_facts()
#         self.load_relations()
#         if not self.facts:
#             return "No facts exist in the knowledge graph yet."
            
#         matched = self.search_existing_facts(query, k=1)
#         if not matched:
#             return "No relevant facts found to start checking."
            
#         start_node = matched[0]
#         start_id = start_node["id"]
        
#         visited = set()
#         queue = [(start_id, 0)]
#         collected_ids = []
        
#         while queue:
#             node_id, dist = queue.pop(0)
#             if node_id in visited:
#                 continue
#             visited.add(node_id)
#             collected_ids.append(node_id)
            
#             if dist < M:
#                 neighbors = []
#                 for rel in self.relations:
#                     if rel["from"] == node_id and rel["to"] not in visited:
#                         neighbors.append((rel["to"], dist + 1))
#                     elif rel["to"] == node_id and rel["from"] not in visited:
#                         neighbors.append((rel["from"], dist + 1))
#                 neighbors.sort(key=lambda x: x[0])
#                 queue.extend(neighbors)
                
#         facts_by_id = {f["id"]: f for f in self.facts}
#         # Explicitly limit facts to exactly 5 in AIM
#         output_lines = [f"Retrieved fact chain starting from fact {start_id} (max {M} hops, limited to 5 facts):"]
#         count = 0
#         for fid in collected_ids:
#             if fid in facts_by_id:
#                 f = facts_by_id[fid]
#                 output_lines.append(f"- Fact {fid}: {f['nl_logic']} (Reason: {f['reason']})")
#                 count += 1
#                 if count >= 5:
#                     break
                
#         return "\n".join(output_lines)


# --- SUT System ---

class R3BaselineActorOnly(System):
    """
    R3 baseline system utilizing single LLM Actor coding agent.
    """
    def __init__(self, model: str = "openrouter/deepseek/deepseek-v4-flash", name="R3BaselineActorOnly", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.model = model
        self.dataset_directory = None
        self.verbose = kwargs.get("verbose", False)
        self.max_steps = kwargs.get("max_steps", 30)
        self.output_dir = kwargs.get("output_dir", os.path.join(os.getcwd(), "testresults"))

    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        self.dataset_directory = dataset_directory

    def _init_output_dir(self, query_id: str) -> None:
        self.question_output_dir = os.path.join(self.output_dir, self.name, query_id)
        self.question_intermediate_dir = os.path.join(self.question_output_dir, "_intermediate")
        os.makedirs(self.question_intermediate_dir, exist_ok=True)

class BM25SearchToolV4(BM25SearchTool):
    def forward(self, query: str) -> str:
        self._initialize_bm25()
        if not self.corpus:
            return "No files found to search."
        query_tokens = bm25s.tokenize(query, stopwords=None, show_progress=False)
        # Change limit to top-5
        results, scores = self.bm25.retrieve(query_tokens, k=min(5, len(self.corpus)), show_progress=False)
        output = []
        for idx, score in zip(results[0], scores[0]):
            chunk = self.corpus[int(idx)]
            output.append(f"### Score: {score:.4f} | Source: {chunk['filepath']}\n{chunk['content']}\n")
        return "\n".join(output)
class RAGSearchToolV4(RAGSearchTool):
    def forward(self, query: str) -> str:
        self._initialize_rag()
        if not self.corpus:
            return "No files found to search."
        query_emb = get_embedding(query)
        if not query_emb:
            return "Failed to compute embedding for query."
        q = np.array(query_emb, dtype=np.float32)
        sims = self.embeddings @ q
        # Change limit to top-5
        top_idx = np.argsort(sims)[::-1][:5]
        output = []
        for idx in top_idx:
            sim = float(sims[idx])
            chunk = self.corpus[int(idx)]
            output.append(f"### Similarity: {sim:.4f} | Source: {chunk['filepath']}\n{chunk['content']}\n")
        return "\n".join(output)
class HybridSemanticThenKeywordSearchToolV4(HybridSemanticThenKeywordSearchTool):
    def __init__(self, dataset_dir: str, subset_files: List[str] = []):
        super().__init__(dataset_dir, subset_files)
        self.rag = RAGSearchToolV4(dataset_dir, subset_files)
        self.bm25 = BM25SearchToolV4(dataset_dir, subset_files)

    def forward(self, query: str) -> str:
        self.rag._initialize_rag()
        if not self.rag.corpus:
            return "No files found to search."
        query_emb = get_embedding(query)
        if not query_emb:
            return self.bm25.forward(query)
            
        q = np.array(query_emb, dtype=np.float32)
        sims = self.rag.embeddings @ q
        
        if len(sims) == 0:
            return self.bm25.forward(query)
            
        sorted_indices = np.argsort(sims)[::-1]
        best_idx = int(sorted_indices[0])
        best_sim = float(sims[best_idx])
        
        if best_sim >= 0.35:
            output = []
            # Change limit to top-5
            for idx in sorted_indices[:5]:
                sim = float(sims[idx])
                chunk = self.rag.corpus[int(idx)]
                output.append(f"### [RAG Search] Similarity: {sim:.4f} | Source: {chunk['filepath']}\n{chunk['content']}\n")
            return "\n".join(output)
        else:
            bm25_res = self.bm25.forward(query)
            return f"### [Fallback to BM25 Search due to low similarity {best_sim:.4f}]\n{bm25_res}"
class LLMKeywordSearchToolV4(LLMKeywordSearchTool):
    def forward(self, query: str) -> str:
        self._initialize_corpus()
        if not self.corpus:
            return "No files found to search."
            
        prompt = f"Extract 1 to 3 important keyword words from this query for exact match search: '{query}'. Output only the keywords separated by spaces, no formatting or extra text."
        messages = [
            {"role": "system", "content": "You are a keyword extraction assistant."},
            {"role": "user", "content": prompt}
        ]
        try:
            keywords_str = call_llm(messages)
            keywords = [k.strip().lower() for k in keywords_str.split() if len(k.strip()) > 2]
        except Exception:
            keywords = [q.lower() for q in query.split() if len(q) > 3]
            
        if not keywords:
            return "No valid search keywords extracted."
            
        matches = []
        for chunk in self.corpus:
            content_lower = chunk["content"].lower()
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                matches.append((score, chunk))
                
        matches.sort(key=lambda x: x[0], reverse=True)
        if not matches:
            return f"No matches found for keywords: {', '.join(keywords)}"
            
        output = []
        # Change limit to top-5
        for score, chunk in matches[:5]:
            output.append(f"### Matches: {score} | Source: {chunk['filepath']}\n{chunk['content']}\n")
        return "\n".join(output)
class ExplorerV4(Explorer):
    """
    ExplorerV4: Explorer agent with LLM-as-a-judge caching logic.
    - Lowers the similarity threshold to > 0.75.
    - Searches up to 3 candidate entries from the Knowledge Mind Map.
    - Validates candidates with a lightweight LLM call returning 'True', 'False', or 'Not enough info'.
    - Uses RAGSearchToolV4 (top-5) to fetch datalake document chunks on fallback.
    """
    def __init__(self, dataset_dir: str, subset_files: List[str] = [], mind_map_fp: str = None, hybrid_kv_fp: str = None, model: str = "openrouter/deepseek/deepseek-v4-flash"):
        super().__init__(dataset_dir, subset_files, mind_map_fp, hybrid_kv_fp, model)
        # Use RAGSearchToolV4 (top-5) instead of standard RAGSearchTool (top-3)
        self.rag = RAGSearchToolV4(dataset_dir, subset_files)

    def evaluate_proposition_semantically(self, prop_text: str) -> Optional[bool]:
        # 1. Search Knowledge Mind Map first with a lower threshold (0.75) and up to 3 results
        cache_results = self.mind_map.search(prop_text, top_k=3, threshold=0.75)
        if cache_results:
            candidates_str = ""
            for idx, entry in enumerate(cache_results):
                candidates_str += f"{idx + 1}. Claim: \"{entry['nl_logic']}\" -> Verdict: \"{entry['result']}\"\n"
            
            prompt = f"""You are a logical equivalence validator.
Analyze if the New Claim is logically equivalent (means the exact same thing) or directly contradictory to any of the Cached Claims.

New Claim: "{prop_text}"

Cached Claims:
{candidates_str}

Evaluate carefully:
- If the New Claim is logically equivalent to a Cached Claim with verdict "True", reply "True".
- If the New Claim is logically equivalent to a Cached Claim with verdict "False", reply "False".
- If the New Claim is directly contradictory to a Cached Claim with verdict "True", reply "False".
- If the New Claim is directly contradictory to a Cached Claim with verdict "False", reply "True".
- If none of the cached claims are logically equivalent or contradictory, or if there is not enough information to be absolutely certain, reply "Not enough info".

Output ONLY one of these exact words: "True", "False", or "Not enough info" (without quotes or any other text)."""
            
            messages = [
                {"role": "system", "content": "You are a precise logical evaluation assistant."},
                {"role": "user", "content": prompt}
            ]
            
            try:
                res = call_llm(messages, model=self.model).strip()
                res = re.sub(r"^['\"`]+|['\"`]+$", "", res).strip()
                if res in {"True", "False", "Not enough info"}:
                    print(f"[ExplorerV4 Cache Match] LLM judged cache hit: '{prop_text}' -> {res}")
                    if res == "True":
                        print(f"[R3V4 Log] Premise: '{prop_text[:60]}...' -> Cache Hit: True")
                        return True
                    elif res == "False":
                        print(f"[R3V4 Log] Premise: '{prop_text[:60]}...' -> Cache Hit: False")
                        return False
                    else:
                        pass
            except Exception as e:
                print(f"[ExplorerV4 Cache LLM Error] {e}")
        
        # Fall back to base class
        val = super().evaluate_proposition_semantically(prop_text)
        print(f"[R3V4 Log] Premise: '{prop_text[:60]}...' -> Datalake: {val}")
        return val


# === DCI TOOLS === # Still for windows, unix implementation soon 
class DciFindPythonTool(Tool):
    name = "dci_find"
    description = "Search for files by glob pattern. Returns matching file paths relative to the search directory. Respects .gitignore. Output is truncated."
    inputs = {
        "pattern": {
            "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'.",
            "type": "string"
        },
        "path": {
            "description": "Directory to search in (default: current directory).",
            "type": "string",
            "nullable": True
        },
        "limit": {
            "description": "Maximum number of results (default: 1000).",
            "type": "integer",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, dataset_dir: str):
        super().__init__()
        self.dataset_dir = dataset_dir

    def forward(self, pattern: str, path: str = ".", limit: int = 1000) -> str:
        try:
            search_dir = os.path.normpath(os.path.join(self.dataset_dir, path or "."))
            if not os.path.exists(search_dir):
                return f"Error: Path not found: {path}"
            
            matches = []
            for root, dirnames, filenames in os.walk(search_dir):
                if ".git" in dirnames:
                    dirnames.remove(".git")
                if "node_modules" in dirnames:
                    dirnames.remove("node_modules")

                for filename in filenames:
                    abs_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(abs_path, search_dir).replace("\\", "/")
                    
                    if "/" in pattern:
                        if fnmatch.fnmatch(rel_path, pattern):
                            matches.append(rel_path)
                    else:
                        if fnmatch.fnmatch(filename, pattern):
                            matches.append(rel_path)
                            
                    if len(matches) >= limit:
                        break
                if len(matches) >= limit:
                    break

            if not matches:
                return "No files found matching pattern"
            
            output = "\n".join(matches)
            if len(matches) >= limit:
                output += f"\n\n[{limit} results limit reached]"
            return output
        except Exception as e:
            return f"Error executing find search: {e}"


class DciGrepPythonTool(Tool):
    name = "dci_grep"
    description = "Search file contents for a pattern. Returns matching lines with file paths and line numbers. Respects .gitignore. Output is truncated."
    inputs = {
        "pattern": {
            "description": "Search pattern (regex or literal string).",
            "type": "string"
        },
        "path": {
            "description": "Directory or file to search (default: current directory).",
            "type": "string",
            "nullable": True
        },
        "file_path": {
            "description": "Alternative parameter name for directory or file to search (for backwards compatibility).",
            "type": "string",
            "nullable": True
        },
        "glob": {
            "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'.",
            "type": "string",
            "nullable": True
        },
        "ignoreCase": {
            "description": "Case-insensitive search (default: false).",
            "type": "boolean",
            "nullable": True
        },
        "literal": {
            "description": "Treat pattern as literal string instead of regex (default: false).",
            "type": "boolean",
            "nullable": True
        },
        "context": {
            "description": "Number of lines to show before and after each match (default: 0).",
            "type": "integer",
            "nullable": True
        },
        "limit": {
            "description": "Maximum number of matches to return (default: 100).",
            "type": "integer",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, dataset_dir: str):
        super().__init__()
        self.dataset_dir = dataset_dir

    def forward(self, pattern: str, path: str = None, file_path: str = None, glob: str = None, ignoreCase: bool = False, literal: bool = False, context: int = 0, limit: int = 100) -> str:
        try:
            target = path or file_path or "."
            search_target = os.path.normpath(os.path.join(self.dataset_dir, target))
            if not os.path.exists(search_target):
                return f"Error: Target path {target} does not exist."
            
            regex_flags = re.IGNORECASE if ignoreCase else 0
            search_pattern = re.escape(pattern) if literal else pattern
            try:
                regex = re.compile(search_pattern, regex_flags)
            except re.error as e:
                return f"Invalid regex pattern: {e}"

            results = []
            match_count = 0
            limit_reached = False

            def search_file(filepath):
                nonlocal match_count, limit_reached
                if limit_reached:
                    return

                if glob:
                    rel_to_dataset = os.path.relpath(filepath, self.dataset_dir).replace("\\", "/")
                    if not fnmatch.fnmatch(rel_to_dataset, glob) and not fnmatch.fnmatch(os.path.basename(filepath), glob):
                        return

                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                    
                    for lineno, line_content in enumerate(lines, 1):
                        if regex.search(line_content):
                            rel_path = os.path.relpath(filepath, self.dataset_dir).replace("\\", "/")
                            
                            if context > 0:
                                start = max(1, lineno - context)
                                end = min(len(lines), lineno + context)
                                for c_line in range(start, end + 1):
                                    c_content = lines[c_line - 1].replace("\r", "").replace("\n", "")
                                    if c_line == lineno:
                                        results.append(f"{rel_path}:{c_line}:{c_content}")
                                    else:
                                        results.append(f"{rel_path}-{c_line}- {c_content}")
                            else:
                                results.append(f"{rel_path}:{lineno}:{line_content.strip()}")
                            
                            match_count += 1
                            if match_count >= limit:
                                limit_reached = True
                                break
                except Exception:
                    pass

            if os.path.isfile(search_target):
                search_file(search_target)
            else:
                for root, dirnames, filenames in os.walk(search_target):
                    if ".git" in dirnames:
                        dirnames.remove(".git")
                    if "node_modules" in dirnames:
                        dirnames.remove("node_modules")

                    for filename in filenames:
                        search_file(os.path.join(root, filename))
                        if limit_reached:
                            break
                    if limit_reached:
                        break

            if not results:
                return "No matches found"
            
            output = "\n".join(results)
            if limit_reached:
                output += f"\n\n[{limit} matches limit reached]"
            return output
        except Exception as e:
            return f"Error executing grep search: {e}"



class DciCatPythonTool(Tool):
    name = "dci_cat"
    description = "Read and output the contents of a file (equivalent to Linux 'cat <file_path>')."
    inputs = {
        "file_path": {
            "description": "The relative path to the file to read.",
            "type": "string"
        }
    }
    output_type = "string"

    def __init__(self, dataset_dir: str):
        super().__init__()
        self.dataset_dir = dataset_dir

    def forward(self, file_path: str) -> str:
        try:
            full_path = os.path.normpath(os.path.join(self.dataset_dir, file_path))
            if not os.path.exists(full_path):
                return f"Error: File {file_path} not found."
            
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            
            if len(lines) > 200:
                return "".join(lines[:200]) + f"\n... (Truncated. Total lines: {len(lines)})"
            return "".join(lines)
        except Exception as e:
            return f"Error executing cat: {e}"


# === SMOLAGENTS WRAPPERS ===

@tool
def write_file(path: str, content: str) -> str:
    """
    Writes the given content to a file at the specified path.
    Args:
        path (str): The file path where the content should be written.
        content (str): The content to write to the file.
    Returns:
        str: Confirmation message indicating the file has been written.
    """
    with open(path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(content)
    return f"written to {path}"

@tool
def list_filepaths(dataset_directory:str) -> list[str]:
   """
   This tool lists all of the file paths for relevant files in the data directory.


   Args:
        dataset_directory (str): The path to the dataset directory.
  
   Returns:
        list[str]: A list of file paths for all files in the dataset directory.
   """
   filepaths = []
   for root, _, files in os.walk(dataset_directory):
       for file in files:
           if file.startswith("."):
               continue
           filepaths.append(os.path.join(root, file))
   return filepaths

@tool
def list_input_filepaths(dataset_directory:str, files:list[str]) -> list[str]:
    """
    This tool lists all of the file paths for given files in the data directory.
    Args:
          dataset_directory (str): The path to the dataset directory.
          files (list[str]): A list of file names to look for.
    Returns:
          list[str]: A list of file paths for files found in the dataset directory.
    """
    # Get all file paths
    filepaths = []
    for root, _, files in os.walk(dataset_directory):
        for file in files:
            if file.startswith("."):
               continue
            filepaths.append(os.path.join(root, file))

    # Match given file names to actual file paths
    selected_filepaths = []
    for pattern in files:
        #print(self.dataset.keys())
        #assert f in self.dataset.keys(), f"File {f} is not in dataset!"
        # Relaxed the assertion to a warning
        matching = [
            f for f in filepaths
            if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(os.path.basename(f), os.path.basename(pattern))
        ]
        if len(matching) == 0:
            print(f"WARNING: File {pattern} is not in dataset!")
        else: # only extend if there are matches
            selected_filepaths.extend(matching)
    return selected_filepaths

@tool
def read_csv(
    filepath: str,
    columns: Optional[List[str]] = None,
    n_rows: Optional[int] = None,
    row_indices: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Reads a CSV file and returns a DataFrame.
    
    You can optionally select specific columns, a number of rows, or specific row indices.

    Args:
        filepath (str): The path to the CSV file.
        columns (List[str], optional): List of column names to return.
        n_rows (int, optional): Number of rows to return from the top.
        row_indices (List[int], optional): Specific row indices to return.

    Returns:
        pd.DataFrame: The selected portion of the CSV file.

    Raises:
        ValueError: If the file is not a CSV.
    """
    if not filepath.endswith(".csv"):
        raise ValueError(f"Unsupported file type: {filepath}. Only CSV files are supported.")

    df = pd.read_csv(filepath, encoding="ISO-8859-1")

    if columns is not None:
        df = df[columns]

    if row_indices is not None:
        df = df.iloc[row_indices]
    elif n_rows is not None:
        df = df.head(n_rows)

    return df

@tool
def get_csv_metadata(filepath: str) -> Dict[str, object]:
    """
    Returns metadata about a CSV file, including column names,
    number of rows, and number of columns.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        Dict[str, object]: Metadata dictionary with keys:
            - 'columns': List of column names
            - 'n_rows': Number of rows
            - 'n_columns': Number of columns
            - 'column_types': Data types of each column
    """
    df = pd.read_csv(filepath, encoding="ISO-8859-1")

    metadata = {
        "columns": df.columns.tolist(),
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "column_types": df.dtypes.apply(lambda dt: dt.name).to_dict()
    }

    return metadata

@tool
def summarize_dataframe(file_path: str) -> Dict[str, object]:
    """Summarizes a CSV file by providing metadata and sample data.
    Args:
        file_path (str): Path to the CSV file.
        Returns:
        dict: A summary dictionary containing:
            - file name
            - columns
            - missing values per column
            - data types of each column
            - sample values from the first 3 rows
            - potential type issues (if any)
    """
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    summary = {
        "file": os.path.basename(file_path),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "sample_values": df.head(3).to_dict(orient="list"),
    }

    # inconsistent types check
    type_issues = {}
    for col in df.columns:
        values = df[col].dropna().astype(str)
        if values.nunique() > 0:
            inferred_types = values.map(lambda v: type(eval(v)) if v.isdigit() else str).value_counts()
            if len(inferred_types) > 1:
                type_issues[col] = inferred_types.to_dict()
    if type_issues:
        summary["potential_type_issues"] = type_issues

    return summary

class ExploreDataTool(Tool):
    name = "explore_data"
    description = "Summarize a CSV file: columns, missing values, data types, sample values, and anomalies."

    def __call__(self, file_path: str):
        try:
            df = pd.read_csv(file_path)
            return summarize_dataframe(df, file_path)
        except Exception as e:
            return {"error": str(e)}

import re

def parse_token_counts(trace: str):
    """
    Given the full LLM trace string, extract input/output token counts
    from the final '[Step ...]' line.

    Returns:
        (input_tokens, output_tokens) as integers, or (0, 0) on failure.
    """
    if not isinstance(trace, str):
        print(f"Trace is not a string (type: {type(trace)}) Cannot parse token counts. Returning 0")
        return 0, 0

    # Extract the last line
    last_line = trace.strip().splitlines()[-1]

    # Regex to extract token counts, allowing commas in numbers
    pattern = r"Input tokens:\s*([\d,]+)\s*\|\s*Output tokens:\s*([\d,]+)"

    m = re.search(pattern, last_line)
    if not m:
        print(f"Cannot parse token counts with regex pattern. Returning 0")
        return 0, 0

    # Remove commas and convert to int
    input_tokens = int(m.group(1).replace(",", ""))
    output_tokens = int(m.group(2).replace(",", ""))

    return input_tokens, output_tokens
def init_llm(model_id: str):
    import os
    from systems.smolagents.smolagents_lib.models import OpenAIServerModel, LiteLLMModel
    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    
    is_openai_sdk = (
        model_id.startswith(("openai/", "openrouter/", "ollama/", "openai-server/")) or
        os.environ.get("OPENAI_API_BASE") is not None or
        os.environ.get("OPENAI_BASE_URL") is not None
    )
    
    if is_openai_sdk:
        clean_model_id = model_id
        for prefix in ["openai/", "openrouter/", "ollama/", "openai-server/"]:
            if clean_model_id.startswith(prefix):
                clean_model_id = clean_model_id[len(prefix):]
                break
                
        api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if (model_id.startswith("openrouter/") or model_id.startswith("openai/")) and not api_base:
            api_base = "https://openrouter.ai/api/v1"
        elif model_id.startswith("ollama/") and not api_base:
            api_base = "http://localhost:11434/v1"
            
        model_params = {
            "model_id": clean_model_id,
            "api_base": api_base,
            "api_key": api_key,
            "custom_role_conversions": custom_role_conversions,
            "max_completion_tokens": 8192,
        }
        
        if "deepseek" in clean_model_id.lower() and "openrouter.ai" in (api_base or ""):
            model_params["extra_body"] = {"reasoning": {"enabled": True}}
            
        if "ollama" in model_id.lower():
            model_params["timeout"] = 10800
            
        return OpenAIServerModel(**model_params)
    else:
        model_params = {
            "model_id": model_id,
            "custom_role_conversions": custom_role_conversions,
            "max_completion_tokens": 8192,
        }
        if "ollama" in model_id.lower():
            model_params["timeout"] = 10800
            
        return LiteLLMModel(**model_params)

def _recover_and_write_outputs_v2(agent, agent_output, answer_path: str, pipeline_code_path: str) -> tuple[str, str]:
    """Recover answer and executed python code from agent and save to disk if files are missing."""
    answer = ""
    try:
        if os.path.exists(answer_path):
            with open(answer_path, "r", encoding="utf-8") as f:
                answer = f.read().strip()
    except Exception:
        pass

    if not answer:
        needs_synthesis = False
        if not agent_output or str(agent_output).strip() in ["", "None", "Failed to parse answer."]:
            needs_synthesis = True
        elif hasattr(agent, "step_number") and hasattr(agent, "max_steps") and agent.step_number >= agent.max_steps:
            needs_synthesis = True

        if needs_synthesis:
            try:
                history = []
                if hasattr(agent, "memory") and hasattr(agent.memory, "steps"):
                    for step in agent.memory.steps:
                        step_num = getattr(step, "step_number", "?")
                        model_output = getattr(step, "model_output", "")
                        observations = getattr(step, "observations", "")
                        history.append(f"Step {step_num}:\nThought: {model_output}\nObservation: {observations}\n")
                
                history_text = "\n".join(history)
                prompt = f"""You are a helper that synthesizes the final answer to a task based on the agent's execution history.
The agent reached the step limit or failed to write the output files.

Execution History:
{history_text}

Provide the final answer to the task based on the log of actions above. Do not include any explanation, just return the final answer.
"""
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                model_id = "openrouter/deepseek/deepseek-v4-flash"
                if hasattr(agent, "model") and hasattr(agent.model, "model_id") and agent.model.model_id:
                    model_id = agent.model.model_id
                
                response = call_llm(messages, model=model_id)
                answer = response.strip()
                print(f"[Fallback Recoverer] Synthesized answer via LLM call: {answer}")
            except Exception as e:
                print(f"[Fallback Recoverer] Synthesis error: {e}")
                answer = str(agent_output).strip() if agent_output else "Failed to parse answer."
        else:
            answer = str(agent_output).strip() if agent_output else "Failed to parse answer."

        try:
            os.makedirs(os.path.dirname(answer_path), exist_ok=True)
            with open(answer_path, "w", encoding="utf-8") as f:
                f.write(answer)
        except Exception as e:
            print(f"[Fallback Recoverer] Error writing answer: {e}")

    pipeline_code = ""
    try:
        if os.path.exists(pipeline_code_path):
            with open(pipeline_code_path, "r", encoding="utf-8") as f:
                pipeline_code = f.read().strip()
    except Exception:
        pass

    if not pipeline_code or pipeline_code == "N/A":
        executed_codes = []
        if hasattr(agent, "memory") and hasattr(agent.memory, "steps"):
            for step in agent.memory.steps:
                tool_calls = getattr(step, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        name = getattr(tc, "name", "")
                        if name == "python_interpreter":
                            args = getattr(tc, "arguments", None)
                            if isinstance(args, dict):
                                code = args.get("code", "")
                                if code:
                                    executed_codes.append(code.strip())
                else:
                    model_output = getattr(step, "model_output", None)
                    if model_output and "```python" in model_output:
                        for block in re.findall(r"```python(.*?)```", model_output, re.DOTALL):
                            executed_codes.append(block.strip())
        if executed_codes:
            pipeline_code = "\n\n# --- Step ---\n".join(executed_codes)
        else:
            pipeline_code = "N/A"

        try:
            os.makedirs(os.path.dirname(pipeline_code_path), exist_ok=True)
            with open(pipeline_code_path, "w", encoding="utf-8") as f:
                f.write(pipeline_code)
        except Exception as e:
            print(f"[Fallback Recoverer] Error writing pipeline code: {e}")

    return answer, pipeline_code


SINGLE_AGENT_TASK_PROMPT_TEMPLATE = """
            Workload name: {dataset_name}
            dataset_directory = {dataset_directory}
            Answer the question: {query}; Use the {dataset_name} dataset. 
            Subset files: {subset_files}
            
            IMPORTANT: PRIORITIZE USING SEARCH TOOLS FIRST!
            - You MUST use the provided search tools (`bm25_search`, `rag_search`, `hybrid_semantic_then_keyword_search`, `llm_keyword_search`) to locate the relevant files and specific document chunks first.
            - Do NOT write Python code to load or read entire large CSV, Excel (.xlsx), or text files into memory (e.g. using `pd.read_csv`, `pd.read_excel`, `open().read()`) unless you have first verified which file/subset is relevant via the search tools. Loading entire datasets directly is extremely slow, memory-intensive, and is highly discouraged.
            - You have access to `consult_supervisor` (to check path/logic quality and get recommendations) and `consult_explorer` (to verify logic queries or search the Mind Map). Use them actively to ensure your path is correct.

            If the subset files is not empty, use only those files to answer the question; otherwise, you have to figure out which files to use on your own.
            If you see unusual or unexpected intermediate results, use the critique agent to evaluate the step and provide feedback.
            DO NOT assume the correctness of the intermediate results and final results. Be skeptical and ensure they are correct by cross-checking with the dataset.
            Especially, if you see missing or incomplete intermediate results (e.g., nan or 0.0), question yourself if it's a coding issue, a logic design issue, or an actual data issue.
            After every step, ask yourself if every step is necessary and if the final answer is correct.
            **Report the final answer (just the answer, no explanation needed) AND the complete code pipeline used to get there in your final response, by following the instructions below.**
            IMPORTANT (use the given write_file tool): 
            - Write your final answer to {answer_path}
            - Write your complete code pipeline to {pipeline_code_path}
            - DO NOT write anything else besides using the write_file tool. E.g., NO intermediate results should be written.
            You can end the task after writing the files.
            IMPORTANT: Never generate plots. Do NOT import matplotlib or any visualization library.
            Do NOT call any plotting functions (plt.figure, plt.plot, plt.show, seaborn, plotly, etc).
            The environment is headless and will crash if graphical code is used.
            Return ONLY numerical or textual results.
        """


class R3V4CompleteFullSystem(System):
    """
    system with:
    - ExplorerV4 integrating LLM-validated caching.
    - Automatic backtrack of maximum 2 steps on failed steps.
    - Top-5 search tools for datalake lookup.
    """
    def __init__(self, model: str = "openrouter/deepseek/deepseek-v4-flash", name="R3V4CompleteFullSystem", *args, **kwargs):
        super().__init__(system_name=name, *args, **kwargs)
        self.model = model
        self.dataset_directory = None
        self.verbose = kwargs.get('verbose', False)
        self.max_steps = kwargs.get('max_steps', 40)
        self.output_dir = kwargs.get('output_dir', os.path.join(os.getcwd(), 'testresults'))
        self.supervisor = Supervisor(model=self.model)

    def process_dataset(self, dataset_directory):
        self.dataset_directory = dataset_directory

    def _init_output_dir(self, query_id):
        self.question_output_dir = os.path.join(self.output_dir, self.name, query_id)
        self.question_intermediate_dir = os.path.join(self.question_output_dir, "_intermediate")
        os.makedirs(self.question_intermediate_dir, exist_ok=True)

    def _build_search_tools(self, subset_files: List[str]) -> list:
        # (top-5)
        return [
            BM25SearchToolV4(self.dataset_directory, subset_files),
            RAGSearchToolV4(self.dataset_directory, subset_files),
            HybridSemanticThenKeywordSearchToolV4(self.dataset_directory, subset_files),
            LLMKeywordSearchToolV4(self.dataset_directory, subset_files),
            DciFindPythonTool(self.dataset_directory),
            DciGrepPythonTool(self.dataset_directory),
            DciCatPythonTool(self.dataset_directory),
        ]

    def serve_query(self, query: str, query_id: str = "default-0", subset_files: List[str] = []) -> Dict[str, Any]:
        self._init_output_dir(query_id)
        dataset_name = query_id.split("-")[0]
        answer_path = os.path.join(self.question_intermediate_dir, "answer.txt")
        pipeline_code_path = os.path.join(self.question_intermediate_dir, "pipeline_code.py")

        mind_map_fp = os.path.join(self.question_intermediate_dir, "mind_map.json")
        hybrid_kv_fp = os.path.join(self.question_intermediate_dir, "hybrid_kv.json")
        self.explorer = ExplorerV4(self.dataset_directory, subset_files, mind_map_fp, hybrid_kv_fp, self.model)

        # Override the supervisor's backtrack method to allow unlimited step backtracking
        def custom_backtrack(max_n_steps: int = 2):
            max_n_steps = max(0, int(max_n_steps))
            if max_n_steps == 0:
                return None

            removed_actor_steps = 0
            last_successful_step = None
            
            if self.supervisor.actor is not None and hasattr(self.supervisor.actor, "memory"):
                steps = getattr(self.supervisor.actor.memory, "steps", [])
                removable = [
                    idx for idx, step in enumerate(steps)
                    if step.__class__.__name__ in {"ActionStep", "PlanningStep"}
                ]
                to_remove_indices = removable[-max_n_steps:]
                
                # Identify the last successful step before the backtracked ones
                first_removed_idx = to_remove_indices[0] if to_remove_indices else None
                if first_removed_idx is not None:
                    for idx in range(first_removed_idx - 1, -1, -1):
                        if steps[idx].__class__.__name__ == "ActionStep":
                            last_successful_step = steps[idx]
                            break

                for idx in reversed(to_remove_indices):
                    del steps[idx]
                    removed_actor_steps += 1

            removed_tot_steps = min(max_n_steps, len(self.supervisor.tot_path))
            if removed_tot_steps:
                self.supervisor.tot_path = self.supervisor.tot_path[:-removed_tot_steps]

            marker_step = len(self.supervisor.tot_path) + 1
            self.supervisor.tot_path.append({
                "step": marker_step,
                "action": "Supervisor backtrack",
                "tool_used": "consult_supervisor",
                "tool_input": f"max_n_steps={max_n_steps}",
                "tool_output_summary": (
                    f"Backtracked {removed_actor_steps} actor memory step(s) "
                    f"and {removed_tot_steps} supervisor path step(s)."
                ),
                "quality": "backtrack",
                "success": True
            })
            self.supervisor.save()
            print(
                "[Supervisor] Custom Backtrack applied "
                f"(max_n_steps={max_n_steps}, actor_steps_removed={removed_actor_steps}, "
                f"tot_steps_removed={removed_tot_steps})"
            )
            target_step_num = getattr(last_successful_step, "step_number", "unknown") if last_successful_step else "None"
            print(f"[R3V4 Log] Backtrack target node: Step {target_step_num}")
            return last_successful_step

        self.supervisor.backtrack = custom_backtrack

        def supervisor_step_callback(step_info, agent=None):
            step = step_info.step_number
            action = step_info.model_output or ""
            r3_tools = _extract_tools_from_step(step_info)
            tool_used = r3_tools
            tool_input = ", ".join([str(tc.arguments) for tc in step_info.tool_calls]) if step_info.tool_calls else "None"
            tool_output = step_info.observations or ""
            success = step_info.error is None
            quality = "good" if success else "bad"
            
            self.supervisor.record_step(step, action, tool_used, tool_input, tool_output, quality, success)
            print(f"[R3V4 Step callback] Recorded step {step}, success={success}")

            if step_info.observations is None:
                step_info.observations = ""

            # Handle failed step, automatic backtrack of up to N-steps (2) 
            if not success:
                print(f"[R3V4 Step callback] Step {step} failed! Triggering automatic backtrack of 2 steps.")
                last_step = self.supervisor.backtrack(max_n_steps=2)
                
                try:
                    judgement = self.supervisor.judge_path_quality()
                    recommendation = judgement.get("recommendation", "")
                    truncated_rec = recommendation[:80] + "..." if len(recommendation) > 80 else recommendation
                    print(f"[R3V4 Log] Supervisor feedback: {truncated_rec}")
                except Exception as e:
                    recommendation = f"Error auditing path quality: {e}"

                feedback = f"\n\n### [Supervisor Backtrack Alert]\nStep {step} failed. The Supervisor backtracked the memory. Recommendation: {recommendation}"
                
                if last_step is not None:
                    if last_step.observations is None:
                        last_step.observations = ""
                    last_step.observations += feedback
                    print(f"[R3V4 Step callback] Appended backtrack feedback to the previous successful step's observations.")
                else:
                    step_info.observations = feedback
                return

            # Automated Supervisor path quality judge (for successful steps)
            try:
                judgement = self.supervisor.judge_path_quality()
                recommendation = judgement.get("recommendation", "")
                if recommendation and recommendation != "No recommendation":
                    truncated_rec = recommendation[:80] + "..." if len(recommendation) > 80 else recommendation
                    print(f"[R3V4 Log] Supervisor feedback: {truncated_rec}")
                    feedback = f"\n\n### [Supervisor Feedback (Auto-Audit)]\n{recommendation}"
                    step_info.observations += feedback
                    print(f"[R3V4 Step callback] Injected Supervisor feedback for step {step}")
            except Exception as e:
                print(f"[R3V4 Step callback] Supervisor audit error: {e}")

            # Automated Explorer assumption check
            if tool_output and not step_info.error:
                try:
                    extract_prompt = f"""You are a logical extractor. Look at the agent's action and observations for this step.
Action:
{action}
Observations:
{tool_output}

Identify the single most critical factual assumption or claim being made about the dataset (e.g. 'The dataset has a column named X', 'Roman cities country column uses Greece', etc.).
If no new factual assumption or claim about the data is being made, reply with 'NONE'.
Otherwise, output ONLY the factual statement/claim, nothing else."""
                    messages = [{"role": "user", "content": extract_prompt}]
                    claim = call_llm(messages, model=self.model).strip()
                    if claim and claim != "NONE" and len(claim) > 5:
                        explorer_res = self.explorer.explore_and_evaluate(claim)
                        verdict = explorer_res.get("result", "Not enough info")
                        reason = explorer_res.get("reason", "")
                        explorer_feedback = f"\n\n### [Explorer Fact Verification (Auto-Verify)]\nClaim: '{claim}'\nVerification: {verdict} (Reason: {reason})"
                        step_info.observations += explorer_feedback
                        print(f"[R3V4 Step callback] Injected Explorer verification for claim: '{claim}' -> {verdict}")
                except Exception as e:
                    print(f"[R3V4 Step callback] Explorer auto-verify error: {e}")

        tools = [
            write_file,
            list_input_filepaths,
        ] + self._build_search_tools(subset_files) + [
            ExplorerProbingTool(self.explorer),
            SupervisorConsultTool(self.supervisor),
        ]

        logger_path = os.path.join(self.question_intermediate_dir, f"{query_id}.txt")
        logger = AgentLogger(log_file=logger_path)

        from systems.smolagents.smolagents_system import init_llm
        llm = init_llm(self.model)

        agent = CodeAgent(
            model=llm,
            tools=tools,
            max_steps=self.max_steps,
            verbosity_level=2,
            additional_authorized_imports=["*"],
            logger=logger,
            step_callbacks=[supervisor_step_callback]
        )
        self.supervisor.attach_actor(agent)

        task = SINGLE_AGENT_TASK_PROMPT_TEMPLATE.format(
            dataset_name=dataset_name,
            dataset_directory=self.dataset_directory.replace("\\", "/"),
            query=query,
            subset_files=subset_files,
            answer_path=answer_path.replace("\\", "/"),
            pipeline_code_path=pipeline_code_path.replace("\\", "/"),
        )

        start_time = time.time()
        agent_output = agent.run(task)
        runtime = time.time() - start_time

        answer, pipeline_code = _recover_and_write_outputs_v2(agent, agent_output, answer_path, pipeline_code_path)

        try:
            with open(logger_path, "r", encoding="utf-8") as f:
                trace = f.read()
            inp, out = parse_token_counts(trace)
        except Exception:
            inp, out = 0, 0

        return {
            "id": query_id,
            "runtime": runtime,
            "explanation": {"id": "main-task", "answer": answer},
            "pipeline_code": pipeline_code,
            "token_usage": inp + out,
            "token_usage_input": inp,
            "token_usage_output": out
        }


# =====================================================================
# VARIANTS
# =====================================================================


class R3V4CompleteBaselineActorOnly(System):
    """
    R3V4CompleteBaselineActorOnly: Actor-only variant using V4 (top-5) search tools.
    No Supervisor, no Explorer, no DCI tools, no backtrack.
    """
    def __init__(self, model: str = "openrouter/deepseek/deepseek-v4-flash", name="R3V4CompleteBaselineActorOnly", *args, **kwargs):
        super().__init__(system_name=name, *args, **kwargs)
        self.model = model
        self.dataset_directory = None
        self.verbose = kwargs.get('verbose', False)
        self.max_steps = kwargs.get('max_steps', 40)
        self.output_dir = kwargs.get('output_dir', os.path.join(os.getcwd(), 'testresults'))

    def process_dataset(self, dataset_directory):
        self.dataset_directory = dataset_directory

    def _init_output_dir(self, query_id):
        self.question_output_dir = os.path.join(self.output_dir, self.name, query_id)
        self.question_intermediate_dir = os.path.join(self.question_output_dir, "_intermediate")
        os.makedirs(self.question_intermediate_dir, exist_ok=True)

    def serve_query(self, query: str, query_id: str = "default-0", subset_files: List[str] = []) -> Dict[str, Any]:
        self._init_output_dir(query_id)
        dataset_name = query_id.split("-")[0]
        answer_path = os.path.join(self.question_intermediate_dir, "answer.txt")
        pipeline_code_path = os.path.join(self.question_intermediate_dir, "pipeline_code.py")

        tools = [
            write_file,
            list_input_filepaths,
            BM25SearchToolV4(self.dataset_directory, subset_files),
            RAGSearchToolV4(self.dataset_directory, subset_files),
            HybridSemanticThenKeywordSearchToolV4(self.dataset_directory, subset_files),
            LLMKeywordSearchToolV4(self.dataset_directory, subset_files)
        ]

        logger_path = os.path.join(self.question_intermediate_dir, f"{query_id}.txt")
        logger = AgentLogger(log_file=logger_path)

        from systems.smolagents.smolagents_system import init_llm
        llm = init_llm(self.model)

        agent = CodeAgent(
            model=llm,
            tools=tools,
            max_steps=self.max_steps,
            verbosity_level=2,
            additional_authorized_imports=["*"],
            logger=logger
        )

        task = SINGLE_AGENT_TASK_PROMPT_TEMPLATE.format(
            dataset_name=dataset_name,
            dataset_directory=self.dataset_directory.replace("\\", "/"),
            query=query,
            subset_files=subset_files,
            answer_path=answer_path.replace("\\", "/"),
            pipeline_code_path=pipeline_code_path.replace("\\", "/"),
        )

        start_time = time.time()
        agent_output = agent.run(task)
        runtime = time.time() - start_time

        answer, pipeline_code = _recover_and_write_outputs_v2(agent, agent_output, answer_path, pipeline_code_path)

        try:
            with open(logger_path, "r", encoding="utf-8") as f:
                trace = f.read()
            inp, out = parse_token_counts(trace)
        except Exception:
            inp, out = 0, 0

        return {
            "id": query_id,
            "runtime": runtime,
            "explanation": {"id": "main-task", "answer": answer},
            "pipeline_code": pipeline_code,
            "token_usage": inp + out,
            "token_usage_input": inp,
            "token_usage_output": out
        }

