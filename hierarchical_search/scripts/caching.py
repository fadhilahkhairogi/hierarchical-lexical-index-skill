import os
import json
import hashlib
import litellm
from typing import List, Dict, Any, Optional
try:
    from hierarchical_index import get_embedding
except ImportError:
    try:
        from .hierarchical_index import get_embedding
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from hierarchical_index import get_embedding

def load_env_robust():
    from dotenv import load_dotenv
    curr = os.path.abspath(__file__)
    for _ in range(10):
        curr = os.path.dirname(curr)
        env_path = os.path.join(curr, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            return
        krama_env = os.path.join(curr, "KramaBench-R3", ".env")
        if os.path.exists(krama_env):
            load_dotenv(krama_env)
            return
        krama_env2 = os.path.join(curr, "RESEARCH", "KramaBench", "KramaBench-R3", ".env")
        if os.path.exists(krama_env2):
            load_dotenv(krama_env2)
            return
    load_dotenv()

load_env_robust()

def call_llm(messages: list, model: str = "openrouter/deepseek/deepseek-v4-flash") -> str:
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

class KVCache:
    def __init__(self, cache_file: str = "kv_cache.json"):
        self.cache_file = cache_file
        self.cache: Dict[str, str] = {}
        self.load()

    def load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"[KVCache] Load error: {e}")

    def save(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[KVCache] Save error: {e}")

    def _get_history_key(self, messages: List[Dict[str, str]]) -> str:
        serialized = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def get(self, messages: List[Dict[str, str]]) -> Optional[str]:
        key = self._get_history_key(messages)
        return self.cache.get(key)

    def set(self, messages: List[Dict[str, str]], response: str):
        key = self._get_history_key(messages)
        self.cache[key] = response
        self.save()


class SearchCache:
    def __init__(self, cache_file: str = "search_cache.json", model: str = "openrouter/deepseek/deepseek-v4-flash"):
        self.cache_file = cache_file
        self.model = model
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"[SearchCache] Load error: {e}")

    def save(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[SearchCache] Save error: {e}")

    def canonicalize(self, query: str) -> str:
        query_clean = query.strip().lower()
        if not query_clean:
            return ""

        query_emb = get_embedding(query_clean)
        best_canonical = None
        best_sim = -1.0
        
        for canonical, info in self.cache.items():
            emb = info.get("embedding")
            if emb:
                sim = sum(q * e for q, e in zip(query_emb, emb))
                if sim > best_sim:
                    best_sim = sim
                    best_canonical = canonical
                    
        if best_sim >= 0.85 and best_canonical:
            return best_canonical

        prompt = (
            f"You are a query standardizer. Simplify the following search query into a concise, standard "
            f"canonical form (e.g. 'Arizona attractions' or 'Roman cities list'). "
            f"Remove conversational filler, details, and formatting. "
            f"Query: '{query}'\n"
            f"Canonical Form:"
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            canonical = call_llm(messages, model=self.model).strip().strip("'\"")
        except Exception:
            canonical = query_clean
            
        if canonical not in self.cache:
            self.cache[canonical] = {
                "results": None,
                "embedding": get_embedding(canonical)
            }
            self.save()
            
        return canonical

    def get(self, canonical_query: str) -> Optional[List[str]]:
        if canonical_query in self.cache:
            return self.cache[canonical_query].get("results")
        return None

    def set(self, canonical_query: str, results: List[str]):
        if canonical_query not in self.cache:
            self.cache[canonical_query] = {
                "embedding": get_embedding(canonical_query)
            }
        self.cache[canonical_query]["results"] = results
        self.save()
