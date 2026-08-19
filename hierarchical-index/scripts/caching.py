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
    for _ in range(5):
        curr = os.path.dirname(curr)
        env_path = os.path.join(curr, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            return
    load_dotenv()

load_env_robust()

def call_llm(messages: list, model: str = "deepseek/deepseek-v4-flash") -> str:
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://opencode.ai/zen/go/v1")
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

class SearchCache:
    def __init__(self, cache_file: str = "search_cache.json", model: str = "deepseek/deepseek-v4-flash"):
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
                    
        if best_sim >= 0.90 and best_canonical:
            return best_canonical

        prompt = (
            f"You are a query standardizer. Simplify the following search query into a concise, standard "
            f"canonical form (e.g. 'Arizona attractions' or 'Roman cities list'). "
            f"Remove conversational filler. Return ONLY the canonical search query text without any labels, prefix, or formatting.\n"
            f"Query: '{query}'"
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
