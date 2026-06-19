# checklist system_1: OpenCode Agent Skill & Slide Specifications

## 1. OpenCode / AgentSkills Template Checklist

*   **[x] Skill Root Directory**: Folder named `system_1` exists in `systems/`.
*   **[x] YAML Metadata Frontmatter (`SKILL.md`)**:
    *   `name`: `hierarchical_index` (valid string matching skill path/intent).
    *   `description`: Fully descriptive text highlighting hierarchical index, search, summarize, expand, KV caching, and search caching.
    *   `license`: `MIT`
    *   `metadata`: Contains `author` and `version`.
*   **[x] Markdown Body (`SKILL.md`)**: Clear documentation, feature summaries, and code usage blocks.
*   **[x] Implementation Directory (`scripts/`)**:
    *   `hierarchical_index.py`: Core logic for tree building and search/summarize/expand methods.
    *   `caching.py`: Core logic for prefix KV caching and canonical query search caching.
    *   `tools.py`: Main tool wrapper exposing functions for agent imports.
*   **[x] Supporting Resource Folders**:
    *   `references/` directory created (empty, template compliant).
    *   `assets/` directory created (empty, template compliant).

---

## 2. Slide Specifications Checklist (D:\Fadhil\Pelajaran\Riset\ACE\RESEARCH\Agentic_AI\Week_1\SS_slide)

*   **[x] Slide 1, 14: Lexical Retrieval Advantage**:
    *   *Specification:* Evidence is structured in cells/fields; lexical terms can connect files efficiently.
    *   *Implementation:* Hierarchical Index structures tables into columns/fields and indexes cell sample values to preserve cell-level evidence.
*   **[x] Slide 2, 8, 9, 11, 12, 15: KV Cache (Conversational turn cache)**:
    *   *Specification:* Caches conversational turns (User/Assistant histories) to speed up model runs and hit cached completions.
    *   *Implementation:* `KVCache` implements prefix conversational turn caching (SHA-256 history hashing and responses retrieval).
*   **[x] Slide 2, 8, 9, 11, 12, 15: Search Cache (Query Canonicalization & Result Cache)**:
    *   *Specification:* Maps query variations (e.g. "Arizona places to visit" -> "Arizona attractions") to a single canonical query and caches results.
    *   *Implementation:* `SearchCache` uses a hybrid semantic search similarity (offline-first) and `deepseek-v4-flash` (online-fallback) to standardize query representation and cache matching node path results.
*   **[x] Slide 3, 4, 5, 10: Index Interface API**:
    *   *Specification:* Avoid thick APIs. Expose predictable, thin Unix-like APIs:
        *   `search(query, scope, granularity, top_k) -> List[Node]`
        *   `summarize(scope, top_k) -> List[str]`
        *   `expand(scope, granularity, sample_size) -> List[Node]`
    *   *Implementation:* `HierarchicalIndex` implements all three methods matching exact naming, parameters, defaults, and semantic/lexical logic.
*   **[x] Slide 5: Hierarchical Index Structure**:
    *   *Specification:* Logical paths: `data` -> `data::table_name` -> `data::table_name::column_name`.
    *   *Implementation:* `HierarchicalIndex._build_index` dynamically walks directories, extracts sheets/columns, and structures them exactly as `data::[table_name]::[column_name]`.
*   **[x] Slide 7, 13: Tool & Agent Feedback Loop**:
    *   *Specification:* Agent interacts with the Tool/Index iteratively, passing keywords and receiving files.
    *   *Implementation:* Implemented as a importable library in `tools.py` allowing the OpenCode agent to call these APIs programmatically at any step.
