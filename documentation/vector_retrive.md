# `fluid/vector_retrive/retrive.py`

Low-level ChromaDB query functions for both persistent (on-disk) and HTTP (remote server) backends.
th
## Functions

### `chroma_db_persistent(...)`

Queries a persistent (on-disk) ChromaDB collection.

```python
chroma_db_persistent(
    query,                    # str — search query
    mid=None,                 # str — unique identifier for this task
    base_url=None,            # str — OpenAI-compatible embed API URL
    api_key=None,             # str — API key for OpenAI-compatible embed
    model=None,               # str — embedding model name
    hf_embedding_model_name=None,  # str — HuggingFace model name/path
    device="cpu",             # str — device for HF model
    path=None,                # str — path to ChromaDB directory
    collection_name=None,     # str — name of the collection
    max_results=8,            # int — max results to return
    meta=None,                # dict — metadata filter
    threshold=0.5,            # float — max distance for results
    ef=None,                  # embedding function (if pre-built)
)
```

**Returns:** `dict` — always, never raises.

```python
# success
{"collection": "...", "results": [...], "mid": "..."}

# failure
{"collection": "...", "results": [], "mid": "...", "error": "..."}
```

### `chroma_db_http(...)`

Queries a ChromaDB collection via HTTP (a running Chroma server). Same interface and return shape as `chroma_db_persistent`, with these additional parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `host` | `str` | Chroma server hostname |
| `port` | `int` | Chroma server port |
| `ssl` | `bool \| None` | Use HTTPS. Defaults to `False` if not specified. |
| `headers` | `dict \| None` | Custom HTTP headers |

## Embedding Function Resolution

Both functions use `_build_embedding_function(...)` to determine which embedder to use, with this priority:

1. **`ef` parameter** — If a pre-built embedding function is passed, it is reused directly. This is the fast path for worker processes.
2. **`hf_embedding_model_name`** — If set, builds a `SentenceTransformerEmbeddingFunction`.
3. **`base_url` + `api_key` + `model`** — If all three are set, builds an `OpenAIEmbeddingFunction`.
4. **None of the above** — Raises `ValueError`.

## Result Filtering

`_extract_and_filter(raw_results, threshold, collection_name)` processes raw ChromaDB output:

- Extracts `ids`, `documents`, `distances`, `metadatas` from the nested result format.
- Validates that all four lists have the same length.
- Filters to only results where `distance <= threshold`.
- Returns a list of `{"id": ..., "document": ..., "distance": ..., "metadata": ...}` dicts.

## Input Validation

`_validate_common_inputs(...)` checks:
- `query` is a non-empty string
- `collection_name` is a non-empty string
- `max_results` is a positive integer
- `threshold` is a number (warns if negative)
- `meta` is a dict or None

## Logging

All operations log to `logs/vector_retrieve.log`.

---

# `fluid/vector_retrive/retriveing_pool.py`

Runs vector retrieval jobs in parallel across worker processes.

## Function: `retriving_pool(query, locations, processes=None)`

The main entry point for parallel retrieval.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | The search query |
| `locations` | `list` | required | List of skill/location configs |
| `processes` | `int \| None` | `None` | Number of worker processes (None = all CPUs) |

**Returns:** `(raw_results, grouped_results)`
- `raw_results` — flat list, one entry per task
- `grouped_results` — nested dict grouped by database/skill → collections

### How It Works

1. **Build tasks** — For each location, for each collection in that location, create a task tuple `(function, kwargs)` pointing to either `chroma_db_persistent` or `chroma_db_http`.

2. **Parallel execution** — All tasks run via `multiprocessing.Pool.map()` using the `"spawn"` context (avoids fork-related deadlocks with loaded models).

3. **Group results** — Results are reorganized from a flat list into a nested structure by skill/database, using `mid` values as keys.

### Location Config Shape

Each location dict in `locations` must have:

```python
{
    "name": "OS",                   # skill name
    "template": "...",              # optional skill-level template
    "device": "cpu",                # optional, default device for collections
    "collections": [
        {
            "name": "sentence_chunks",  # ChromaDB collection name
            "collec_template": "...",   # optional collection-level template
            "max_result": 8,            # max results per query
            "meta": None,               # metadata filter
            "threshold": 1.0,           # distance threshold
            "em_setup": {
                "hf_local": "path/to/model",   # HuggingFace model path
                # OR
                "openai_embed": {
                    "base_url": "...",
                    "api_key": "...",
                    "model": "...",
                },
            },
        },
    ],
    "backend": {
        "persistent_backend": {"path": "path/to/chromadb"},
        # OR
        "http_backend": {"host": "...", "port": 8000, "ssl": False},
    },
}
```

### Worker Process Caching

`_get_cached_ef(...)` caches embedding functions per worker process. The cache key is `(hf_embedding_model_name, device, base_url, api_key, model)`. This avoids reloading the model for every query in the same worker.

### Spawn Context

Uses `multiprocessing.get_context("spawn")` instead of the default `"fork"` on Linux. Forking after a model is loaded can inherit stuck locks and deadlock.

### `TOKENIZERS_PARALLELISM`

Set to `"false"` at module load to prevent HuggingFace tokenizer threads from causing deadlocks during process spawning.

### Error Handling

- Individual locations/collections that fail validation are skipped, not fatal.
- Worker tasks that crash return error dicts, not exceptions.
- If the pool itself fails (rare — e.g., pickling problem), the error is re-raised.

### Logging

Logs to `logs/vector_retrieve.log`.
