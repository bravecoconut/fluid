# `fluid/regular_interface/srcf.py`

Self-Referential Context Filtering — a conversation memory system that keeps only the messages most relevant to the current task.

## Class: `SRCF`

### Purpose

When conversations grow long, sending the entire history to the LLM wastes tokens and can dilute focus. SRCF solves this by embedding older messages into a temporary ChromaDB collection and querying them for relevance against the most recent messages. Only semantically relevant older messages are retained.

### Constructor

```python
SRCF(
    srcf_percent=80,
    srcf_last_n=5,
    srcf_threshold=0.323,
    srcf_em_model="modules/regular_interface/embedding_models/BAAI_bge-small-en-v1.5/",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `srcf_percent` | `int` | `80` | Percentage of older messages to consider for filtering (the rest are always kept) |
| `srcf_last_n` | `int` | `5` | Number of recent messages to use as the query for similarity |
| `srcf_threshold` | `float` | `0.323` | Maximum embedding distance for a message to be considered relevant |
| `srcf_em_model` | `str` | (BGE small) | Path to the local embedding model |

### How It Works

1. **Group messages into turns** — Messages are grouped by user turns (a user message + all following assistant/tool messages until the next user message). This prevents tool calls from being separated from their results.

2. **Split by percentage** — The first `srcf_percent%` of turns are the "old" messages; the rest are "recent" and always kept.

3. **Embed old messages** — Non-tool messages from old turns are added to a temporary in-memory ChromaDB collection.

4. **Query with recent context** — The last `srcf_last_n` messages are used as queries against the collection.

5. **Filter by threshold** — Only turns with embedding distance ≤ `srcf_threshold` are kept.

6. **Return** — Returns `(relevant_messages, leftover_messages)`:
   - `relevant_messages` — filtered old messages that are semantically relevant
   - `leftover_messages` — the recent messages that were never filtered (always included)

### Method: `srcf_go(messages)`

**Parameters:**
- `messages` — list of message dicts in OpenAI format

**Returns:** `(relevant_messages, leftover_messages)` — two lists of message dicts

### Internal State

- Uses an in-memory ChromaDB client (no persistence) — the collection is created fresh each time.
- The collection name is a random integer to avoid collisions.
- Embedding is done via `SentenceTransformerEmbeddingFunction`.

### System Message

When SRCF is active, a system message is prepended (from `system_message.py`) explaining to the LLM that some messages may be missing and the retained context is sufficient.
