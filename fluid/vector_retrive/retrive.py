import logging
import warnings
from typing import Optional
import os
import chromadb
from chromadb.utils import embedding_functions
from fluid.utill import _build_logger

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "vector_retrieve.log")

logger = _build_logger(log_dir=LOG_DIR, log_file=LOG_FILE)


# ---------------------------------------------------------------------------
# INPUT VALIDATION HELPERS
#
# These check that arguments make sense BEFORE we try to use them, so
# mistakes show up as a clear "here's what's wrong" message instead of a
# confusing crash deep inside the chromadb library.
# ---------------------------------------------------------------------------
def _validate_common_inputs(query, collection_name, max_results, threshold, meta):
    if not query or not isinstance(query, str):
        raise ValueError(f"'query' must be a non-empty string, got: {query!r}")

    if not collection_name or not isinstance(collection_name, str):
        raise ValueError(f"'collection_name' must be a non-empty string, got: {collection_name!r}")

    if not isinstance(max_results, int) or max_results <= 0:
        raise ValueError(f"'max_results' must be a positive integer, got: {max_results!r}")

    if not isinstance(threshold, (int, float)):
        raise ValueError(f"'threshold' must be a number, got: {threshold!r}")

    if meta is not None and not isinstance(meta, dict):
        raise ValueError(f"'meta' must be a dict or None, got: {meta!r}")

    # not fatal, just suspicious -- distances are almost always between 0
    # and 2 for the embedding models people commonly use, so a negative
    # threshold would filter out literally everything, which is probably
    # a mistake rather than intentional
    if threshold < 0:
        warnings.warn(
            f"threshold={threshold} is negative -- this will filter out ALL results. "
            "Did you mean a positive number?"
        )


def _build_embedding_function(ef, hf_embedding_model_name, device, base_url, api_key, model):
    """
    Returns the embedding function to use for this query.

    If the caller already built one (`ef`, e.g. a worker process reusing a
    cached model), we just reuse it -- this is what keeps repeated calls
    fast. Otherwise we build a new one from whichever settings were given.
    """
    if ef is not None:
        return ef

    if hf_embedding_model_name:
        if not isinstance(hf_embedding_model_name, str):
            raise ValueError(
                f"'hf_embedding_model_name' must be a string, got: {hf_embedding_model_name!r}"
            )
        logger.info(f"Building HuggingFace embedding function: {hf_embedding_model_name}")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=hf_embedding_model_name,
            device=device,
        )

    if base_url and model and api_key:
        logger.info(f"Building OpenAI-compatible embedding function: {model} @ {base_url}")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            api_base=base_url,
            model_name=model,
        )

    # none of the above matched -- nothing we can build an embedder from
    raise ValueError(
        "No embedding function could be built. Provide either "
        "'hf_embedding_model_name', or all three of 'base_url' + 'api_key' + 'model'."
    )


def _extract_and_filter(raw_results, threshold, collection_name):
    """
    Pulls ids/documents/distances/metadatas out of Chroma's raw response
    and filters by threshold. Checks the four lists actually line up in
    length before zipping them -- if Chroma ever returns something
    malformed, this fails with a clear message instead of silently
    dropping/misaligning data.
    """
    ids = raw_results.get("ids", [[]])[0]
    docs = raw_results.get("documents", [[]])[0]
    dists = raw_results.get("distances", [[]])[0]
    metas = raw_results.get("metadatas", [[]])[0]

    if not (len(ids) == len(docs) == len(dists) == len(metas)):
        raise ValueError(
            f"Chroma returned mismatched result lists for collection {collection_name!r}: "
            f"ids={len(ids)} documents={len(docs)} distances={len(dists)} metadatas={len(metas)}"
        )

    return [
        {"id": id_, "document": doc, "distance": dist, "metadata": md}
        for id_, doc, dist, md in zip(ids, docs, dists, metas)
        if dist <= threshold
    ]


# ---------------------------------------------------------------------------
# PUBLIC FUNCTIONS
# ---------------------------------------------------------------------------
def chroma_db_persistent(
    query: str,
    mid: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    hf_embedding_model_name: Optional[str] = None,
    device: str = "cpu",
    path: Optional[str] = None,
    collection_name: Optional[str] = None,
    max_results: int = 8,
    meta: Optional[dict] = None,
    threshold: float = 0.5,
    ef=None,
) -> dict:
    """
    Queries a persistent (on-disk) ChromaDB collection.

    Always returns a dict shaped like:
        {"collection": ..., "results": [...], "mid": ...}
    On failure, it instead returns:
        {"collection": ..., "results": [], "mid": ..., "error": "..."}
    It never raises -- so one bad call can't crash a caller looping over
    many collections (e.g. retriving_pool).
    """
    logger.info(f"[persistent] START query={query!r} collection={collection_name!r} path={path!r} mid={mid}")

    try:
        _validate_common_inputs(query, collection_name, max_results, threshold, meta)

        if not path or not isinstance(path, str):
            raise ValueError(f"'path' must be a non-empty string, got: {path!r}")

        built_ef = _build_embedding_function(
            ef, hf_embedding_model_name, device, base_url, api_key, model
        )

        client = chromadb.PersistentClient(path=path)
        collec = client.get_collection(name=collection_name, embedding_function=built_ef)

        raw_results = collec.query(
            query_texts=query,
            n_results=max_results,
            where=meta,
        )

        filtered = _extract_and_filter(raw_results, threshold, collection_name)

        logger.info(
            f"[persistent] DONE collection={collection_name!r} "
            f"matched={len(filtered)} mid={mid}"
        )

        return {
            "collection": collection_name,
            "results": filtered,
            "mid": mid,
        }

    except Exception as e:
        # log the full traceback to the file (exc_info=True), but keep the
        # returned error message short and readable
        logger.error(
            f"[persistent] FAILED collection={collection_name!r} path={path!r} error={e}",
            exc_info=True,
        )
        return {
            "collection": collection_name,
            "results": [],
            "mid": mid,
            "error": str(e),
        }


def chroma_db_http(
    query: str,
    mid: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    hf_embedding_model_name: Optional[str] = None,
    device: str = "cpu",
    host: Optional[str] = None,
    port: Optional[int] = None,
    ssl: Optional[bool] = None,
    headers: Optional[dict] = None,
    collection_name: Optional[str] = None,
    max_results: int = 8,
    meta: Optional[dict] = None,
    threshold: float = 0.5,
    ef=None,
) -> dict:
    """
    Queries a ChromaDB collection over HTTP (a running Chroma server).
    Same guarantees as chroma_db_persistent: always returns a dict, never
    raises, logs everything to logs/vector_retrieve.log.
    """
    logger.info(
        f"[http] START query={query!r} collection={collection_name!r} "
        f"host={host!r} port={port!r} mid={mid}"
    )

    try:
        _validate_common_inputs(query, collection_name, max_results, threshold, meta)

        if not host or not isinstance(host, str):
            raise ValueError(f"'host' must be a non-empty string, got: {host!r}")

        if port is None:
            raise ValueError("'port' is required for chroma_db_http")

        if headers is not None and not isinstance(headers, dict):
            raise ValueError(f"'headers' must be a dict or None, got: {headers!r}")

        if ssl is None:
            # not fatal -- just flag it so it's obvious in the logs/terminal
            # rather than silently assuming "no encryption"
            warnings.warn("'ssl' was not specified -- defaulting to False (http, not https).")
            ssl = False

        built_ef = _build_embedding_function(
            ef, hf_embedding_model_name, device, base_url, api_key, model
        )

        client = chromadb.HttpClient(host=host, port=port, ssl=ssl, headers=headers)
        collec = client.get_collection(name=collection_name, embedding_function=built_ef)

        raw_results = collec.query(
            query_texts=query,
            n_results=max_results,
            where=meta,
        )

        filtered = _extract_and_filter(raw_results, threshold, collection_name)

        logger.info(
            f"[http] DONE collection={collection_name!r} "
            f"matched={len(filtered)} mid={mid}"
        )

        return {
            "collection": collection_name,
            "results": filtered,
            "mid": mid,
        }

    except Exception as e:
        logger.error(
            f"[http] FAILED collection={collection_name!r} host={host!r} port={port!r} error={e}",
            exc_info=True,
        )
        return {
            "collection": collection_name,
            "results": [],
            "mid": mid,
            "error": str(e),
        }