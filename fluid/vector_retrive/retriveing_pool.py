# IMPORTANT: this must be set before chromadb/torch/transformers get imported
# anywhere in this process. It stops HuggingFace's tokenizer library from
# spinning up background threads that can deadlock when the process forks
# to create worker processes later. Keep this as the very first lines of
# this file.
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import multiprocessing as mp
from typing import Optional

from chromadb.utils import embedding_functions
from fluid.vector_retrive.retrive import chroma_db_http, chroma_db_persistent
from fluid.utill import mid, _build_logger


LOG_DIR = "./logs"
LOG_FILE = os.path.join(LOG_DIR, "vector_retrieve.log")


logger = _build_logger(log_dir=LOG_DIR, log_file=LOG_FILE)

# ---------------------------------------------------------------------------
# Why "spawn" instead of the default "fork" on Linux -- see earlier notes:
# forking after a model is already loaded can inherit stuck locks and hang
# forever with no error. "spawn" starts each worker completely fresh.
# ---------------------------------------------------------------------------
_ctx = mp.get_context("spawn")

_worker_ef_cache = {}


def _get_cached_ef(hf_embedding_model_name, device, base_url, api_key, model):
    """
    Returns a cached embedding function for this worker process, building
    it only the first time it's needed.
    """
    cache_key = (hf_embedding_model_name, device, base_url, api_key, model)

    if cache_key in _worker_ef_cache:
        return _worker_ef_cache[cache_key]

    ef = None

    if hf_embedding_model_name:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=hf_embedding_model_name,
            device=device,
        )

    if base_url and model and api_key:
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            api_base=base_url,
            model_name=model,
        )

    _worker_ef_cache[cache_key] = ef
    return ef


def _run_task(task):
    """
    Runs one retrieval job inside a worker process. Must stay a top-level
    function (not a lambda or inner function) so it can be pickled and
    sent to worker processes.

    Never raises -- always returns a dict, even on failure, so one bad
    task can never take down the whole pool.map() call.
    """
    try:
        func, kwargs = task
    except (TypeError, ValueError):
        logger.error(f"_run_task: received a malformed task, expected (func, kwargs): {task!r}")
        return {"error": "malformed task", "collection": None, "mid": None}

    # --- BUG FIX vs. the previous version ---
    # Building the embedding function used to happen OUTSIDE any try/except.
    # If it failed (e.g. a bad model name, or Ollama unreachable), the
    # exception would escape _run_task entirely and crash pool.map() for
    # EVERY task, not just this one -- even though chroma_db_persistent /
    # chroma_db_http themselves were already hardened to never raise. This
    # closes that gap.
    try:
        ef = _get_cached_ef(
            hf_embedding_model_name=kwargs.get("hf_embedding_model_name"),
            device=kwargs.get("device", "cpu"),
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model"),
        )
    except Exception as e:
        logger.error(
            f"_run_task: failed to build embedding function for "
            f"collection={kwargs.get('collection_name')!r}: {e}",
            exc_info=True,
        )
        return {
            "error": f"embedding function build failed: {e}",
            "collection": kwargs.get("collection_name"),
            "mid": kwargs.get("mid"),
        }

    try:
        return func(**kwargs, ef=ef)
    except Exception as e:
        logger.error(
            f"_run_task: task failed for collection={kwargs.get('collection_name')!r}: {e}",
            exc_info=True,
        )
        return {
            "error": str(e),
            "collection": kwargs.get("collection_name"),
            "mid": kwargs.get("mid"),
        }


def _add_collection_task(
    location, collection, collection_index, query, mids, registry, tasks,
    default_device, db_id, persistent_cfg, http_cfg,
):
    """
    Validates one collection entry and appends its task to `tasks` /
    `registry`. Raises on bad config -- the caller (`_add_location_tasks`)
    decides whether to skip it or stop everything.
    """
    if not isinstance(collection, dict) or "name" not in collection:
        raise ValueError(f"collection #{collection_index} must be a dict with a 'name' key")

    if "em_setup" not in collection or not isinstance(collection["em_setup"], dict):
        raise ValueError(f"collection {collection.get('name', '?')!r} is missing an 'em_setup' dict")

    em_setup = collection["em_setup"]
    openai_embed = em_setup.get("openai_embed") or {}
    collection_mid = mid(mids=mids, length=8)

    registry.append({
        "db_id": db_id,
        "db_info": {
            "name": location["name"],
            "template": location.get("template", ""),
        },
        "collection_info": {
            "mid": collection_mid,
            "collection_name": collection["name"],
            "collec_template": collection.get("collec_template", ""),
        },
    })

    common_kwargs = dict(
        query=query,
        mid=collection_mid,
        base_url=openai_embed.get("base_url"),
        api_key=openai_embed.get("api_key"),
        model=openai_embed.get("model"),
        hf_embedding_model_name=em_setup.get("hf_local"),
        device=collection.get("device", default_device),
        collection_name=collection["name"],
        max_results=collection.get("max_result", 8),
        meta=collection.get("meta"),
        threshold=collection.get("threshold", 1.0),
    )

    # NOTE: this is if/elif, not two separate ifs -- if a location has
    # BOTH persistent_backend and http_backend configured, only the
    # persistent one runs for that collection. This is a deliberate change
    # from earlier versions (which ran both). Flagging it clearly since
    # it's an easy thing to not notice: if you actually want both backends
    # queried per collection, this needs to go back to two separate `if`
    # blocks -- just say the word and I'll switch it back.
    if persistent_cfg:
        if "path" not in persistent_cfg:
            raise ValueError(f"persistent_backend for {location['name']!r} is missing 'path'")
        tasks.append((chroma_db_persistent, {**common_kwargs, "path": persistent_cfg["path"]}))

    elif http_cfg:
        if "host" not in http_cfg or "port" not in http_cfg:
            raise ValueError(f"http_backend for {location['name']!r} is missing 'host' or 'port'")
        tasks.append((
            chroma_db_http,
            {
                **common_kwargs,
                "host": http_cfg["host"],
                "port": http_cfg["port"],
                "ssl": http_cfg.get("ssl", False),
                "headers": http_cfg.get("headers"),
            },
        ))


def _add_location_tasks(location, location_index, query, mids, registry, tasks):
    """
    Validates one location entry and builds tasks for all its collections.
    Raises on bad top-level config -- the caller decides whether to skip
    the whole location or stop everything.
    """
    if not isinstance(location, dict):
        raise ValueError(f"location #{location_index} must be a dict, got: {type(location).__name__}")

    if "name" not in location:
        raise ValueError(f"location #{location_index} is missing required key 'name'")

    if "collections" not in location or not isinstance(location["collections"], list) or not location["collections"]:
        raise ValueError(f"location {location['name']!r} must have a non-empty 'collections' list")

    if "backend" not in location or not isinstance(location["backend"], dict):
        raise ValueError(f"location {location['name']!r} is missing a 'backend' dict")

    backend = location["backend"]
    persistent_cfg = backend.get("persistent_backend")
    http_cfg = backend.get("http_backend")

    if not persistent_cfg and not http_cfg:
        raise ValueError(f"location {location['name']!r} has no 'persistent_backend' or 'http_backend' configured")

    default_device = location.get("device", "cpu")
    db_id = mid(mids=mids, length=6)

    for collection_index, collection in enumerate(location["collections"]):
        try:
            _add_collection_task(
                location, collection, collection_index, query, mids, registry, tasks,
                default_device, db_id, persistent_cfg, http_cfg,
            )
        except Exception as e:
            # one bad collection inside an otherwise-fine location
            # shouldn't stop the other collections in it from running
            logger.warning(
                f"Skipping collection #{collection_index} in location "
                f"{location['name']!r} due to config error: {e}"
            )
            continue


def _group_results(results, registry):
    """
    Reorganizes the flat `results` list into a nested structure grouped by
    database -> collections, using each result's `mid` to find which
    registry entry it belongs to.

    Uses a dict lookup instead of re-scanning the whole registry for every
    result -- same outcome, much faster once you have many collections.
    """
    grouped_results = {}
    registry_by_mid = {entry["collection_info"]["mid"]: entry for entry in registry}

    for result in results:
        entry = registry_by_mid.get(result.get("mid"))

        if entry is None:
            logger.warning(
                f"Result with mid={result.get('mid')!r} did not match any "
                "known collection -- dropping it from grouped results."
            )
            continue

        bucket = grouped_results.setdefault(entry["db_id"], {
            "skill_name": entry["db_info"]["name"],
            "skill_template": entry["db_info"]["template"],
            "db_collections": [],
        })

        bucket["db_collections"].append({
            "name": entry["collection_info"]["collection_name"],
            "collec_template": entry["collection_info"]["collec_template"],
            "results_contexts": result.get("results", []),
            "_debug": result,
        })

    return grouped_results


def retriving_pool(query: str, locations: list, processes: Optional[int] = None):
    """
    Runs every (collection x backend) retrieval job in parallel across
    worker processes. Returns (raw_results, grouped_results):
      - raw_results: flat list, one entry per task, in task order
      - grouped_results: same data reorganized by database -> collections

    Malformed individual locations/collections are skipped (and logged)
    rather than crashing the whole run. Fundamentally wrong input (wrong
    types for query/locations/processes) raises immediately with a clear
    message, since that's a programming mistake worth catching loudly
    rather than silently working around.
    """
    if not query or not isinstance(query, str):
        raise ValueError(f"'query' must be a non-empty string, got: {query!r}")

    if not isinstance(locations, list):
        raise ValueError(f"'locations' must be a non-empty list, got: {locations!r}")

    if processes is not None and (not isinstance(processes, int) or processes <= 0):
        raise ValueError(f"'processes' must be a positive integer or None, got: {processes!r}")

    logger.info(f"retriving_pool START query={query!r} locations_count={len(locations)} processes={processes}")

    mids = []
    registry = []
    tasks = []

    for location_index, location in enumerate(locations):
        try:
            _add_location_tasks(location, location_index, query, mids, registry, tasks)
        except Exception as e:
            name = location.get("name", "?") if isinstance(location, dict) else repr(location)
            logger.warning(f"Skipping location #{location_index} ({name!r}) due to config error: {e}")
            continue

    if not tasks:
        logger.warning("retriving_pool: no valid tasks were built -- returning empty results.")
        return [], {}

    logger.info(f"retriving_pool: dispatching {len(tasks)} task(s) across up to {processes or 'all'} worker(s).")

    try:
        with _ctx.Pool(processes=processes) as pool:
            results = pool.map(_run_task, tasks)
    except Exception as e:
        # this is bigger than one task failing -- the pool itself couldn't
        # run (rare, e.g. a pickling problem). No partial result to salvage
        # here, so log it fully and re-raise rather than pretending it
        # succeeded with empty results.
        logger.error(f"retriving_pool: worker pool failed entirely: {e}", exc_info=True)
        raise

    grouped_results = _group_results(results, registry)

    ok_count = sum(1 for r in results if "error" not in r)
    logger.info(
        f"retriving_pool DONE: {ok_count} succeeded, {len(results) - ok_count} failed, "
        f"out of {len(results)} task(s)."
    )

    return results, grouped_results
