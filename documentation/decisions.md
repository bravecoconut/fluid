# Fluid — Decision Choices

Every framework involves trade-offs. This document captures the design decisions made across Fluid and leaves space for the reasoning behind each one.

---

## Architecture

### Why a skill-based architecture instead of a flat tool list?

> _Tools are grouped into skills, and skills can be activated/deactivated at runtime. The agent doesn't start with every tool available._

**Why:** agent can start with tools at run time or even add or switch skills on runtime. that the reason it have that `default` flag in skills. whenever you start it, you can choose what should be the defualt skill agent must have at runtime, according to the character of your agent. And agent will get context (with you `template`) and tools according to the default skills. I exists just because model don't get all his tools and context at once, so it can carry 100 of tools without any problem.
<!-- Write your reasoning here -->

---

### Why let the agent manipulate its own skills at runtime?

> _The agent has built-in tools (`add_new_skill`, `remove_skill`, `switch_skill`) to add and remove skills mid-conversation._

**Why:** Same answer as i described above. these inline tools let agent manipulate his skills. so model can activate tools mid task and complete the task without any break.
<!-- Write your reasoning here -->

---

### Why cap `max_new_skill` at a fixed number?

> _There's a hard limit on how many non-default skills the agent can activate at once._

**Why:** This depends on the Language model you are using. and their is no limits on non-default skill in it by default, you have to define that `max_new_skill` in RegularInterface. if your model has long context capabilities than it depend on it. else model will end with doing nothing or wrong task by confusing.
<!-- Write your reasoning here -->

---

### Why separate default and non-default skills?

> _Default skills are always active and their tools are always available. Non-default skills must be explicitly activated._

**Why:** default deines you agent's main character and capabilties, while Non-default skills are surival kit of agent at mid task, that mokes your agent work long running task without any moment. 
<!-- Write your reasoning here -->

---

## Generation

### Why OpenAI-compatible API instead of direct provider SDKs?

> _Fluid uses the `openai` Python client for all LLM calls, pointed at any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter, etc.)._

**Why:** I think OpenAI is a standard choice that support most of poviders like Gemini, Anthropic, HuggingFace and so on. So you also don't need hang with just one provider.
<!-- Write your reasoning here -->

---

### Why streaming instead of blocking calls?

> _All LLM responses are streamed chunk by chunk, yielding events as they arrive._

**Why:** just because for of how want to get updates about whats going on under-the-hood. you look at the taking actions live and can make sure you are getting what you need, and stop in middle if agent is suppose to do wrong.
<!-- Write your reasoning here -->

---

### Why per-stream buffers and stop events?

> _Each stream gets its own `buffer_id`, stop event, and chunk buffer instead of sharing global state._

**Why:** While i was making it, i started from making `fluid/generation/openai.py`. just to get response first, so i thought, it would be better to have tha for future scalling. just to stop agent from generating tokens. chunk buffer is the main reason i implement buffer id in it, that enable us to get full buffer using just buffer id of that stream.
<!-- Write your reasoning here -->

---

### Why yield error dicts instead of raising exceptions during streaming?

> _Operational failures during streaming (bad API key, network drop) yield an error chunk rather than raising._

**Why:** because i don't like Raise. it makes me frustated, while handling it just from yield works same... Both points to Failure.
<!-- Write your reasoning here -->

---

## Tool Execution

### Why subprocess isolation instead of in-process calls?

> _External tools run in child processes (`subprocess.Popen`) rather than being called directly in the agent process._

**Why:** just because we can get all the outputs, input, and other stuffs as they seems like in a ordinary terminal. it keep alive and give you process id so can also track it using your custom logic.
<!-- Write your reasoning here -->

---

### Why configurable timeouts with agent-driven extend/terminate?

> _Tools have timeouts. When a timeout expires, the agent (not the user) decides whether to extend or kill the process._

**Why:** this is to prevent `run and forgot` paradox. while you use any ordinary agent, you have seen that, if the process is going take 100+ hours, your agent will wait for that hour and then comes and see `oh, i though it should work!` then another running task for that much time. this is what let your agent check the tool results mid process and make decision about, weather continue or terminate. 
<!-- Write your reasoning here -->

---

### Why `_venus_timeout` and `_tool_comment` as hidden parameters?

> _Every tool gets two extra parameters injected: `_venus_timeout` (timeout hint) and `_tool_comment` (short description of intent). These are stripped from arguments before execution._

**Why:** _venus_timeout is what let agent pass timeout in function, but i extract it from the args and use it saparetly in tool runner. and _tool_comment so can get what model it going to do and show on you INTERFACE or Site, it gives the simulation that the agent is making texts content in mid reasoning.
<!-- Write your reasoning here -->

---

### Why sequential tool execution instead of parallel?

> _When the model calls multiple tools in one turn, they are executed one at a time, not in parallel._

**Why:** just because i don't hav must time to implement parallel tool call architecture. even this sequential running takes my week of thinking. if you ever have a time, just think about it and implement it.
<!-- Write your reasoning here -->

---

### Why `multiprocessing.Process` with `Manager` queues instead of threading?

> _Tool execution uses `multiprocessing.Process` with `Manager().Queue()` for IPC instead of threads._

**Why:** if you are a real python programmer, than you should know that threading is really not simultanous as it seem. hav you listen about GIL (Global Interpreter Lock), GIL is my nightmare, that make me use subprocess, but mutiprocessing is what makes bypass GIL. agent can't wait any GIL in mid task, agent have to bypass all the limitation of time or barriers.
<!-- Write your reasoning here -->

---

## Tool Discovery

### Why decorator-based registration (`@tool`)?

> _Tools are discovered by scanning a directory for Python files and importing them. Functions marked with `@tool` are automatically registered._

**Why:** i think, that is a single way of doing. we can't pass tools.py in RegularInterface! 
<!-- Write your reasoning here -->

---

### Why scan-and-import over explicit registration?

> _Instead of requiring a list of tool modules, Fluid walks a directory tree and imports every `.py` file it finds._

**Why:** because Fluid is developer friendly, it lets you define tools any where in my project directory and use it seamlessly,. it is better that write list of modules also for tools, with skills. hence skill still need list, but is the only way it can get it, because, in its first version (that i through in a trash) takes json of skills and tools. and writtin a json for any lazy developer is no a friendly way of doing it... developer may get out of control.
<!-- Write your reasoning here -->

---

## Vector Retrieval

### Why ChromaDB?

> _Fluid uses ChromaDB for all vector storage and retrieval — both persistent (on-disk) and HTTP (remote server) modes._

**Why:** because, chromaDB is the sinle database that i run on my PC, Milvus like DBs are too heavy for my PC. if you ever got a time, just you favorite database by yourself.
<!-- Write your reasoning here -->

---

### Why support both persistent and HTTP backends?

> _Each skill's backend can be either an on-disk ChromaDB directory or a remote Chroma server._

**Why:** sometimes, i host my DBs and LLMs of Kaggle, but sometime host DB on my disk at development. this is why it supports both. doesn't matter where is you DB, on you disk or other's disk.
<!-- Write your reasoning here -->

---

### Why parallel retrieval with `multiprocessing.Pool`?

> _All collection queries across all skills are dispatched to a process pool and run simultaneously._

**Why:** so you can use any amount vector databases and collection, not just one. and retrieving from each is little time consumng, that's it uses pool with the same function multple times to search parallely across all your workers (CPU cores).
<!-- Write your reasoning here -->

---

### Why `spawn` context instead of `fork`?

> _The retrieval pool uses `multiprocessing.get_context("spawn")` instead of the default `fork` on Linux._

**Why:** spawn is a safer and cross platform standard. multiprocessing.get_context("spawn") starts a completely fresh Python interpreter from scratch, whereas fork clones the parent process's existing memory space instantly. fork is historically faster, it is highly prone to deadlocks and crashes. because of this, spawn is the safer, cross-platform standard, and Python 3.14 officially dropped raw fork as the default on Linux (i made it on ubuntu) in favor of safer methods like forkserver and spawn.
<!-- Write your reasoning here -->

---

### Why cache embedding functions per worker process?

> _Worker processes cache their embedding functions so the model is only loaded once per worker, not once per query._

**Why:** so we don't need to load model once for database. workers are process, not sandboxes. building an embedding function is the expensive part, not running it. look at what _build_embedding_function does: for SentenceTransformerEmbeddingFunction, "building" it means loading actual model weights off disk (or downloading them) and initializing them onto a device.. that's seconds of work, versus milliseconds for an actual query embedding... this when the _worker_ef_cache exists, it is module level dict. so it lives for the lifetime of the worker process (not the lifetime of a single task). _get_cached_ef builds the embedding function once, keyed on (hf_embedding_model_name, device, base_url, api_key, model), and every subsequent task in that same worker that needs the same config just gets the cached object back instead of rebuilding it.
<!-- Write your reasoning here -->

---

### Why support both HuggingFace local and OpenAI-compatible embeddings?

> _Embedding functions can be built from either a local HuggingFace model or an OpenAI-compatible API._

**Why:** so you can models from you disk or models from any provider.
<!-- Write your reasoning here -->

---

## SRCF (Self-Referential Context Filtering)

### Why filter old messages by embedding similarity?

> _Instead of just truncating old messages, SRCF keeps the ones most semantically relevant to the current task._

**Why:** it is to remove noice from messages. it works well with more than 80 messages, not for short length. truncating may trancate some usefull facts for the user query. 
<!-- Write your reasoning here -->

---

### Why group messages into turns before filtering?

> _Messages are grouped into turns (user message + all following messages) before filtering. This prevents tool calls from being separated from their results._

**Why:** so agent don't get only the query that ranked, but with all the messages including system, tool or etc of that turn of message.
<!-- Write your reasoning here -->

---

### Why an in-memory ChromaDB client for SRCF?

> _SRCF uses `chromadb.Client()` (in-memory) rather than a persistent or HTTP client._

**Why:** using full database for it would be more than enough, thats why use in memory to do it on the fly.
<!-- Write your reasoning here -->

---

### Why a configurable threshold and percentage?

> _SRCF has `srcf_percent` (how much history to consider), `srcf_last_n` (how many recent messages to use as queries), and `srcf_threshold` (max distance for relevance)._

**Why:** just get your perfect combination, which ever combination fits you, keep it.
<!-- Write your reasoning here -->

---

## System Messages

### Why template-based context injection?

> _Skills and collections have `{placeholder}` templates that wrap retrieved context before injecting it into the system prompt._

**Why:** so you can tell your agent about the context use are providing.
<!-- Write your reasoning here -->

---

### Why separate system messages for default vs non-default skills?

> _Two separate system messages are constructed: one for context from default skills, one for the catalog of available non-default skills._

**Why:** giving both at once may be overwelm models.
<!-- Write your reasoning here -->

---

## Error Handling

### Why never-raise semantics in retrieval functions?

> _`chroma_db_persistent` and `chroma_db_http` never raise. They return error dicts so one bad collection doesn't crash the whole retrieval run._

**Why:** i already don't like Raise or crashing my programme.
<!-- Write your reasoning here -->


---



---

### Why `SRCF` (Self-Referential Context Filtering)?

> _The name for the conversation memory filtering system._

**Why:** because it don't need summarization, instead it uses that same messages list to get relevent messages, no new text are created.
<!-- Write your reasoning here -->

---

