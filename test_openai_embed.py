import sys
import logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

from fluid.vector_retrive.retriveing_pool import retriving_pool

skills = [
    {
        "name": "OS",
        "default": True,
        "template": "This is your OS skill. Use these tools for file operations, shell commands, downloads, and system tasks.\n{placeholder}",
        "device": "cpu",
        "collections": [
            {
                "name": "sentence_chunks",
                "max_result": 5,
                "meta": None,
                "collec_template": "Relevant context for your task:\n{placeholder}",
                "threshold": 1,
                "em_setup": {
                    "openai_embed": {
                        "base_url": "https://mostly-marc-obtain-wake.trycloudflare.com/v1",
                        "api_key": "ollama",
                        "model": "znbang/bge:large-en-v1.5-f16",
                    },
                },
            }
        ],
        "backend": {
            "persistent_backend": {
                "path": "run_ignore/os_agent_demo",
            },
        },
    },
]

try:
    raw_contexts, contexts = retriving_pool(
        query="Download the file from this URL",
        locations=skills,
    )
    print("Raw Contexts:")
    print(raw_contexts)
    print("Grouped Contexts:")
    print(contexts)
except Exception as e:
    print(f"Error: {e}")
