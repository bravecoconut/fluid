# fluid/regular_interface/srcf.py

#########################################################
# SRCF stands for 'Self-Referential Context Filtering'. #
#########################################################

import json

from fluid.utill import trim_messages, mid, _group_into_turns
from fluid.generation.openai import OpenAICom

import chromadb
from chromadb.utils import embedding_functions
import random


class SRCF:
    def __init__(
        self,
        srcf_percent=80,
        srcf_last_n=5,
        srcf_threshold=0.323,
        srcf_em_model="modules/regular_interface/embedding_models/BAAI_bge-small-en-v1.5/",
    ):
        self.srcf_percent = srcf_percent
        self.srcf_last_n = srcf_last_n
        self.srcf_threshold = srcf_threshold

        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=srcf_em_model
            )
        )

        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=str(random.randint(1, 999999999)),
            embedding_function=self.embedding_function,
        )

    def srcf_go(self, messages):
        grouped_messages = _group_into_turns(messages=messages)

        first_m, last_m = trim_messages(
            grouped_messages=grouped_messages,
            percent=self.srcf_percent,
        )

        content = []
        ids = []
        metadatas = []

        for index_a, group in enumerate(first_m):
            for index_b, message in enumerate(group):
                if message["role"] not in ("tool", "tool_call") and message.get(
                    "content"
                ):
                    content.append(message["content"])
                    uid = f"{index_a}:{index_b}"

                    ids.append(uid)
                    metadatas.append(
                        {"group": json.dumps(group), "index": index_a}
                    )  # using group as metadata, so don't need to filter or clean again. and direcly get that group in results

        self.collection.add(
            documents=content,
            ids=ids,
            metadatas=metadatas,
        )

        contents = []
        for message in reversed(messages):
            if len(contents) < self.srcf_last_n:
                contents.append(message["content"])
            else:
                break

        results = self.collection.query(
            query_texts=contents,
            n_results=len(
                first_m
            ),  # to get all the results and get filtered using threshold
        )

        print(json.dumps(results, indent=4))

        _results_groups = []

        _results_groups = {}

        for query_metadatas, query_distances in zip(
            results["metadatas"], results["distances"]
        ):
            for metadata, distance in zip(query_metadatas, query_distances):
                index = metadata["index"]
                if (
                    index not in _results_groups
                    or distance < _results_groups[index]["distance"]
                ):
                    _results_groups[index] = {
                        "group": json.loads(metadata["group"]),
                        "index": index,
                        "distance": distance,
                    }

        _results_groups = sorted(_results_groups.values(), key=lambda x: x["index"])
        print(json.dumps(_results_groups, indent=4))

        relevant_messages = []

        for group in _results_groups:
            if group["distance"] > self.srcf_threshold:
                continue
            relevant_messages.extend(group["group"])

        print(json.dumps(relevant_messages, indent=4))

        print(len(messages))
        print(len(relevant_messages))
        print(len(last_m))

        leftover_messages=[]
        for item in last_m:
            if isinstance(item, list):
                leftover_messages.extend(item)
            else:
                leftover_messages.append(item)


        return relevant_messages, leftover_messages
