import logging
from langchain.docstore.document import Document
import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_experimental.graph_transformers.llm import UnstructuredRelation
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks
from langchain_aws import ChatBedrock
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers.json import JsonOutputParser 
import boto3
import google.auth

system_prompt_parts = [
    "You are an algorithm designed to extract information in structured formats to build a knowledge graph. ",
    "Your task is to identify the entities and relations requested with the user prompt from a given text. ",
    "You must generate the output in JSON format containing a list with JSON objects. ",
    'Each object should have the keys: "head", "head_type", "relation", "tail", "tail_type". ',
    'The "head" key must contain the text of the extracted entity with one type. ',
    "Attempt to extract as many entities and relations as you can. ", 
    "Maintain Entity Consistency: When extracting entities, it's vital to ensure consistency. ", 
    'If an entity, such as "Haidar Jupiter Fund", is mentioned multiple times in the text but is referred to by different names or pronouns ', 
    '(e.g. "the Fund", "us", "we", "our"), always use the most complete identifier for that entity. ',
    'The knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial. ',
    "IMPORTANT NOTES: \n - Don't add any explanation and text."
]

GRAPH_BUILDER_SYSTEM_PROMPT = "".join(system_prompt_parts)
system_message = SystemMessage(content = GRAPH_BUILDER_SYSTEM_PROMPT)

GRAPH_BUILDER_HUMAN_PROMPT = """Based on the following example, extract entities and 
        relations from the provided text. Attempt to extract as many entities and relations as you can.

        Below are a number of examples of text and their extracted entities and relationships.
        {examples}

        For the following text or table, extract entities and relations as in the provided example. 
        {format_instructions}
        Text: {input} 
        IMPORTANT NOTES:
        - Each key must have a valid value, 'null' is not allowed. """

EXAMPLES = [
    {
        "text": (
            "Haidar Jupiter Fund ('the Fund') is an opportunistic, "
            "global-macro fund that takes exposure to liquid asset markets. "
        ),
        "head": "Haidar Jupiter Fund",
        "head_type": "Fund",
        "relation": "USES_STRATEGY",
        "tail": "global macro",
        "tail_type": "Strategy",
    },
    {
        "text": (
            "Haidar Jupiter Fund ('the Fund') is an opportunistic, "
            "global-macro fund that takes exposure to liquid asset markets. "
        ),
        "head": "Haidar Jupiter Fund",
        "head_type": "Fund",
        "relation": "HAS_EXPOSURE_TO",
        "tail": "Liquid Asset Markets",
        "tail_type": "Market",
    },
    {
        "text": (
            "The ASV Fund (“Fund”) is a multi-manager, multi-strategy, Asia focused fund which seeks absolute returns across market conditions."
        ),
        "head": "ASV Fund",
        "head_type": "Fund",
        "relation": "HAS_CHARACTERISTIC",
        "tail": "multi-manager",
        "tail_type": "Characteristic",
    },
    {
        "text": "The ASV Fund (“Fund”) is a multi-manager, multi-strategy, Asia focused fund which seeks absolute returns across market conditions.",
        "head": "ASV Fund",
        "head_type": "Fund",
        "relation": "HAS_CHARACTERISTIC",
        "tail": "multi-strategy",
        "tail_type": "Characteristic",
    },
    {
        "text": "Microsoft Word is a lightweight app that accessible offline",
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "HAS_CHARACTERISTIC",
        "tail": "accessible offline",
        "tail_type": "Characteristic",
    },
]

parser = JsonOutputParser(pydantic_object = UnstructuredRelation)
human_prompt = PromptTemplate(template = GRAPH_BUILDER_HUMAN_PROMPT, 
                              input_variables = ['input'], 
                              partial_variables = {
                                  'examples': EXAMPLES,
                                  'format_instructions': parser.get_format_instructions(),
                                }
)

human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

DEFAULT_PROMPT = ChatPromptTemplate.from_messages([system_message, human_message_prompt])

