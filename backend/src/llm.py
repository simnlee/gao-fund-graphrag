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
from src.shared.constants import MODEL_VERSIONS

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

def get_llm(model: str):
    """Retrieve the specified language model based on the model name."""
    env_key = "LLM_MODEL_CONFIG_" + model
    env_value = os.environ.get(env_key)
    logging.info("Model: {}".format(env_key))
    if "gemini" in model:
        credentials, project_id = google.auth.default()
        model_name = MODEL_VERSIONS[model]
        llm = ChatVertexAI(
            model_name=model_name,
            convert_system_message_to_human=True,
            credentials=credentials,
            project=project_id,
            temperature=0,
            safety_settings={
                HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
    elif "openai" in model:
        model_name = MODEL_VERSIONS[model]
        llm = ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=model_name,
            temperature=0,
        )

    elif "azure" in model:
        model_name, api_endpoint, api_key, api_version = env_value.split(",")
        llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=api_endpoint,
            azure_deployment=model_name,  # takes precedence over model parameter
            api_version=api_version,
            temperature=0,
            max_tokens=None,
            timeout=None,
        )

    elif "anthropic" in model:
        model_name, api_key = env_value.split(",")
        llm = ChatAnthropic(
            api_key=api_key, model=model_name, temperature=0, timeout=None
        )

    elif "fireworks" in model:
        model_name, api_key = env_value.split(",")
        llm = ChatFireworks(api_key=api_key, model=model_name)

    elif "groq" in model:
        model_name, base_url, api_key = env_value.split(",")
        llm = ChatGroq(api_key=api_key, model_name=model_name, temperature=0)

    elif "bedrock" in model:
        model_name, aws_access_key, aws_secret_key, region_name = env_value.split(",")
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )

        llm = ChatBedrock(
            client=bedrock_client, model_id=model_name, model_kwargs=dict(temperature=0)
        )

    elif "ollama" in model:
        model_name, base_url = env_value.split(",")
        llm = ChatOllama(base_url=base_url, model=model_name)

    elif "diffbot" in model:
        model_name = "diffbot"
        llm = DiffbotGraphTransformer(
            diffbot_api_key=os.environ.get("DIFFBOT_API_KEY"),
            extract_types=["entities", "facts"],
        )
    
    else: 
        model_name, api_endpoint, api_key = env_value.split(",")
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=api_endpoint,
            model=model_name,
            temperature=0,
        )
            
    logging.info(f"Model created - Model Version: {model}")
    return llm, model_name


def get_combined_chunks(chunkId_chunkDoc_list):
    chunks_to_combine = int(os.environ.get("NUMBER_OF_CHUNKS_TO_COMBINE"))
    logging.info(f"Combining {chunks_to_combine} chunks before sending request to LLM")
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]
    combined_chunks_ids = [
        [
            document["chunk_id"]
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        ]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(
            Document(
                page_content=combined_chunks_page_content[i],
                metadata={"combined_chunk_ids": combined_chunks_ids[i]},
            )
        )
    return combined_chunk_document_list


def get_graph_document_list(
    llm, combined_chunk_document_list, allowedNodes, allowedRelationship
):
    futures = []
    graph_document_list = []
    if llm.get_name() == "ChatOllama":
        node_properties = False
    else:
        node_properties = ["description"]
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        node_properties=node_properties,
        allowed_nodes=allowedNodes,
        allowed_relationships=allowedRelationship,
        prompt=DEFAULT_PROMPT
    )
    with ThreadPoolExecutor(max_workers=10) as executor:
        for chunk in combined_chunk_document_list:
            chunk_doc = Document(
                page_content=chunk.page_content.encode("utf-8"), metadata=chunk.metadata
            )
            futures.append(
                executor.submit(llm_transformer.convert_to_graph_documents, [chunk_doc])
            )

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            graph_document = future.result()
            graph_document_list.append(graph_document[0])

    return graph_document_list


def get_graph_from_llm(model, chunkId_chunkDoc_list, allowedNodes, allowedRelationship):
    
    llm, model_name = get_llm(model)
    combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list)
    
    if  allowedNodes is None or allowedNodes=="":
        allowedNodes =[]
    else:
        allowedNodes = allowedNodes.split(',')    
    if  allowedRelationship is None or allowedRelationship=="":   
        allowedRelationship=[]
    else:
        allowedRelationship = allowedRelationship.split(',')
        
    graph_document_list = get_graph_document_list(
        llm, combined_chunk_document_list, allowedNodes, allowedRelationship
    )
    return graph_document_list
