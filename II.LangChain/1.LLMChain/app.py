#### 1 - LLMCHAIN ####
# https://python.langchain.com/docs/modules/chains
# https://api.python.langchain.com/en/latest/chains/langchain.chains.llm.LLMChain.html#

from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate

################################################################################################
# Test 1 : 
################################################################################################

prompt = PromptTemplate(
    input_variables = ["question"],
    template = """
    Your name is Alex and your aim is to satisfy the demand of the human
    You should always reply in french

    Human: {question}
    Chatbot:
    """
)
chain = LLMChain(
    llm = AzureChatOpenAI(
        openai_api_type = "azure",
        openai_api_version = "2023-10-01-preview",    
        deployment_name = "GPT35Turbo16k-0613",
        azure_endpoint = "https://fr1azupocdatascienceaoai.openai.azure.com/",
        openai_api_key = "e719f55b7dd545a1a7e4c56cd6d4af87",
        temperature = 1
    ), 
    prompt = prompt
)
print(chain.invoke({"question": "raconte moi une blague s'il te plait"})["text"])
print(chain.invoke({"question": "combien fait 2000 - 24 ?"})["text"])
print(chain.invoke({"question": "comment tu t'appelles ?"})["text"])
print(chain.invoke({"question": "qui suis-je ?"})["text"])

################################################################################################
# Test 2 : 
################################################################################################

prompt = PromptTemplate(
    input_variables = ["question"],
    template = """
    Your name is Alex and your aim is to satisfy the demand of the human
    The human's name is Charlotte
    You should always reply in french

    Human: {question}
    Chatbot:
    """
)
chain = LLMChain(
    llm = AzureChatOpenAI(
        openai_api_type = "azure",
        openai_api_version = "2023-10-01-preview",    
        deployment_name = "GPT35Turbo16k-0613",
        azure_endpoint = "https://fr1azupocdatascienceaoai.openai.azure.com/",
        openai_api_key = "e719f55b7dd545a1a7e4c56cd6d4af87",
        temperature = 1
    ), 
    prompt = prompt
)

print(chain.invoke({"question": "qui suis-je ?"})["text"])