#### 4 - AGENTS ####
# https://python.langchain.com/docs/modules/agents/
# https://python.langchain.com/docs/modules/agents/tools/custom_tools

################################################################################################
# Introduction
################################################################################################

import math
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent

llm = AzureChatOpenAI(
    openai_api_type = "azure",
    openai_api_version = "2023-10-01-preview",    
    deployment_name = "GPT35Turbo16k-0613",
    azure_endpoint = "https://fr1azupocdatascienceaoai.openai.azure.com/",
    openai_api_key = "e719f55b7dd545a1a7e4c56cd6d4af87",
    temperature = 1
) 

prompt = PromptTemplate(
    input_variables = ["agent_scratchpad", "input"],
    template = """
    Your name is Alex and your aim is to satisfy the demand of the human
    You should always reply in french

    Agent: {agent_scratchpad}
    Human: {input}
    Chatbot:
    """
)
prompt = hub.pull("hwchase17/openai-functions-agent")

# https://www.delftstack.com/fr/howto/python/fibonacci-sequence-python/
def fibonacci_f(input):
    """
    returns the nth number of the fibonnaci suite
    """
    return round(((1 + math.sqrt(5)) ** int(input) - (1 - math.sqrt(5)) ** int(input)) / (2 ** int(input) * math.sqrt(5)), 0)

def pi_f(input):    
    """
    returns the nth decimal of the number pi
    """
    return str(math.pi).split(".")[1][int(input) + 1]

def celcius_to_farenheit_f(input):
    """
    convert a value of temperature in celcius to farenheit
    """
    return round(((float(input) * 9 / 5) + 32), 1)

def farenheit_to_celcius_f(input):
    """
    convert a value of temperature in farenheit to celcius
    """
    return round(((float(input) - 32) * 5 / 9), 1)

tools = [
    Tool.from_function(
        func = fibonacci_f,
        name = "fibonacci",
        description = "returns the nth number of the fibonnaci suite",
        return_direct = True,
    ),
    Tool.from_function(
        func = pi_f,
        name = "pi",
        description = "returns the nth decimal of the number pi",
        return_direct = True,
    ),
    Tool.from_function(
        func = celcius_to_farenheit_f,
        name = "celcius_to_farenheit",
        description = "convert a value of temperature in celcius to farenheit",
        return_direct = True,
    ),
    Tool.from_function(
        func = farenheit_to_celcius_f,
        name = "farenheit_to_celcius",
        description = "convert a value of temperature in farenheit to celcius",
        return_direct = True,
    ),
]

agent = create_openai_functions_agent(
    llm = llm, 
    tools = tools, 
    prompt = prompt
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent = agent,
    tools = tools,
    return_intermediate_steps = True,
    verbose = True
)

agent_executor.invoke({"input": "Hi !"})
agent_executor.invoke({"input": "Quelle est le 25eme nombre de la suite de finonacci ?"})
agent_executor.invoke({"input": "Quelle est la 13eme décimale de Pi ?"})
agent_executor.invoke({"input": "Combien font 36,6°C en °F ?"})
agent_executor.invoke({"input": "Quelle valeur de °F correspond a 36,6° C ?"})
agent_executor.invoke({"input": "Combien font 1 000 000°F en °C ?"})

################################################################################################
# Pour aller plus loin
# https://python.langchain.com/docs/integrations/toolkits/python?ref=blog.langchain.dev
# https://newsletter.theaiedge.io/p/introduction-to-langchain-augmenting
################################################################################################

from langchain_experimental.tools import PythonREPLTool

tools = [PythonREPLTool()]

instructions = """
You are an agent designed to write and execute python code to answer questions.
You have access to a python REPL, which you can use to execute python code.
If you get an error, debug your code and try again.
Only use the output of your code to answer the question. 
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
"""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

agent = create_openai_functions_agent(
    llm = llm,
    tools = tools,
    prompt = prompt
)
agent_executor = AgentExecutor(
    agent = agent, 
    tools = tools, 
    return_intermediate_steps = True,
    verbose = True
)
agent_executor.invoke({"input": "Quelle est le 25eme nombre de la suite de finonacci ?"})