#### 2 - PROMPT ####
# https://python.langchain.com/docs/modules/model_io/prompts/

from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate


llm = AzureChatOpenAI(
        openai_api_type = "azure",
        openai_api_version = "2023-10-01-preview",    
        deployment_name = "GPT35Turbo16k-0613",
        azure_endpoint = "https://fr1azupocdatascienceaoai.openai.azure.com/",
        openai_api_key = "e719f55b7dd545a1a7e4c56cd6d4af87",
        temperature = 1
    )

#################################################################################################
######################################### PROMPT 1 ##############################################
#################################################################################################

# Text generation
prompt_1 = PromptTemplate(
    input_variables = ["topic"],
    template = """
    Generate a 500 words bedtime story for a child about the human's topic.
    Please answer in french only

    Human: {topic}
    Chatbot: 
    """
)
chain = LLMChain(
    llm = llm,
    prompt = prompt_1
)
human_input_1 = """
dragon
"""
print(chain.invoke({"topic": human_input_1})["text"])

#################################################################################################
######################################### PROMPT 2 ##############################################
#################################################################################################

# Information extraction
prompt_2 = PromptTemplate(
    input_variables = ["human_input"],
    template = """
    Your name is Alex and your aim is to retrieve the following information from the human input
        - a name : for instance Albert Einstein (format the surname in capitals)
        - a birthdate : for instance the 1879, the 14th of march
        - a nationality : for instance German
        - a job : for instance Physicist

    Please format you answer in french and follow these standards:
        'Les personnes mentionnées dans l'input sont:
            - Personne 1:
                - Identité : Prénom NOM (surname in capitals)
                - Date de naissance : 01/01/1970 (dd/MM/YYYY)
                - Nationalité(s) : Nationalité_1, Nationalité_2, ... (capitalize first letter)
                - Métier : métier
            - Personne 2: ...
        '
    More than 1 individual can be cited within the human input.

    Human: {human_input}
    Chatbot: 
    """
)
chain = LLMChain(
    llm = llm,
    prompt = prompt_2
)
human_input_2 = """
Albert Einstein né le 14 mars 1879 à Ulm et mort le 18 avril 1955 à Princeton, 
est un physicien théoricien. Il fut successivement allemand, apatride, suisse et de 
double nationalité helvético-américaine.
Margaret Heafield Hamilton, née Margaret Heafield le 17 août 1936, est une informaticienne, 
ingénieure système et cheffe d'entreprise américaine.
Rosalind Franklin est une physicochimiste britannique, née le 25 juillet 1920 à Notting Hill 
et morte le 16 avril 1958 à Chelsea.
Charlotte Rieux est un dev python qui présente devant Cellenza Paris le 15/02/2024 
"""
print(chain.invoke({"human_input": human_input_2})["text"])
print(chain.invoke({"topic": "qui est Charlotte Rieux"})["text"])

#################################################################################################
######################################### PROMPT 3 ##############################################
#################################################################################################

# Prompt generation
prompt_3 = PromptTemplate(
    input_variables = ["human_question"],
    template = """
    Your name is Alex and your aim is to create a prompt from human input.
    Please answer in French.
    Reply only with the prompt

    Human: {human_question}
    """
)
chain = LLMChain(
    llm = llm,
    prompt = prompt_3
)
human_input_3 = """
I would like a prompt that returns all things that I cannot ask to gpt
"""
new_prompt = chain.invoke({"human_question": human_input_3})["text"]
print(new_prompt)

# |
# |
# |
# V

prompt_4 = PromptTemplate(
    input_variables = ["human_question"],
    template = """
    Your name is Alex and you answer to human question

    Human: {human_question}
    Chatbot: 
    """
)
chain = LLMChain(
    llm = llm,
    prompt = prompt_4
)
human_input_4 = f"""
{new_prompt}
"""
print(chain.invoke({"human_question": human_input_4})["text"])
