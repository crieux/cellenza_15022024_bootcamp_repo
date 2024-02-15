#### 3 - MEMORY ####
# https://python.langchain.com/docs/modules/memory/

from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema.messages import messages_from_dict

prompt = PromptTemplate(
    input_variables = ["question", "chat_history"],
    template = """
    Your name is Alex and your aim is to satisfy the demand of the human.
    You should always reply in french.
    If the user is asking for chat_history, then you can return it

    Human: {question}
    Chat_history: {chat_history}
    Chatbot:
    """
)
chat_history = messages_from_dict(
    [
        {"type": "human", "data": {"content": "Salut, je m'appelle Charlotte"}},
        {"type": "ai", "data": {"content": "Bonjour Charlotte !"}},
        {"type": "human", "data": {"content": "Tu connais la blague de la couleur du cheval blanc d'Henry IV ?"}},
        {"type": "ai", "data": {"content": "Non, racontes ?"}},
        {"type": "human", "data": {"content": "Bah c'est super facile il est blanc."}}
    ]
)
memory = ConversationBufferMemory(
    chat_memory = ChatMessageHistory(messages = chat_history),
    memory_key = "chat_history",
    input_key = "question",
    return_messages = True,
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
    prompt = prompt,
    memory = memory
)
print(chain.invoke({"question": "Quelle est la couleur du cheval d'Henry IV ?"})["text"])
print(chain.invoke({"question": "Quel est l'historique de tous nos messages ?"})["text"])

print(chain.invoke({"question": "blabla ?"})["text"])
print(chain.invoke({"question": "Quel est l'historique de tous nos messages ?"})["text"])