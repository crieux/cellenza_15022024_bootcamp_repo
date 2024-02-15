#### 5 - VECTOR STORE ####
# https://python.langchain.com/docs/modules/data_connection/vectorstores/
# https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.RecursiveCharacterTextSplitter.html
# https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html

from langchain_community.document_loaders import TextLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate

raw_document = TextLoader("II.LangChain/5.Vector store/doc.txt", encoding = "utf-8").load()[0].page_content
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000, # token per chunk
    chunk_overlap = 100, # overlap in tokens between chunk    
    length_function = len,        
)
# 1 token ~= 1 word 

chunked_document = text_splitter.create_documents([raw_document])
db = FAISS.from_documents(
    chunked_document, 
    AzureOpenAIEmbeddings(
        openai_api_type = "azure",
        openai_api_version = "2023-10-01-preview",    
        deployment = "TextEmbeddingAda-002", # 120k / min
        azure_endpoint = "https://fr1azupocdatascienceaoai.openai.azure.com/",
        openai_api_key = "e719f55b7dd545a1a7e4c56cd6d4af87"
    )
)

prompt = PromptTemplate(
    input_variables = ["question"],
    template = """
    Your name is Alex and your aim is to reply to the human based on retrieved documents
    You should always reply in french

    Context: {context}
    Human: {question}
    Chatbot:
    """
)
chain = load_qa_chain(
    llm = AzureChatOpenAI(
        openai_api_type = "azure",
        openai_api_version = "2023-10-01-preview",    
        deployment_name = "GPT35Turbo16k-0613", # 120k / min
        azure_endpoint = "https://fr1azupocdatascienceaoai.openai.azure.com/",
        openai_api_key = "e719f55b7dd545a1a7e4c56cd6d4af87",
        temperature = 1
    ), 
    prompt = prompt
)

# Question 1
query = "What's the topic of the document ?"
docs = db.similarity_search(
    query,
    top = 5
)
docs
len(docs)

print(chain.invoke({"input_documents": docs,"question": query})["output_text"])

# Question 2
query = "When was it written ?"
docs = db.similarity_search(
    query,
    top = 5
)
docs
len(docs)
print(chain.invoke({"input_documents": docs,"question": query})["output_text"])

# Question 3
query = "Who is cited in this document ?"
docs = db.similarity_search(
    query,
    top = 5
)
docs
len(docs)
print(chain.invoke({"input_documents": docs,"question": query})["output_text"])