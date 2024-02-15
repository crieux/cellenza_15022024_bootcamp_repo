# Use case with Google Custom Search
# https://console.cloud.google.com/apis/library

# Step 1 : Imports and utils
from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
import requests, urllib.request, http.client, urllib.error
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

GOOGLE_API_KEY = "AIzaSyDhWb11HX6Z-VLk3iakL1xVVnYzja1dZ04"
GOOGLE_SEARCH_ENGINE_ID = "1377da92307a64915"
LLM = AzureChatOpenAI(
    openai_api_type = "azure",
    openai_api_version = "2023-10-01-preview",    
    deployment_name = "GPT35Turbo16k-0613",
    azure_endpoint = "https://fr1azupocdatascienceaoai.openai.azure.com/",
    openai_api_key = "e719f55b7dd545a1a7e4c56cd6d4af87",
    temperature = 1
)
EMBEDDINGS = AzureOpenAIEmbeddings(
    openai_api_type = "azure",
    openai_api_version = "2023-10-01-preview",    
    deployment = "TextEmbeddingAda-002", # 120k / min
    azure_endpoint = "https://fr1azupocdatascienceaoai.openai.azure.com/",
    openai_api_key = "e719f55b7dd545a1a7e4c56cd6d4af87"
)

def get_web_page_content(search_items):
    url_links = []
    page_contents = []
    if search_items:
        for search_item in search_items:
            url_link = search_item.get("link")
            url_links.append(url_link)
            try:
                page_content = urllib.request.urlopen(url_link).read()
            except http.client.IncompleteRead:
                page_content = ""
                pass
            except urllib.error.HTTPError:
                page_content = ""
                pass
            soup = BeautifulSoup(page_content, features = "lxml").get_text().replace("\n", "").replace("\t", "").replace("\r", "")
            page_contents.append(soup)
    return url_links, page_contents

def retrieval_augmented_generation(human_query):
    # Step 2 : Extract keywords from human query
    keyword_prompt = PromptTemplate(
        input_variables = ["question"],
        template = """
        Summarize the sentence with at most 4 french expressions of 1 word and format 
        your answer with only the expressions separate with blank space
        Please use masculine words and infinitive for verbs, and avoid slang words

        - question : {question}
        """
    )
    chain = LLMChain(
        llm = LLM,
        prompt = keyword_prompt
    )
    keywords = chain.invoke({"question": human_query})["text"]
    #print(keywords)

    # Step 3 : Google custoum search with Wikipedia and read pages content
    query = keywords

    google_search = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_SEARCH_ENGINE_ID}&q={query}&start={1 * 10 + 1}"
    google_search_data = requests.get(google_search).json()
    google_search_items = google_search_data.get("items")
    url_links, page_contents = get_web_page_content(google_search_items[:2])
    #print(pages_content)

    # Step 4 : Create vector store from the results
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000, # token per chunk
        chunk_overlap = 100, # overlap in tokens between chunk    
        length_function = len,        
    ) 
    chunked_document = text_splitter.create_documents(" ".join(page_contents))
    db = FAISS.from_documents(
        chunked_document, 
        EMBEDDINGS,
    )
    docs = db.similarity_search(
        human_query,
        top = 5
    )

    # Step 5 : Generate answer from google research

    prompt = PromptTemplate(
        input_variables = ["question"],
        template = """
        Synthetize the documents given through the context and format a nearly 300 words response
        If no document are provided, just reply that you could'nt find any informations
        You should always reply in French.

        - context : {context}   
        - question : {question}
        """
    )
    chain = load_qa_chain(
        llm = LLM, 
        prompt = prompt
    )
    ai_answer = chain.invoke({"input_documents": docs, "question": human_query})["output_text"]
    results = ai_answer + "\nSources : \n    -" + "\n    - ".join([url for url in url_links])
    return results

#print(retrieval_augmented_generation("Qui était Ada Lovelace ?"))
#print(retrieval_augmented_generation("Quelle est la capitale du Québec et quelle est son histoire ?"))
#print(retrieval_augmented_generation("Qu'est ce que l'analog horror ?"))

############################################################
# Streamlit page for chatbot interaction
# https://streamlit.io/
# python -m streamlit run '.\III.Use case\app.py'
############################################################

import streamlit as st

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning) 

st.title("🦜🔗 RAG use case with Wikipedia")
st.header("Interact with Wikipedia through ChatGPT")

# Initial message
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """Bonjour, je suis votre assistant de recherche sur Wikipedia, que recherchons
            nous aujourd'hui ?
            """,
        }
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

@st.cache_data(max_entries = 50, persist = False, show_spinner = False)
def generate_response(query_text):
    response = retrieval_augmented_generation(query_text)
    return str(response)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            placeholder = st.empty()
            full_response = ""
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)