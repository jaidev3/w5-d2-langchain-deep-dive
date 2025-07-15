import json

import schedule
import time
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
import cachetools.func
import os
import requests

from dotenv import load_dotenv

load_dotenv()



# Load policies
from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path='data/policies.json',
    jq_schema='.policies[] | .title + ": " + .content',
    text_content=False)

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Load FAQs similarly
faq_loader = JSONLoader(
    file_path='data/policies.json',
    jq_schema='.faqs[] | .question + ": " + .answer',
    text_content=False)
faq_docs = faq_loader.load()
faq_texts = text_splitter.split_documents(faq_docs)
vectorstore.add_documents(faq_texts)

# Load templates
templates = {}
with open('data/policies.json') as f:
    data = json.load(f)
    for tmpl in data['templates']:
        templates[tmpl['name']] = tmpl['template']

@cachetools.func.ttl_cache(maxsize=128, ttl=600)
def semantic_search(query):
    return vectorstore.similarity_search(query, k=3)

def extract_text_from_email(body):
    soup = BeautifulSoup(body, 'html.parser')
    return soup.get_text()

def generate_response(query, relevant_docs):
    # Simple prompt
    context = '\n'.join([doc.page_content for doc in relevant_docs])
    prompt = f"Based on the query: {query}\nContext: {context}\nGenerate an appropriate response using a suitable template."
    response = llm.invoke(prompt)
    return response.content

def process_emails():
    # Search for unread emails
    search_result = agent_executor.invoke({'input': "Search for unread messages"})
    messages = search_result['output']  # Parse accordingly
    # Assuming it returns list of message ids
    for msg_id in messages:
        msg = agent_executor.invoke({'input': f"Get message {msg_id}"})
        body = msg['output']  # Parse body
        text = extract_text_from_email(body)
        # Assume query is the text
        relevant_docs = semantic_search(text)
        response_text = generate_response(text, relevant_docs)
        # Create draft
        agent_executor.invoke({'input': f"Create a draft email replying to {msg_id} with body {response_text}"})
        # Mark as read if needed

# Gmail setup
credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
gmail_toolkit = GmailToolkit(api_resource=api_resource)

tools = gmail_toolkit.get_tools()

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Schedule
schedule.every(10).minutes.do(process_emails)

while True:
    schedule.run_pending()
    time.sleep(1) 