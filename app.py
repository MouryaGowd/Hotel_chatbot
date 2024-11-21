import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Load documents from the CSV
loader = CSVLoader(file_path="knowledge.csv")
documents = loader.load()

# Initialize OpenAI embeddings with the API key from environment variable
embeddings = OpenAIEmbeddings(openai_api_key="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")


# Initialize FAISS vectorstore
db = FAISS.from_documents(documents, embeddings)

# Function to retrieve similar responses
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# Define the prompt template
template = """
You are Mourya Residency's representative, a knowledgeable and friendly virtual assistant dedicated to helping guests. You have detailed knowledge about Mourya Residency, including its location, services, amenities, policies, nearby attractions, and contact information.

Your goal is to assist guests with accurate and concise answers to their queries, 
including room availability, pricing, booking procedures, transportation guidance, and any other questions related to Mourya Residency. Always respond professionally and politely, ensuring guests
feel valued and well-informed.

You will follow all rules below:

1. Response should be very similar or even identical to past best practices in terms of length, tone of voice, explanation, and hospitality.
2. If the best practices are irrelevant, then try to mimic the style of best practices.

Below is a message I received from the user:
{message}

Here is a list of best practices of how we respond to users in similar scenarios:
{best_practice}

Please write the best response that you should send to the user.
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"], template=template
)

# Initialize the LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
chain = LLMChain(llm=llm, prompt=prompt)

# Function to generate a response
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

# Test the chatbot
message = "What are the lunch timings?"
response = generate_response(message)
print(response)

def main():
    st.set_page_config(
        page_title="Customer response generator",page_icon=":bird:")
    st.header("Customer response generator :bird:")
    message = st.text_area("customer message")
    if message:
        st.write("Generating best practice message...")
        result = generate_response (message)
        st.info (result)
        

if __name__=='__main__':
 main()
