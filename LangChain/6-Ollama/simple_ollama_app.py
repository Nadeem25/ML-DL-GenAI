import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
os.environ['LANGSMITH_PROJECT'] = os.getenv("LANGSMITH_PROJECT")
os.environ['LANGSMITH_TRACING'] = os.getenv("LANGSMITH_TRACING")



# Step 1: Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked"),
        ("user", "Question:{question}")
    ]
)

# Step 2: Streamlit framwork
st.title("LangChain Demo with Gemma Model")
question = st.text_input("What question you have in mind?")

# Step 3: CAll Ollama2 Model
llm = Ollama(model="gemma:2b") #- Load the given model
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if question:
    st.write(chain.invoke({"question":question}))


