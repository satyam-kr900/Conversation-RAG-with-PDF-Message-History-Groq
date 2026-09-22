import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from dotenv import load_dotenv
load_dotenv()


prompt=ChatPromptTemplate.from_messages([('system','You are an helpful assistant'),('human','Question:{Question}')])

def generate_r(query,llm,temperature):
  model=Ollama(model=llm,temperature=temperature)
  parser=StrOutputParser()
  chain= prompt| model|parser
  result=chain.invoke({'Question':query})
  return result

st.title("Ollama Q&A ChatBot ")  

query_user=st.text_input("Hey What's Your Query")

st.sidebar.title("Settings")

model_name=st.sidebar.selectbox('Select the LLM:',['llama3.2:1b','gemma3','gemma3:1b'])

temperature=st.sidebar.slider("Select Temperature:",min_value=0.0,max_value=2.0,value=1.0)

response=generate_r(query_user,model_name,temperature)

if st.button("Answer"):
  if query_user:
    st.write(response)
  else:
    st.warning("please Enter some query")