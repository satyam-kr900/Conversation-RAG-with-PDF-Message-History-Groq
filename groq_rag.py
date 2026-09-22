import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langchain_classic.chains import create_retrieval_chain 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain  


load_dotenv()


model = ChatGroq(model='openai/gpt-oss-20b', temperature=0)

prompt = ChatPromptTemplate.from_template(
    "Answer from the following context only. Please provide the most accurate results "
    "based on the context only.\n\nContext:\n{context}\n\nQuestion: {input}"
)

def generate_embedding():
    if "vectors" not in st.session_state:
        
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader('data')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.splitted = st.session_state.splitter.split_documents(st.session_state.docs)
        
        
        st.session_state.vectors = FAISS.from_documents(st.session_state.splitted, embedding=st.session_state.embeddings)
        st.write("Vector database is ready!")

st.title("GROQ LPU RAG ChatBot")  
st.write("Please CLICK the BUTTON below to ADD embeddings of your Personal data")

if st.button("Generate Embedding"):
    generate_embedding()
  
user_prompt = st.text_input("ENTER YOUR QUERY FROM THE UPLOADED Document")

if st.button("Answer"):
    if user_prompt:
        if "vectors" not in st.session_state:
            st.error("Please click 'Generate Embedding' button first before searching!")
        else:
            document_context = create_stuff_documents_chain(model, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieved_data = create_retrieval_chain(retriever, document_context)
            
            response = retrieved_data.invoke({'input': user_prompt})
            
            st.write("**** ANSWER ****")
            st.write(response['answer'])
            
            with st.expander("Context from the Documents:"):
                if "context" in response:
                    for i, doc in enumerate(response['context']):
                        st.write(f"**Chunk {i+1}:**")
                        st.write(doc.page_content)
                        st.write('---------------------')
