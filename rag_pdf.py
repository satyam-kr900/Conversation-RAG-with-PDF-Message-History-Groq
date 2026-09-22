import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


load_dotenv()

st.title("Conversation RAG with PDF + Message History Groq")


# Groq API Key
api_key = st.text_input(
    "Provide the GROQ API Key",
    type="password"
)


if api_key:

    # Groq Model
    model = ChatGroq(
        model_name="openai/gpt-oss-20b",
        groq_api_key=api_key
    )

    # Session ID
    session_id = st.text_input(
        "Please provide your session ID:",
        value="default-session"
    )

    # Store chat histories
    if "store" not in st.session_state:
        st.session_state.store = {}

    # Upload PDF
    uploaded_file = st.file_uploader(
        "Upload Your PDF:",
        type="pdf"
    )

    if uploaded_file:

        # Save uploaded PDF temporarily
        temp_pdf = "./temporary.pdf"

        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load PDF
        loader = PyPDFLoader(temp_pdf)
        documents = loader.load()
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        splits = splitter.split_documents(documents)
        # Ollama Embeddings
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )
        # Create Chroma Vector Store
        vector = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_data"
        )
        # Retriever
        retriever = vector.as_retriever()
        # History Aware Retriever
        contextualize_system_prompt = """
        Given a chat history and the latest user question,
        formulate a standalone question which can be understood
        without the chat history.
        Do NOT answer the question.
        Only reformulate the question if necessary.
        """

        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            model,
            retriever,
            contextualize_prompt
        )
        # Question Answering Chain
        

        system_prompt = """
        You are a helpful AI assistant.
        Answer the user's question using ONLY the provided context.
        If the answer is not available in the context,
        say "I don't know based on the provided context."
        Context:
        {context}
        """
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        document_chain = create_stuff_documents_chain(
            model,
            qa_prompt
        )
        # Retrieval Chain
        retriever_chain = create_retrieval_chain(
            history_aware_retriever,
            document_chain
        )
      
        # Chat History
        

        def get_session_history(session_id):

            if session_id not in st.session_state.store:

                st.session_state.store[session_id] = (
                    InMemoryChatMessageHistory()
                )

            return st.session_state.store[session_id]


        # Runnable with Message History

        conversation_rag_chain = RunnableWithMessageHistory(
            retriever_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        # User Question
        

        user_input = st.text_input(
            "Ask a Question about your PDF"
        )

        if user_input:

            session_history = get_session_history(session_id)

            response = conversation_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {
                        "session_id": session_id
                    }
                }
            )

            st.subheader("Assistant Answer")

            st.write(response["answer"])

            with st.expander("Chat History"):

                for message in session_history.messages:

                    if hasattr(message, "content"):
                        st.write(
                            f"**{message.type}:** {message.content}"
                        )

else:
    st.warning(
        "Please enter your GROQ API key to continue."
    )