import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Updated import
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import asyncio
import logging

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or ""

def ensure_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as e:
        if "There is no current event loop" in str(e):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    return loop

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    ensure_event_loop()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(vector_store):
    ensure_event_loop()
    prompt_template = """
    # ROLE & TASK
    You are an expert document analysis assistant. Your task is to provide accurate, comprehensive, and well-structured answers based EXCLUSIVELY on the provided context from uploaded PDF documents.

    # CONTEXT ANALYSIS
    Carefully analyze the provided context to extract relevant information. Consider the following:
    - Key facts, figures, and data points
    - Main concepts and ideas
    - Relationships between different pieces of information
    - Technical terminology and definitions

    # RESPONSE GUIDELINES
    1. **Accuracy First**: Base your answer strictly on the provided context. Do not speculate or use external knowledge.
    2. **Completeness**: Provide a thorough answer that covers all relevant aspects from the context.
    3. **Clarity**: Use clear, professional language. Structure your response for easy reading.
    4. **Citation Awareness**: If specific sections or pages are referenced in the context, mention this.
    5. **Uncertainty Handling**: If the context is incomplete or ambiguous, acknowledge this.

    # FORMATTING REQUIREMENTS
    - Use bullet points for lists
    - Use bold for key terms or important concepts
    - Use sections with headings for complex answers
    - Keep paragraphs concise and focused

    # WHEN INFORMATION IS INCOMPLETE
    If the context doesn't contain sufficient information to answer the question fully:
    - State what information IS available from the context
    - Clearly indicate what information is missing
    - Suggest what additional context might be needed for a complete answer

    # CONTEXT PROVIDED: {context}

    # QUESTION TO ANSWER: {question}

    # YOUR RESPONSE:
"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    
    return conversation_chain

def main():
    ensure_event_loop()
    st.set_page_config(page_title="Chat With Multiple PDFs", layout="centered")
    st.header("Chat with Multiple PDFs using Gemini")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and click Submit & Process", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        ensure_event_loop()
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.warning("No text could be extracted from the PDFs. Please try different files.")
                            return
                            
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.session_state.processed = True
                        st.success("Processing complete! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {str(e)}")
            else:
                st.warning("Please upload at least one PDF file before processing.")

        if st.button("Clear Conversation History"):
            if "memory" in st.session_state:
                st.session_state.memory.clear()
                st.success("Conversation history cleared!")
            else:
                st.warning("No conversation history to clear")

    if st.session_state.get("processed", False):
        user_question = st.text_input("Ask a question from the uploaded PDFs:")
        if user_question:
            try:
                ensure_event_loop()
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

                if "chat_chain" not in st.session_state:
                    st.session_state.chat_chain = get_conversational_chain(new_db)

                response = st.session_state.chat_chain({"question": user_question})
                st.write("Reply:", response["answer"])
                
                if "memory" in st.session_state and st.session_state.memory.chat_memory.messages:
                    st.subheader("Conversation History")
                    for msg in st.session_state.memory.chat_memory.messages:
                        sender = "You" if msg.type == "human" else "AI"
                        st.markdown(f"**{sender}:** {msg.content}")
                
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")
    else:
        st.info("Please upload and process PDF files first to enable question answering.")

if __name__ == "__main__":
    main()