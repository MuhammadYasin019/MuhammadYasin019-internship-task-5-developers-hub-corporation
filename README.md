#  Context-Aware Chatbot  

## 🎯 Objective
The goal of this project is to enable users to **chat with multiple PDF documents** by leveraging a **Retrieval-Augmented Generation (RAG) pipeline** combined with conversational memory. This allows accurate, context-aware answers based exclusively on uploaded files.

---

## ⚙️ Methodology / Approach
1. **PDF Processing**:
   - Extracts text from uploaded PDFs using `PyPDF2`
   - Splits into manageable chunks with overlap for context retention

2. **Embeddings & Vector Store**:
   - Uses Google Generative AI embeddings (`models/embedding-001`)
   - Stores vectors in FAISS for efficient retrieval

3. **Conversational Retrieval**:
   - Uses Gemini (`gemini-1.5-flash`) as the LLM
   - Combines retrieved context with user queries
   - Maintains conversation history with `ConversationBufferMemory`

4. **Streamlit Interface**:
   - Upload multiple PDFs
   - Process and store content
   - Ask questions interactively
   - View conversation history

---

## 📊 Key Results / Observations
- Handles **multiple PDFs** simultaneously  
- Provides **structured, context-aware answers**  
- Retains **conversation history** for better multi-turn dialogue  
- Easily extendable for enterprise document search and QA  

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rag-pdf-chatbot.git
   cd rag-pdf-chatbot
