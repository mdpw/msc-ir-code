# =======================================
# 🏥 Fast Hospital Assistant Chatbot
# =======================================
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

 
# ---- Load FAISS Knowledge Base ----
embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
vectordb = FAISS.load_local("hospital_faiss_index", embedding, allow_dangerous_deserialization=True)
 
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})
 
# ---- Streamlit UI ----
st.set_page_config(page_title="🏥 Fast Hospital Assistant Chatbot", layout="wide")
st.title("🏥 Virtual Hospital Assistant (FAISS + Ollama)")
 
# ---- Smalltalk filter ----
def is_smalltalk(msg: str) -> bool:
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    return msg.lower().strip() in greetings
 
# ---- Model selector ----
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["phi3:mini", "gemma:2b", "mistral"]
)
 
# ---- Define Ollama LLM ----
llm = OllamaLLM(
    model=model_choice,
    streaming=True,
    options={"num_ctx": 1024, "num_predict": 120, "temperature": 0.3}
)
 
# ---- Strict grounded prompt ----
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a hospital assistant AI.
 
Answer ONLY using the Hospital Context.
Keep replies short, clear, and factual (max 3 sentences).  
If the context does not contain an answer, reply:  
"I'm sorry, I don't have that information in the hospital database."
 
Hospital Context:
{context}
 
Patient Question:
{question}
 
Response:
"""
)
 
# Create RAG chain using LCEL
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
 
# ---- Memory for chat history ----
if "history" not in st.session_state:
    st.session_state["history"] = []
 
# ---- Render history ----
for role, msg in st.session_state["history"]:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(msg)
 
# ---- Chat input ----
user_message = st.chat_input("Ask me about hospital services, departments, or guidelines...")
 
if user_message:
    # 1. Display user msg
    st.session_state["history"].append(("user", user_message))
    with st.chat_message("user"):
        st.markdown(user_message)
 
    # 2. Handle smalltalk separately
    if is_smalltalk(user_message) or len(user_message.split()) < 2:
        bot_reply = "👋 Hello! How can I assist you with hospital information today?"
        # Save assistant reply
        st.session_state["history"].append(("assistant", bot_reply))
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
    
    else:
        # 3. Run RAG pipeline
        bot_reply = rag_chain.invoke(user_message)
        sources = retriever.invoke(user_message)  # Get sources separately

        # Show response with sources in chat message
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
            with st.expander("🔍 Context from FAISS (debug)"):
                if sources:
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**Doc {i}:** {doc.page_content}")
                else:
                    st.warning("⚠️ No hospital documents retrieved.")
        
        # 4. Save assistant reply
        st.session_state["history"].append(("assistant", bot_reply))
