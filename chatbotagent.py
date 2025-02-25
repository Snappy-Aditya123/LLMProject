from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
import streamlit as st
import DBMS  # Your database module
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from summarization import TextSummarizer
class ChatbotAgent:
    def __init__(self, model="llama3", keep_alive=False, history_limit=10):
        self.llm = ChatOllama(model=model, temperature=0.7, keep_alive=keep_alive, num_predict=300, num_thread=6)
        self.vector_db = Chroma(persist_directory="chroma_db", embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
        self.db = DBMS.ChatDatabase()
        self.history_limit = history_limit
        self.summrizer = TextSummarizer()
        
    def trim_chat_history(self, message_history):
        if len(message_history) > self.history_limit:
            # Summarize older messages
            summary = self.summrizer.summarize_conversation(message_history[:-self.history_limit])
            summarized_message = SystemMessage(f"Summary of previous conversation: {summary}")
            return [summarized_message] + message_history[-self.history_limit:]
        return message_history

    def chat(self, message_history,summaryofcv):
        if not message_history:
            return
        if not summaryofcv:
            cv_summary = "not given yet"
        else:
            cv_summary = self.summrizer.summarize_text(summaryofcv)
        # Trim chat history to manage memory efficiently
        message_history = self.trim_chat_history(message_history)
        # Extract latest user message
        last_message = message_history[-1]
        user_prompt = last_message.content
        # Retrieve relevant documents
        retriever = self.vector_db.as_retriever()
        context_docs = "not using for now"#retriever.get_relevant_documents(user_prompt)
        formatted_context =   "no context"#"\n\n".join(doc.page_content for doc in context_docs) if context_docs else ""
        
        # Create system message (not stored in message history)
        system_message = [SystemMessage(f"""
        Context from vector DB (for reference only, do not let it bias responses):
        {formatted_context}
        This is the cv of the user: {cv_summary} 
        use this cv to get more information of the user and answer questions based on it.
        """),]
        if cv_summary:
            print(cv_summary[:100])
        
        # Prepare input messages (system message + chat history)
        input_messages =  message_history[:-1] + system_message + [HumanMessage(user_prompt)] 
        print(input_messages)
        # Generate response using LLM   
        response = self.llm.stream(input_messages)
        # Save conversation in the database
        full_response = ""
        for chunk in response:
            full_response += chunk.content
            yield chunk.content
        # Collect response and save conversation
        self.db.insert_chat(user_prompt, full_response)
        
