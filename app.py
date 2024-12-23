import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
if 'all_chats' not in st.session_state:
    st.session_state.all_chats = {}
if 'scraped_urls' not in st.session_state:
    st.session_state.scraped_urls = set()

def save_chat_history():
    """Save chat history to a JSON file"""
    if not os.path.exists('chat_histories'):
        os.makedirs('chat_histories')
    
    filename = f'chat_histories/chat_history_{st.session_state.current_chat_id}.json'
    chat_data = {
        'chat_id': st.session_state.current_chat_id,
        'url': st.session_state.all_chats[st.session_state.current_chat_id]['url'],
        'timestamp': st.session_state.all_chats[st.session_state.current_chat_id]['timestamp'],
        'messages': st.session_state.chat_history
    }
    
    with open(filename, 'w') as f:
        json.dump(chat_data, f)

def load_chat_history(chat_id):
    """Load chat history from a JSON file"""
    filename = f'chat_histories/chat_history_{chat_id}.json'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def create_new_chat():
    """Create a new chat session"""
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat_id = chat_id
    st.session_state.chat_history = []
    st.session_state.vector_store = None
    st.session_state.conversation_chain = None
    st.session_state.all_chats[chat_id] = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'url': None
    }
    return chat_id

def switch_chat(chat_id):
    """Switch to a different chat session"""
    if st.session_state.current_chat_id:
        save_chat_history()
    
    st.session_state.current_chat_id = chat_id
    chat_data = load_chat_history(chat_id)
    if chat_data:
        st.session_state.chat_history = chat_data['messages']
        if chat_data['url'] in st.session_state.scraped_urls:
            data = scrape_data(chat_data['url'])
            if 'error' not in data:
                chunks = chunk_data(data)
                st.session_state.vector_store = create_embeddings_chroma(chunks)
                st.session_state.conversation_chain = initialize_conversation_chain(st.session_state.vector_store)

def scrape_data(url):
    """Scrape data from a given URL"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        structured_data = {
            "url": url,
            "headings": [],
            "paragraphs": [],
            "other_text": []
        }
        
        # Extract headings
        for heading in range(1, 7):
            for tag in soup.find_all(f"h{heading}"):
                text = tag.get_text(strip=True)
                if text:
                    structured_data["headings"].append(text)
        
        # Extract paragraphs
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if text:
                structured_data["paragraphs"].append(text)
        
        # Remove duplicates while maintaining order
        structured_data["headings"] = list(dict.fromkeys(structured_data["headings"]))
        structured_data["paragraphs"] = list(dict.fromkeys(structured_data["paragraphs"]))
        
        return structured_data
    
    except Exception as e:
        return {"error": str(e)}

def chunk_data(data, chunk_size=256):
    """Split data into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    
    combined_text = []
    if 'headings' in data:
        combined_text.extend(data['headings'])
    if 'paragraphs' in data:
        combined_text.extend(data['paragraphs'])
    if 'other_text' in data:
        combined_text.extend(data['other_text'])
    
    full_text = "\n\n".join(combined_text)
    chunks = text_splitter.split_text(full_text)
    
    return [Document(page_content=chunk) for chunk in chunks]

def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):
    """Create embeddings using Chroma"""
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    return vector_store

def initialize_conversation_chain(vector_store):
    """Initialize the conversation chain"""
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={'k': 5}),
        memory=memory,
        chain_type='stuff',
        verbose=False
    )
    
    return conversation_chain

def main():
    st.set_page_config(page_title="Web Chat Bot", layout="wide")
    
    # Sidebar for chat management
    with st.sidebar:
        st.title("Chat Management")
        
        # New Chat button
        if st.button("New Chat", key="new_chat"):
            create_new_chat()
            st.rerun()
        
        # Clear current chat
        if st.button("Clear Current Chat", key="clear_chat"):
            if st.session_state.current_chat_id:
                st.session_state.chat_history = []
                save_chat_history()
                st.rerun()
        
        # Display chat history
        st.subheader("Chat History")
        if st.session_state.all_chats:
            for chat_id, chat_info in sorted(st.session_state.all_chats.items(), 
                                          key=lambda x: x[1]['timestamp'], 
                                          reverse=True):
                chat_title = f"{chat_info['timestamp']}"
                if chat_info['url']:
                    chat_title += f"\n{chat_info['url'][:30]}..."
                
                if st.button(chat_title, key=f"chat_{chat_id}"):
                    switch_chat(chat_id)
                    st.rerun()

    # Main content
    st.title("Web Scraping Chat Application")
    
    # Create new chat if none exists
    if not st.session_state.current_chat_id:
        create_new_chat()
    
    # URL input
    col1, col2 = st.columns([3, 1])
    with col1:
        url = st.text_input("Enter URL to scrape:")
        scrape_button = st.button("Scrape and Process")
    
    if scrape_button:
        with st.spinner("Scraping and processing data..."):
            data = scrape_data(url)
            
            if "error" in data:
                st.error(f"Error scraping data: {data['error']}")
            else:
                chunks = chunk_data(data)
                vector_store = create_embeddings_chroma(chunks)
                conversation_chain = initialize_conversation_chain(vector_store)
                
                st.session_state.vector_store = vector_store
                st.session_state.conversation_chain = conversation_chain
                st.session_state.scraped_urls.add(url)
                st.session_state.all_chats[st.session_state.current_chat_id]['url'] = url
                save_chat_history()
                st.success("Data processed successfully! You can now start chatting.")
    
    # Chat interface
    st.header("Chat Interface")
    
    # Display current URL being processed
    if st.session_state.all_chats[st.session_state.current_chat_id]['url']:
        st.info(f"Currently chatting about: {st.session_state.all_chats[st.session_state.current_chat_id]['url']}")
    
    if st.session_state.vector_store is None:
        st.info("Please scrape a website first to start chatting.")
    else:
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message("user" if message[0] == "user" else "assistant"):
                    st.write(message[1])
        
        user_question = st.chat_input("Ask a question about the scraped content:")
        
        if user_question:
            st.session_state.chat_history.append(("user", user_question))
            
            with st.spinner("Thinking..."):
                response = st.session_state.conversation_chain.invoke(
                    {"question": user_question}
                )
            
            st.session_state.chat_history.append(("assistant", response['answer']))
            save_chat_history()
            st.rerun()

if __name__ == "__main__":
    main()