import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
import re

# Load environment variables
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in the .env file or system environment.")
    try:
        llm = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0.5,
            max_tokens=512,
            REMOVED_SECRETapi_key=api_key
        )
        return llm
    except Exception as e:
        raise ValueError(f"Error initializing ChatGroq: {str(e)}")

def format_references(source_documents):
    """Format source documents into a readable reference list, including figure mentions if available."""
    references = []
    for i, doc in enumerate(source_documents, 1):
        metadata = doc.metadata
        page_content = doc.page_content.lower()
        # Check for figure mentions in page_content (e.g., "Figure 1.1", "Fig. 1-2")
        figure_mention = "No figure mentioned"
        figure_matches = re.findall(r'(?:figure|fig\.)\s*[\d\.\-]+\b', page_content, re.IGNORECASE)
        if figure_matches:
            figure_mention = ", ".join(figure_matches)
        ref = (
            f"{i}. **{os.path.basename(metadata.get('source', 'Unknown Source'))}** "
            f"(Book Page {metadata.get('page_label', 'Unknown')}, "
            f"PDF Page {metadata.get('page', 'Unknown')}, "
            f"Figure: {figure_mention})"
        )
        references.append(ref)
    return "\n".join(references)

def main():
    st.title("Medical Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if message['role'] == 'assistant' and 'references' in message:
                with st.expander("View References", expanded=False):
                    st.markdown("**References**:\n" + message['references'])

    prompt = st.chat_input("Enter your medical query here (e.g., What is Biosynthetic Machinery?)")

    if prompt:
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        You are a medical expert tasked with providing a detailed and accurate explanation based solely on the provided context from 'Robbins Basic Pathology 10th Edition'. 
        Craft a response that mirrors the style and structure of the textbook, including:
        - Relevant topics, subtopics, and detailed explanations as presented in the context.
        - Specific biological processes, mechanisms, or interactions described in the text.
        - Any relevant terminology, examples, or relationships to other cellular processes.
        - A note on any limitations if the context lacks sufficient detail for a complete answer.
        Use the exact phrasing and organization (e.g., headings, bullet points) from the context where possible to reflect the book's narrative style. 
        Do not include information beyond the provided context or speculate on details not present.

        Context: {context}
        Question: {question}
        """
        
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            with st.chat_message('assistant'):
                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_documents = response["source_documents"]
                references = format_references(source_documents)

                st.markdown(result)
                with st.expander("View References", expanded=False):
                    st.markdown("**References**:\n" + references)

                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': result,
                    'references': references
                })

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()