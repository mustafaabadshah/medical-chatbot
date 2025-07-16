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
            groq_api_key=api_key
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
    st.markdown("Enter a query below to get detailed answers from *Robbins Basic Pathology 10th Edition*. Click 'View References' to see sources.")

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

        # Check if query is about MSCs or similar terms for concise response
        is_concise_query = any(term in prompt.lower() for term in ["mscs", "mesenchymal stem cells"])

        if is_concise_query:
            CUSTOM_PROMPT_TEMPLATE = """
            You are a medical expert tasked with providing a concise and accurate answer based solely on the provided context from 'Robbins Basic Pathology 10th Edition'. 
            Extract and reproduce the relevant information exactly as presented in the textbook, focusing on the specific term or topic asked (e.g., MSCs). 
            Include key details, terminology, and relationships to other processes as in the context, but keep the response brief and direct. 
            If the context lacks sufficient detail, state the limitation clearly.

            Context: {context}
            Question: {question}
            """
        else:
            CUSTOM_PROMPT_TEMPLATE = """
            You are a medical expert tasked with providing a detailed and accurate explanation based solely on the provided context from 'Robbins Basic Pathology 10th Edition'. 
            Extract and reproduce the content exactly as presented in the textbook, including:
            - Relevant topics, subtopics, and detailed explanations as written in the context.
            - Specific biological processes, mechanisms, or interactions described in the text.
            - Any relevant terminology, examples, or relationships to other cellular processes.
            - A note on any limitations if the context lacks sufficient detail for a complete answer.
            Use the exact phrasing, organization (e.g., headings, bullet points), and style from the context to mirror the book's narrative. 
            Do not rewrite, summarize, or add information beyond the provided context.

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
