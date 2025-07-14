import os
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
import re

# Load environment variables
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db, None
    except Exception as e:
        return None, f"Error loading vector store: {str(e)}"

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
        return llm, None
    except Exception as e:
        return None, f"Error initializing ChatGroq: {str(e)}"

def format_references(source_documents):
    """Format source documents into a readable reference list, including figure mentions if available."""
    references = []
    for i, doc in enumerate(source_documents, 1):
        metadata = doc.metadata
        page_content = doc.page_content.lower()
        figure_mention = "No figure mentioned"
        if "figure" in page_content or "fig." in page_content:
            figure_matches = re.findall(r'(?:figure|fig\.)\s*[\d\.\-]+', page_content, re.IGNORECASE)
            if figure_matches:
                figure_mention = ", ".join(figure_matches)
        ref = (
            f"{i}. **{metadata.get('source', 'Unknown Source')}** "
            f"(Page {metadata.get('page_label', 'Unknown')}, "
            f"PDF Page {metadata.get('page', 'Unknown')}, "
            f"Figure: {figure_mention})"
        )
        references.append(ref)
    return "\n".join(references)

def chatbot_response(message, history, show_references):
    """Handle user query and return response with optional references."""
    CUSTOM_PROMPT_TEMPLATE = """
    You are a medical expert tasked with providing a detailed and accurate explanation based solely on the provided context from a medical textbook. 
    Use the context to craft a comprehensive answer to the user's question, including:
    - A clear definition of the topic.
    - Detailed descriptions of its components, functions, and biological significance.
    - Relevant processes, mechanisms, or interactions mentioned in the context.
    - Any limitations in the context that prevent a complete answer.
    Structure the response with clear sections (e.g., Definition, Components, Functions, Significance) for readability, 
    similar to a textbook explanation. Do not include information beyond the provided context, and avoid speculative details.

    Context: {context}
    Question: {question}
    """

    try:
        vectorstore, error = get_vectorstore()
        if vectorstore is None:
            return [{"role": "assistant", "content": f"Error: {error}"}]

        llm, error = load_llm()
        if llm is None:
            return [{"role": "assistant", "content": f"Error: {error}"}]

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )

        response = qa_chain.invoke({'query': message})
        result = response["result"]
        source_documents = response["source_documents"]
        references = format_references(source_documents)

        # Return response in OpenAI-style format
        output = [{"role": "user", "content": message}, {"role": "assistant", "content": result}]
        if show_references:
            output.append({"role": "assistant", "content": f"**References**:\n{references}"})
        return output

    except Exception as e:
        return [{"role": "assistant", "content": f"Error: {str(e)}"}]

def clear_chat():
    """Clear the chat history."""
    return []

def main():
    with gr.Blocks(theme="soft") as interface:
        gr.Markdown("# Medical Chatbot")
        gr.Markdown("Ask medical questions based on *Robbins Basic Pathology 10th Edition*. Check 'Show References' to view sources.")
        chatbot = gr.Chatbot(label="Chat History", type="messages")
        message = gr.Textbox(label="Enter your medical query here (e.g., What is Biosynthetic Machinery?)", placeholder="Type your question here...")
        show_references = gr.Checkbox(label="Show References", value=False)
        submit_button = gr.Button("Submit Query")
        clear_button = gr.Button("Clear Chat")

        # Wire up the submit button to the chatbot response
        submit_button.click(
            fn=chatbot_response,
            inputs=[message, chatbot, show_references],
            outputs=[chatbot]
        )

        # Wire up the clear button to reset the chatbot
        clear_button.click(
            fn=clear_chat,
            outputs=[chatbot]
        )

    # Enable public sharing with authentication
    interface.launch(share=True)

if __name__ == "__main__":
    main()