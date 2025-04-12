# app.py
import streamlit as st
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv # To load .env file for local dev

# Import functions from utils.py
from utils import (
    load_and_process_pdf,
    get_faiss_index,
    load_embedding_model,
    retrieve_relevant_context,
    generate_safeguarding_response,
    GEMINI_MODEL_NAME,
    SAFETY_SETTINGS,
    PDF_PATH # Use PDF_PATH from utils
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Safeguarding Support Agent",
    page_icon="🛡️",
    layout="wide"
)

# --- Load API Key ---
# Prioritize Streamlit secrets, then environment variable, then input
load_dotenv() # Load .env file if it exists (for local development)
API_KEY_INPUT = None
configured = False

# Try Streamlit secrets first (for deployed apps)
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    if google_api_key:
        try:
            genai.configure(api_key=google_api_key)
            st.sidebar.success("API Key configured from Secrets.", icon="✔️")
            configured = True
        except Exception as e:
            st.sidebar.error(f"Invalid API Key in Secrets: {e}")
    else:
         st.sidebar.warning("GOOGLE_API_KEY not found in Streamlit Secrets.")
except FileNotFoundError:
    st.sidebar.info("Streamlit `secrets.toml` not found. Checking environment variables...")
    # Try environment variable next
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        try:
            genai.configure(api_key=google_api_key)
            st.sidebar.success("API Key configured from Environment Variable.", icon="✔️")
            configured = True
        except Exception as e:
            st.sidebar.error(f"Invalid API Key in Environment Variable: {e}")
    else:
        st.sidebar.warning("GOOGLE_API_KEY environment variable not set.")
        st.sidebar.info("Please enter your Google AI Studio API Key below.")
        API_KEY_INPUT = st.sidebar.text_input("Google AI Studio API Key:", type="password", key="api_key_input")
        if API_KEY_INPUT:
            try:
                genai.configure(api_key=API_KEY_INPUT)
                st.sidebar.success("API Key configured from Input.", icon="✔️")
                configured = True
            except Exception as e:
                 st.sidebar.error(f"Invalid API Key provided: {e}")


# --- Title and Description ---
st.title("🛡️ Safeguarding Support Agent (Nottingham)")
st.caption(f"Powered by Google Gemini ({GEMINI_MODEL_NAME}) | Based on policy document: `{PDF_PATH}`")
st.markdown("""
Welcome! Ask questions about the safeguarding policy.
This AI assistant uses the uploaded policy document to provide guidance based on its content.
**Remember:** This tool provides support but does *not* replace professional judgment, training, or mandatory reporting procedures. Always consult the Designated Safeguarding Lead (DSL) when required.
""")

# --- Initialization State & PDF Processing ---
# Use session state to store processed data and initialization status
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "messages" not in st.session_state:
    st.session_state.messages = [] # Store chat history: {"role": "user/assistant/system", "content": ...}
if "chat_session" not in st.session_state:
     st.session_state.chat_session = None
if "vector_store_data" not in st.session_state:
     st.session_state.vector_store_data = {"chunks": None, "embeddings": None}
if "faiss_index" not in st.session_state:
     st.session_state.faiss_index = None


# Button to initialize or re-initialize the system
# Placed in sidebar or main area depending on preference
if st.sidebar.button("Load/Reload Policy & Initialize AI", key="init_button", disabled=not configured):
    if not configured:
        st.error("Please configure the Google API Key before initializing.")
    else:
        with st.spinner(f"Processing `{PDF_PATH}` and initializing AI... Please wait."):
            st.session_state.messages = [] # Clear history on re-init
            st.session_state.chat_session = None # Reset chat session
            st.session_state.faiss_index = None # Reset index

            # Load PDF, generate embeddings (uses @st.cache_data)
            chunks, embeddings = load_and_process_pdf(PDF_PATH)

            if chunks and embeddings is not None:
                st.session_state.vector_store_data["chunks"] = chunks
                st.session_state.vector_store_data["embeddings"] = embeddings

                # Build FAISS index (using cached embeddings)
                st.session_state.faiss_index = get_faiss_index(embeddings)

                if st.session_state.faiss_index is not None:
                    # Initialize Gemini Model and Chat
                    try:
                        model = genai.GenerativeModel(GEMINI_MODEL_NAME, safety_settings=SAFETY_SETTINGS)
                        st.session_state.chat_session = model.start_chat(history=[]) # Start fresh chat
                        st.session_state.initialized = True
                        st.sidebar.success("System Initialized Successfully!")
                        st.rerun() # Rerun to update UI state
                    except Exception as e:
                        st.error(f"Failed to initialize Gemini Model/Chat: {e}")
                        st.session_state.initialized = False
                else:
                     st.error("Failed to build FAISS index after processing PDF.")
                     st.session_state.initialized = False
            else:
                st.error("Failed to process PDF or generate embeddings.")
                st.session_state.initialized = False

# Display initialization status
if not configured:
    st.warning("⚠️ Please configure your Google API Key in the sidebar to begin.")
elif not st.session_state.initialized:
    st.info("ℹ️ Click 'Load/Reload Policy & Initialize AI' in the sidebar to start.")
else:
    st.success("✅ System Initialized. You can now ask questions.")


# --- Chat Interface ---
# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Render markdown, including potentially unsafe HTML for simulation box
        st.markdown(message["content"], unsafe_allow_html=True if message["role"] == "system" else False)

# React to user input using st.chat_input
if prompt := st.chat_input("Ask your question here...", disabled=not st.session_state.initialized):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Placeholder for streaming effect (optional)
        full_response_content = ""
        function_sim_output = None
        start_time = time.time()

        with st.spinner("Thinking..."):
            # 1. Retrieve Context (Requires initialized system)
            embedding_model = load_embedding_model() # Get cached model
            if st.session_state.faiss_index and embedding_model:
                 context = retrieve_relevant_context(
                     prompt,
                     st.session_state.faiss_index,
                     embedding_model,
                     st.session_state.vector_store_data["chunks"],
                     st.session_state.vector_store_data["embeddings"] # Pass embeddings
                 )
            else:
                 context = "Error: Vector store not ready."
                 st.error("Could not retrieve context because the vector store is not initialized.")

            # 2. Generate Response (Requires initialized chat session)
            if st.session_state.chat_session:
                try:
                    ai_response_text, function_sim_output = generate_safeguarding_response(
                        st.session_state.chat_session, # Use chat session from state
                        prompt,
                        context,
                        user_role="School Staff Member" # Example role
                    )
                    full_response_content = ai_response_text
                except Exception as e:
                     full_response_content = f"An error occurred: {e}"
                     st.error(full_response_content)
            else:
                 full_response_content = "Error: Chat session not initialized."
                 st.error(full_response_content)

        end_time = time.time()

        # Display the final response
        message_placeholder.markdown(full_response_content)
        st.caption(f"Response generated in {end_time - start_time:.2f} seconds")

    # Add AI response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response_content})

    # If a function simulation occurred, display it separately and add to history
    if function_sim_output:
        st.warning("Function Simulation Triggered:")
        # Use markdown with unsafe_allow_html=True because the function returns an HTML div
        st.markdown(function_sim_output, unsafe_allow_html=True)
        # Add a system message to history indicating the simulation happened
        st.session_state.messages.append({"role": "system", "content": function_sim_output})
