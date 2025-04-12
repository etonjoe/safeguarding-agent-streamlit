# utils.py
import os
import json
import datetime
import streamlit as st # Import streamlit for caching etc.
import google.generativeai as genai
# VVV Corrected this line VVV
from google.generativeai.types import HarmCategory, HarmBlockThreshold # Removed Part from here
# ^^^ Corrected this line ^^^

# PDF Processing & Vector Store Libraries
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Configuration Constants ---
# These could potentially be Streamlit inputs if needed
PDF_PATH = "safeguarding_policy.pdf"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GEMINI_MODEL_NAME = 'gemini-1.5-flash' # Or 'gemini-1.5-pro', 'gemini-pro'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 4 # Number of context chunks to retrieve

# Safety settings for Gemini
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# --- Tool/Function Definitions ---

# Python function that gets executed when Gemini calls "simulate_dsl_escalation"
# MODIFIED: Returns the notification string instead of printing
def simulate_dsl_escalation(concern_summary: str, urgency: str, reported_by: str, details: str) -> str:
    """
    Simulates the action of escalating a safeguarding concern to the DSL.
    Returns a formatted string summarizing the simulated action.
    """
    # Add location and current time context
    location = "Nottingham, UK"
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Use local time for display
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") # Use UTC for logging

    notification = f"""
    <div style="border: 2px solid red; padding: 10px; border-radius: 5px; background-color: #ffebee;">
    <strong>🚨 DSL ESCALATION SIMULATION ({location}) 🚨</strong><br>
    ---------------------------------------------------------<br>
    <strong>Simulated Time:</strong> {current_time_str}<br>
    <strong>Urgency:</strong> {urgency}<br>
    <strong>Reported By:</strong> {reported_by}<br>
    <strong>Concern Summary:</strong> {concern_summary}<br>
    <strong>Details Provided:</strong> {details}<br>
    ---------------------------------------------------------<br>
    <strong>Action:</strong> Logged and simulated notification sent to DSL. (In a real system, this would trigger an alert.)
    </div>
    """
    print(f"--- DSL Escalation Simulated ({timestamp} / Location: {location}) ---") # Keep console log with UTC
    return notification # Return the formatted string for Streamlit display

# Tool definition for Gemini
dsl_escalation_tool = {
    "function_declarations": [
        {
            "name": "simulate_dsl_escalation",
            "description": "Use this function ONLY when a safeguarding concern requires immediate escalation to the Designated Safeguarding Lead (DSL) based on the LATEST policy context provided. Requires summarizing the concern and assessing urgency derived from the LATEST query and context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concern_summary": {
                        "type": "string",
                        "description": "A concise summary of the core safeguarding concern from the LATEST user query (e.g., 'Child disclosed physical harm by sibling')."
                    },
                    "urgency": {
                        "type": "string",
                        "description": "The urgency level based on the LATEST policy context (e.g., 'Immediate', 'High', 'Standard'). Use 'Immediate' for disclosures of harm, immediate danger, etc."
                    },
                     "details": {
                        "type": "string",
                        "description": "Provide key details from the LATEST query or context that justify the escalation (e.g., 'Child stated sibling hits them', 'Observed unexplained injuries')."
                     }
                },
                "required": ["concern_summary", "urgency", "details"]
            }
        }
    ]
}

# --- Core Functions ---

# Use Streamlit caching for resource-intensive loading
@st.cache_resource
def load_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    """Loads the Sentence Transformer model using Streamlit's caching."""
    print(f"[Cache] Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)

@st.cache_data # Cache based on input PDF path (or file content if using uploader)
def load_and_process_pdf(pdf_path=PDF_PATH):
    """Loads, splits PDF text, generates embeddings, and creates FAISS index."""
    print(f"[Cache] Processing PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        st.error(f"Policy PDF not found at: {pdf_path}. Please place the file correctly.")
        return None, None # Return None if file not found

    # 1. Load and Split
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                 full_text += extracted + "\n"

        chunks = []
        start = 0
        while start < len(full_text):
            end = min(start + CHUNK_SIZE, len(full_text))
            if end < len(full_text):
                period_pos = full_text.rfind('.', start, end)
                newline_pos = full_text.rfind('\n', start, end)
                boundary = max(period_pos, newline_pos)
                if boundary > start + (CHUNK_SIZE * 0.5):
                     end = boundary + 1
            chunks.append(full_text[start:end].strip())
            next_start = start + CHUNK_SIZE - CHUNK_OVERLAP
            if next_start <= start :
                 next_start = start + int(CHUNK_SIZE * 0.1)
            start = next_start if next_start < end else end

        processed_chunks = [chunk for chunk in chunks if chunk and len(chunk.split()) > 5]
        if not processed_chunks:
             st.warning("No text chunks could be extracted from the PDF. Check the PDF content.")
             return None, None
        print(f"[Cache] PDF split into {len(processed_chunks)} chunks.")
    except Exception as e:
        st.error(f"Error reading or splitting PDF: {e}")
        return None, None

    # 2. Generate Embeddings
    try:
        embedding_model = load_embedding_model() # Get cached model
        print("[Cache] Generating embeddings...")
        embeddings = embedding_model.encode(processed_chunks, show_progress_bar=False) # Progress bar might not work well with caching/streamlit
        print("[Cache] Embeddings generated.")
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None, None

    # Return chunks and embeddings separately for index creation outside cache if needed
    return processed_chunks, embeddings

# Function to get or build the FAISS index (handles state)
def get_faiss_index(embeddings):
     """Builds FAISS index from embeddings."""
     if embeddings is None:
          return None
     try:
          dimension = embeddings.shape[1]
          index = faiss.IndexFlatL2(dimension)
          index.add(np.array(embeddings).astype('float32'))
          print(f"[State] FAISS index accessed/built with {index.ntotal} vectors.")
          return index
     except Exception as e:
          st.error(f"Error building FAISS index from embeddings: {e}")
          return None

def retrieve_relevant_context(query: str, index, embedding_model, text_chunks: list[str], k: int = TOP_K_RESULTS) -> str:
    """Retrieves the top-k relevant text chunks from the vector store."""
    # Renamed embeddings_np argument to avoid conflict if passed directly, though it's not used here
    if index is None or embedding_model is None or text_chunks is None:
         st.warning("Vector store components not available for retrieval.")
         return "Error: Could not retrieve context."
    try:
        print(f"[RT] Embedding query: '{query[:50]}...'")
        query_embedding = embedding_model.encode([query])
        print(f"[RT] Searching FAISS index for top {k} relevant chunks...")
        distances, indices = index.search(np.array(query_embedding).astype('float32'), k)

        # Filter out potential invalid indices (if k > number of documents)
        valid_indices = [i for i in indices[0] if 0 <= i < len(text_chunks)]
        retrieved_chunks = [text_chunks[i] for i in valid_indices]

        print(f"[RT] Retrieved {len(retrieved_chunks)} chunks.")
        return "\n\n---\n\n".join(retrieved_chunks)
    except Exception as e:
        st.error(f"Error during context retrieval: {e}")
        return "Error: Failed to retrieve context from policy."


def generate_safeguarding_response(
    chat, # Removed ChatSession type hint
    query: str,
    context: str,
    user_role: str = "School Staff Member"
    ):
    """
    Generates a response using Gemini, handles function calls.

    Returns:
        tuple: (response_text: str, function_simulation_output: str or None)
    """
    print("[RT] Preparing request for Gemini (using chat history)...")
    prompt_content = f"""
    *Instructions for AI:*
    You are an AI Safeguarding Support Agent for school staff in Nottingham, UK. Your response time should reflect the current time ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).
    Your purpose is to provide clear, actionable guidance based *strictly* on the provided safeguarding policy context.
    Prioritize child safety. Do not provide advice outside of the given policy context.
    If the policy context is missing or insufficient for the *latest query*, state that clearly.
    Use the conversation history (implicitly provided) to understand follow-up questions and maintain context.
    However, base your specific safeguarding advice, procedures, and reporting requirements *strictly* on the **Newly Retrieved Policy Context** provided below, as this is most relevant to the **Latest User Query**.

    *User Role:* {user_role}
    *Current Location:* Nottingham, England, United Kingdom

    *Newly Retrieved Policy Context (Relevant to Latest Query):*
    --- START OF CONTEXT ---
    {context}
    --- END OF CONTEXT ---

    *Latest User Query:*
    {query}

    *Your Task:*
    1. Analyze the **Latest User Query** in light of the **Newly Retrieved Policy Context** and conversation history.
    2. Provide a concise summary of the situation and main risks based *only* on the **Newly Retrieved Policy Context**.
    3. Generate clear, numbered Action Steps based *only* on the procedures described in the **Newly Retrieved Policy Context**.
    4. Determine if reporting to the DSL is required and state the urgency based *only* on the **Newly Retrieved Policy Context**. (e.g., "Report Required: Yes (Immediate)").
    5. **Crucially:** If the situation described in the **Latest User Query** clearly requires *immediate* escalation according to the **Newly Retrieved Policy Context**, you MUST call the `simulate_dsl_escalation` function with details derived from the latest query/context. Do not call it for general advice or based on outdated history if the new context doesn't support it.
    6. Format your response clearly with headings: "Summary:", "Action Steps:", "Report Required:".
    7. Be factual, objective, and avoid judgmental language.

    *Your Response to the Latest User Query:*
    """

    print("[RT] Sending request to Gemini model...")
    function_simulation_output = None # Initialize

    try:
        # Send the LATEST query/context/instructions using the persistent chat object
        response = chat.send_message(
            prompt_content,
            tools=[dsl_escalation_tool] # Pass list of tools
            )

        # --- Function Call Handling Logic ---
        response_part = response.candidates[0].content.parts[0] if response.candidates else None
        # Check for function_call attribute directly on the part
        if response_part and hasattr(response_part, 'function_call') and response_part.function_call:
            function_call = response_part.function_call
            print(f"[RT] Gemini requested function call: {function_call.name}")

            if function_call.name == "simulate_dsl_escalation":
                try:
                    args = function_call.args
                    # *** EXECUTE THE ACTUAL PYTHON FUNCTION ***
                    # It now returns the HTML string
                    function_simulation_output = simulate_dsl_escalation(
                        concern_summary=args.get('concern_summary', 'N/A'),
                        urgency=args.get('urgency', 'N/A'),
                        reported_by=user_role,
                        details=args.get('details', 'N/A')
                    )

                    # *** Send function result back to Gemini ***
                    print("[RT] Sending function execution result back to Gemini...")
                    response = chat.send_message(
                         # VVV Corrected this line VVV
                         part=genai.Part(function_response={ # Use genai.Part
                              "name": function_call.name,
                              # Send confirmation back, not the HTML
                              "response": {"status": "DSL Escalation Simulated OK"}
                              }),
                         # ^^^ Corrected this line ^^^
                         tools=[dsl_escalation_tool] # Resend tools
                         )

                except Exception as e:
                    error_msg = f"Failed to execute function call '{function_call.name}': {e}"
                    print(f"[ERROR] {error_msg}")
                    st.error(error_msg)
                    # Optionally inform Gemini about the error
                    try:
                        # VVV Also correct this line if sending error back VVV
                        chat.send_message(part=genai.Part(function_response={"name": function_call.name, "response": {"error": error_msg}}))
                        # ^^^ Also correct this line if sending error back ^^^
                    except Exception as send_err: print(f"Error sending error back to Gemini: {send_err}")
            else:
                 print(f"[WARN] Gemini requested unknown function: {function_call.name}")
                 st.warning(f"Model requested an unsupported action '{function_call.name}'.")

        # Extract final text response from Gemini
        final_response_text = ""
        # Ensure response.candidates exists and has content before accessing parts
        if response.candidates and response.candidates[0].content.parts:
            final_response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
        elif not function_simulation_output: # If no text AND no function call happened
             final_response_text = "[AI did not provide a textual response for this turn]"

        return final_response_text, function_simulation_output

    except Exception as e:
        error_msg = f"An error occurred during Gemini API call: {e}"
        print(f"[ERROR] {error_msg}")
        # Attempt to get partial response if available
        partial_text = "[ERROR]"
        try:
             # Check response candidate structure carefully after error
             if response and response.candidates and len(response.candidates) > 0 and response.candidates[0].content and response.candidates[0].content.parts:
                 partial_text += " Partial response: " + "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
        except Exception: pass # Ignore errors trying to get partial text
        st.error(error_msg)
        return partial_text, None # Return error text
