# utils.py (Version with SyntaxError Fix and Index Sanitization)
import os
import json
import datetime
import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # Removed Part
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import traceback # Import traceback

# --- Configuration Constants ---
PDF_PATH = "safeguarding_policy.pdf"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GEMINI_MODEL_NAME = 'gemini-1.5-flash' # Or 'gemini-1.5-pro', 'gemini-pro'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 4

# Safety settings for Gemini
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# --- Tool/Function Definitions ---

def simulate_dsl_escalation(concern_summary: str, urgency: str, reported_by: str, details: str) -> str:
    """Simulates DSL escalation and returns formatted string."""
    location = "Nottingham, UK" # Using location context
    # Use Saturday, April 12, 2025 17:04:12 BST context if needed, otherwise use current time
    # For consistency, let's use the system's current time but acknowledge the context time
    context_time_str = "2025-04-12 17:04:12 BST" # From user context
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # System time
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    notification = f"""
    <div style="border: 2px solid red; padding: 10px; border-radius: 5px; background-color: #ffebee;">
    <strong>🚨 DSL ESCALATION SIMULATION ({location}) 🚨</strong><br>
    ---------------------------------------------------------<br>
    <strong>Simulated Time (System):</strong> {current_time_str}<br>
    <strong>(Context Time was: {context_time_str})</strong><br>
    <strong>Urgency:</strong> {urgency}<br>
    <strong>Reported By:</strong> {reported_by}<br>
    <strong>Concern Summary:</strong> {concern_summary}<br>
    <strong>Details Provided:</strong> {details}<br>
    ---------------------------------------------------------<br>
    <strong>Action:</strong> Logged and simulated notification sent to DSL. (In a real system, this would trigger an alert.)
    </div>
    """
    print(f"--- DSL Escalation Simulated ({timestamp} / Location: {location}) ---")
    return notification

# VVV --- Corrected and Complete Dictionary Definition --- VVV
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
                }, # Added comma if needed, ensure correct closing braces
                "required": ["concern_summary", "urgency", "details"]
            } # Ensure correct closing braces
        } # Ensure correct closing braces
    ] # Ensure correct closing braces
}
# ^^^ --- Corrected and Complete Dictionary Definition --- ^^^


# --- Core Functions ---

@st.cache_resource
def load_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    """Loads the Sentence Transformer model using Streamlit's caching."""
    print(f"[Cache] Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model '{model_name}': {e}")
        return None

def load_and_split_pdf(pdf_path=PDF_PATH):
    """Loads text from a PDF and splits it into chunks."""
    print(f"Loading and splitting PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        st.error(f"Policy PDF not found at: {pdf_path}.")
        return None
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted: full_text += extracted + "\n"
        if not full_text.strip():
            st.warning("No text could be extracted from the PDF.")
            return []
        chunks = []
        start = 0
        while start < len(full_text):
            end = min(start + CHUNK_SIZE, len(full_text))
            if end < len(full_text):
                period_pos = full_text.rfind('.', start, end); newline_pos = full_text.rfind('\n', start, end)
                boundary = max(period_pos, newline_pos)
                if boundary > start + (CHUNK_SIZE * 0.5): end = boundary + 1
            chunks.append(full_text[start:end].strip())
            next_start = start + CHUNK_SIZE - CHUNK_OVERLAP
            if next_start <= start : next_start = start + int(CHUNK_SIZE * 0.1)
            start = next_start if next_start < end else end
        processed_chunks = [chunk for chunk in chunks if chunk and len(chunk.split()) > 5]
        print(f"PDF split into {len(processed_chunks)} chunks.")
        return processed_chunks
    except Exception as e:
        st.error(f"Error reading or splitting PDF: {e}")
        return None

def create_vector_store(text_chunks):
    """Generates embeddings, sanitizes them, builds FAISS index, returns store components."""
    if not text_chunks:
        st.error("Cannot create vector store: No text chunks provided.")
        return None, None, None
    embedding_model = load_embedding_model()
    if embedding_model is None:
        st.error("Cannot create vector store: Embedding model failed to load.")
        return None, None, None
    try:
        print("Generating embeddings for text chunks...")
        embeddings_np = embedding_model.encode(text_chunks, show_progress_bar=True)
        print(f"Generated {embeddings_np.shape[0]} embeddings with dimension {embeddings_np.shape[1]}.")
        print("Sanitizing embeddings (checking for NaN/Inf)...")
        nan_mask = np.isnan(embeddings_np).any(axis=1)
        inf_mask = np.isinf(embeddings_np).any(axis=1)
        invalid_mask = nan_mask | inf_mask
        if invalid_mask.any():
            num_invalid = invalid_mask.sum()
            st.warning(f"Found and removed {num_invalid} invalid embedding(s) (NaN or Inf) out of {embeddings_np.shape[0]}. Corresponding text chunks were skipped.")
            print(f"[WARN] Removing {num_invalid} invalid embeddings.")
            valid_mask = ~invalid_mask
            embeddings_np_sanitized = embeddings_np[valid_mask]
            text_chunks_sanitized = [chunk for i, chunk in enumerate(text_chunks) if valid_mask[i]]
            if embeddings_np_sanitized.shape[0] == 0:
                st.error("No valid embeddings remained after sanitization. Cannot build index.")
                return None, None, None
        else:
            print("No invalid embeddings found.")
            embeddings_np_sanitized = embeddings_np
            text_chunks_sanitized = text_chunks
        print("Creating FAISS index with sanitized embeddings...")
        dimension = embeddings_np_sanitized.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings_np_sanitized).astype('float32'))
        print(f"FAISS index created successfully with {index.ntotal} vectors.")
        return index, text_chunks_sanitized, embeddings_np_sanitized
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        traceback.print_exc()
        return None, None, None

def retrieve_relevant_context(query: str, index, embedding_model, text_chunks: list[str], k: int = TOP_K_RESULTS) -> str:
    """Retrieves the top-k relevant text chunks from the vector store with added validation."""
    if index is None or embedding_model is None or text_chunks is None:
         st.warning("Vector store components not available for retrieval.")
         return "Error: Could not retrieve context - Vector store not initialized."
    try:
        print(f"[RT] Embedding query: '{query[:50]}...'")
        query_embedding_list = embedding_model.encode([query])
        if query_embedding_list is None or query_embedding_list.shape[0] == 0:
             st.error("Failed to generate query embedding (returned None or empty).")
             return "Error: Failed to generate query embedding."
        query_embedding_np = np.array(query_embedding_list).astype('float32')
        if query_embedding_np.ndim != 2 or query_embedding_np.shape[0] != 1:
             st.error(f"Query embedding has unexpected shape: {query_embedding_np.shape}")
             return "Error: Query embedding shape issue."
        if np.isnan(query_embedding_np).any():
            st.error("Query embedding contains NaN values. Cannot perform search.")
            print(f"[ERROR] Query embedding contains NaN: {query_embedding_np}")
            return "Error: Invalid query embedding generated (NaN)."
        if np.isinf(query_embedding_np).any():
            st.error("Query embedding contains Inf values. Cannot perform search.")
            print(f"[ERROR] Query embedding contains Inf: {query_embedding_np}")
            return "Error: Invalid query embedding generated (Inf)."
        print(f"[RT] Searching FAISS index (size {index.ntotal}) for top {k} relevant chunks...")
        distances, indices = index.search(query_embedding_np, k)
        if indices is None or len(indices) == 0 or len(indices[0]) == 0:
             print("[RT] No relevant indices found by FAISS search.")
             return "No specific policy context found for this query."
        valid_indices_in_index = [idx for idx in indices[0] if 0 <= idx < index.ntotal]
        if not valid_indices_in_index:
            print("[RT] No valid indices returned by FAISS search.")
            return "No specific policy context found matching the query criteria."
        retrieved_chunks = [text_chunks[i] for i in valid_indices_in_index]
        print(f"[RT] Retrieved {len(retrieved_chunks)} chunks.")
        return "\n\n---\n\n".join(retrieved_chunks)
    except Exception as e:
        print("--- ERROR DURING CONTEXT RETRIEVAL ---")
        traceback.print_exc()
        print("--- END ERROR TRACEBACK ---")
        error_message_detail = f"Details: {e}"
        if "The truth value of an array" in str(e):
             st.error(f"Persistent error during FAISS search: {e}. This might indicate an issue with the index or specific query interaction, even after sanitization.")
             error_message_detail = f"Persistent FAISS Error: {e}"
        else:
             st.error(f"Error during context retrieval: {e}")
        return f"Error: Failed to retrieve context from policy. {error_message_detail}"

def generate_safeguarding_response(
    chat, # Removed ChatSession type hint
    query: str,
    context: str,
    user_role: str = "School Staff Member"
    ):
    """Generates a response using Gemini, handles function calls."""
    # Add location context from user prompt
    location = "Nottingham, England, United Kingdom" # From user context
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # System time
    context_time_str = "2025-04-12 17:04:12 BST" # From user context

    prompt_content = f"""
    *Instructions for AI:*
    You are an AI Safeguarding Support Agent for school staff in Nottingham, UK. Your response time should reflect the current system time ({current_time_str}). The user context time is {context_time_str}.
    Your purpose is to provide clear, actionable guidance based *strictly* on the provided safeguarding policy context.
    Prioritize child safety. Do not provide advice outside of the given policy context.
    If the policy context is missing or insufficient for the *latest query*, state that clearly.
    Use the conversation history (implicitly provided) to understand follow-up questions and maintain context.
    However, base your specific safeguarding advice, procedures, and reporting requirements *strictly* on the **Newly Retrieved Policy Context** provided below, as this is most relevant to the **Latest User Query**.

    *User Role:* {user_role}
    *Current Location:* {location}

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
    function_simulation_output = None
    try:
        response = chat.send_message(prompt_content, tools=[dsl_escalation_tool])
        response_part = response.candidates[0].content.parts[0] if response.candidates else None
        if response_part and hasattr(response_part, 'function_call') and response_part.function_call:
            function_call = response_part.function_call
            print(f"[RT] Gemini requested function call: {function_call.name}")
            if function_call.name == "simulate_dsl_escalation":
                try:
                    args = function_call.args
                    function_simulation_output = simulate_dsl_escalation(
                        concern_summary=args.get('concern_summary', 'N/A'),
                        urgency=args.get('urgency', 'N/A'),
                        reported_by=user_role,
                        details=args.get('details', 'N/A')
                    )
                    print("[RT] Sending function execution result back to Gemini...")
                    response = chat.send_message(
                         part=genai.Part(function_response={ # Use genai.Part
                              "name": function_call.name,
                              "response": {"status": "DSL Escalation Simulated OK"}
                              }),
                         tools=[dsl_escalation_tool]
                         )
                except Exception as e:
                    error_msg = f"Failed to execute function call '{function_call.name}': {e}"
                    print(f"[ERROR] {error_msg}")
                    st.error(error_msg)
                    try:
                        chat.send_message(part=genai.Part(function_response={"name": function_call.name, "response": {"error": error_msg}}))
                    except Exception as send_err: print(f"Error sending error back to Gemini: {send_err}")
            else:
                 print(f"[WARN] Gemini requested unknown function: {function_call.name}")
                 st.warning(f"Model requested an unsupported action '{function_call.name}'.")
        final_response_text = ""
        if response.candidates and response.candidates[0].content.parts:
            final_response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
        elif not function_simulation_output:
             final_response_text = "[AI did not provide a textual response for this turn]"
        return final_response_text, function_simulation_output
    except Exception as e:
        error_msg = f"An error occurred during Gemini API call: {e}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        partial_text = "[ERROR]"
        try:
             if response and response.candidates and len(response.candidates) > 0 and response.candidates[0].content and response.candidates[0].content.parts:
                 partial_text += " Partial response: " + "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
        except Exception: pass
        st.error(error_msg)
        return partial_text, None
