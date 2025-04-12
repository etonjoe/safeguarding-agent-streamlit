# Safeguarding Support Agent (Streamlit + Gemini)

This application provides an AI-powered interface to query a school's safeguarding policy document using Google's Gemini model with Retrieval-Augmented Generation (RAG).

**Disclaimer:** This tool is intended for informational support and does *not* replace professional judgment, mandatory training, or required reporting procedures. Always consult with your Designated Safeguarding Lead (DSL) for definitive guidance and follow established protocols.

## Features

* **Policy Q&A:** Ask questions in natural language about the safeguarding policy.
* **RAG Implementation:** Retrieves relevant sections from the policy PDF to ground the AI's answers.
* **Conversational Memory:** Remembers the context of the current chat session.
* **Function Calling:** Simulates actions like escalating to the DSL based on policy triggers identified by the AI.
* **Streamlit UI:** Easy-to-use web interface.

## Project Structure
safeguarding-agent-streamlit/ 

 ├── .gitignore
 
 ├── app.py                   # Main Streamlit application
 
 ├── requirements.txt         # Python dependencies
 
 ├── safeguarding_policy.pdf  # Your policy document (PLACE IT HERE)
 
 ├── utils.py                 # Helper functions (PDF, Vector Store, Gemini Call)
 
 └── README.md                # This file


## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd safeguarding-agent-streamlit
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place Policy Document:**
    * Make sure your school's safeguarding policy document is in PDF format.
    * Rename it to `safeguarding_policy.pdf`.
    * Place this file directly inside the `safeguarding-agent-streamlit` directory.

5.  **Configure Google API Key:** You need an API key from Google AI Studio ([https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)). You can configure it in one of these ways (in order of preference for deployment):
    * **(Recommended for Deployment - e.g., Streamlit Community Cloud):** Use Streamlit Secrets. Create a file `.streamlit/secrets.toml` with the following content:
        ```toml
        GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        ```
        *Make sure this file is NOT committed to GitHub (it's included in the `.gitignore`).*
    * **(Environment Variable):** Set an environment variable named `GOOGLE_API_KEY`:
        ```bash
        # macOS/Linux
        export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        # Windows (Command Prompt)
        set GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        # Windows (PowerShell)
        $env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        ```
        You can also create a `.env` file in the project root (add your key there like `GOOGLE_API_KEY=...`) - this is loaded automatically for local development thanks to `python-dotenv`. *Ensure `.env` is in your `.gitignore`.*
    * **(Manual Input - Local Development Only):** If no secrets or environment variable is found, the Streamlit app will prompt you to enter the API key directly in the sidebar.

## Running the Application

1.  **Ensure your virtual environment is active.**
2.  **Run the Streamlit app from the terminal:**
    ```bash
    streamlit run app.py
    ```
3.  **Open your web browser** to the local URL provided by Streamlit (usually `http://localhost:8501`).
4.  **Configure API Key** if prompted.
5.  Click "**Load/Reload Policy & Initialize AI**" in the sidebar. Wait for the processing to complete.
6.  Start asking questions in the chat input box!

## How it Works

1.  **Initialization:** When you click the "Initialize" button, the app reads `safeguarding_policy.pdf`.
2.  **Chunking & Embeddings:** The PDF text is split into smaller chunks. A sentence-transformer model converts these chunks into numerical vectors (embeddings).
3.  **Vector Store:** A FAISS index is built in memory to store these embeddings for efficient searching.
4.  **Chatting:**
    * When you ask a question, your query is embedded.
    * FAISS finds the most similar chunks (embeddings) from the policy document (Retrieval).
    * Your query, the retrieved policy chunks (Context), and the chat history are sent to the Gemini model (Generation).
    * Gemini generates an answer, grounded in the provided policy context. If the query triggers an escalation based on the policy, it may call the `simulate_dsl_escalation` function.
    * The response and any simulated actions are displayed.

## Deployment to GitHub

1.  Create a new repository on GitHub.
2.  Initialize Git in your local `safeguarding-agent-streamlit` directory (if you haven't already):
    ```bash
    git init
    git add .
    git commit -m "Initial commit of Safeguarding Agent Streamlit app"
    ```
3.  Add the GitHub repository as a remote:
    ```bash
    git remote add origin <your-github-repo-url>
    git branch -M main
    git push -u origin main
    ```
4.  **Important:** Ensure your `.gitignore` file is correctly configured to prevent committing sensitive information like `.env` files or `.streamlit/secrets.toml`.

## Further Development Ideas

* Allow uploading different policy PDFs via the UI.
* Implement more robust error handling.
* Add user authentication.
* Persist the FAISS index to disk for faster restarts (especially for large policies).
* Integrate with actual notification systems instead of simulation.
* Add options to select different Gemini or embedding models.
* Improve text chunking strategies.
