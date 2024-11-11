# alignmind_task



![Sampele Image](Assets\chatbot_image.png)



# RAG Chatbot with Gemini Pro Model
This project implements a Retrieval-Augmented Generation (RAG) chatbot using the Gemini Pro LLM model, FAISS as a vector database, and LangChain’s retrieval chain. The chatbot is designed to respond accurately and contextually to user queries by integrating retrieval-based search with response generation.



# Project Structure

├── src
│   ├── __init__.py          # Initializes the src module                                                                                                                                   
│   ├── helper.py            # Contains helper functions used across the project
│   └── prompt.py            # Manages prompts for interacting with the LLM
├── static
│   ├── style.css            # CSS styles for the web interface
│   └── .gitkeep             # Keeps the static directory in version control
├── templates
│   └── chat.html            # HTML template for the chatbot interface
├── .env                     # Environment variables file (do not commit API keys here)
├── setup.py                 # Script for setting up the project dependencies
├── research
│   └── trials.ipynb         # Jupyter notebook for experimental trials and testing
├── app.py                   # Main application file to run the FastAPI or Flask server
├── store_index.py           # Script to load and index documents into FAISS




# Setup Instructions

Clone the Repository:
git clone [<repository-url>](https://github.com/Harikrishnan46624/alignmind_task.git)
cd <repository-directory>

Install Dependencies:
pip install -r requirements.txt

Environment Variables:
Create a .env file in the project root.
Add your Gemini Pro API key and other environment variables:

GEMINI_API_KEY=your_gemini_api_key

Indexing Documents:
Run store_index.py to load and index your documents in FAISS. Ensure you have some PDF or document files ready to be loaded.
python store_index.py

Run the Application:
Start the API server by running app.py.

python app.py
Access the Chatbot Interface:

Go to http://localhost:8000/chat to interact with the chatbot using the provided HTML interface.
