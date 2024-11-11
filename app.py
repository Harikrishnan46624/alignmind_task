from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.prompt import *
import google.generativeai as genai
from langchain_google_genai import  ChatGoogleGenerativeAI


app = Flask(__name__)


embeddings = download_hugging_face_embeddings()


DB_FAISS_PATH = 'vectorstore/db_faiss'

# Load the FAISS database with the embeddings
db = FAISS.load_local("vectorstore/db_faiss", embeddings=embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


chain_type_kwargs = {"prompt": PROMPT}


# llm = CTransformers(model=r"E:\projects\Medical-Chatbot-Project-\model\llama-2-7b-chat.ggmlv3.q2_K.bin",
#                   model_type="llama",
#                   config={'max_new_tokens':512,
#                           'temperature':0.8})

def setup_llm_model():
  Google_Api_Key = "AIzaSyCqmUocH4Vjf2qGMwqluIxVKvLFwNBcPWU"
  genai.configure(api_key=Google_Api_Key)
  model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=Google_Api_Key, temperature=0.2, convert_system_message_to_human=True)
  return model


llm = setup_llm_model()

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=db.as_retriever(search_kwargs={'k': 5}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug=False)