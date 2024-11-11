from src.helper import download_hugging_face_embeddings, load_split_pdf
from langchain.vectorstores import FAISS


text_chunks = load_split_pdf("Assets\Attention is All You Need.pdf")


embeddings = download_hugging_face_embeddings()



#Create Vector Database
def create_db(text_chunks, embeddings):
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    
    db = FAISS.from_texts(text_chunks, embeddings)
    db.save_local(DB_FAISS_PATH)


create_db(text_chunks, embeddings)