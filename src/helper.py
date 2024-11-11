from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings




#Extract data from the PDF

def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls = PyPDFLoader)
    documents = loader.load()
    return documents



#Create text chunk
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap = 1000)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings



#Load and split the data
def load_split_pdf(file_path):
  pdf_loader = PyPDFLoader(file_path)
  pages = pdf_loader.load_and_split()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
  context = "\n\n".join(str(page.page_content)for page in pages)
  texts = text_splitter.split_text(context)
  return texts