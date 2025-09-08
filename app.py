import os
import shutil
from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

# Load API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Tell FastAPI where your HTML files are
templates = Jinja2Templates(directory="templates")  

# ---------------- PDF Functions ----------------
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_conversational_chain():
    prompt_template = """
    You are a friendly and respectful AI. Answer conversational messages naturally. For knowledge-based questions, 
    reply briefly using only the uploaded document. If the answer isn’t in the document,
    respond: "Make sure you ask from uploaded document." Keep answers concise, clear, and relevant.
    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# ---------------- In-Memory FAISS Storage ----------------
vector_db = None  # Global variable to store the FAISS index in memory

def get_vector_store(text_chunks):
    global vector_db
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', async_client=False)
    vector_db = FAISS.from_texts(chunks, embedding=embeddings)

# ---------------- Pydantic Model for JSON ----------------
class QuestionRequest(BaseModel):
    question: str

# ---------------- Routes ----------------
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    global vector_db
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = get_pdf_text(file_path)
    chunks = get_text_chunks(text)

    get_vector_store(chunks)

    os.remove(file_path)
    return JSONResponse({"message": "PDF processed successfully!"})

@app.post("/ask/")
async def ask_question(req: QuestionRequest):
    global vector_db
    if vector_db is None:
        return JSONResponse({"error": "No PDF uploaded yet!"})
    
    docs = vector_db.similarity_search(req.question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": req.question}, return_only_outputs=True)
    return JSONResponse({"answer": response["output_text"]})

# Optional: Handle HEAD request to avoid 405 warnings from Render
@app.head("/")
async def root_head():
    return {}

# ---------------- Render Deployment Fix ----------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render assigns this automatically
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
