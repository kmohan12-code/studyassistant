import os
import shutil
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores.faiss import FAISS

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")  


# ----------- Utility functions -----------

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


def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', async_client=False)

        # Delete old FAISS index if it exists
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")

        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True

    except Exception as e:
        # Handles quota exceeded or embedding API errors
        print("Error creating vector store:", str(e))
        return False


def get_conversational_chain():
    prompt_template = """
    Be a friendly AI and answer respectfully. First, check what the user asked.
    If it’s a conversational message (like greetings, introductions, or small talk), respond naturally without using the uploaded document. 
    If it’s a knowledge-based or subject-related question, answer in the shortest possible paragraph using only the uploaded document. 
    If the answer is not in the document, reply with: 'Make sure you ask from uploaded document'
    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# ----------- Routes -----------

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = get_pdf_text(file_path)
    chunks = get_text_chunks(text)
    success = get_vector_store(chunks)

    os.remove(file_path)

    if not success:
        return JSONResponse(
            {"error": "Failed to create embeddings. Possibly quota exceeded."}, status_code=500
        )

    return JSONResponse({"message": "PDF processed successfully!"})


@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    if not os.path.exists("faiss_index/index.faiss"):
        return JSONResponse(
            {"error": "No FAISS index found. Please upload and process a PDF first."},
            status_code=400
        )

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', async_client=False)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return JSONResponse({"answer": response["output_text"]})

    except Exception as e:
        return JSONResponse(
            {"error": f"Error during question answering: {str(e)}"}, status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
