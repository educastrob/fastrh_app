import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
import requests
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_MODEL = "gemini-1.5-flash"

persist_directory = "./chroma_data"
chroma_client = chromadb.PersistentClient(path=persist_directory)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
collection = chroma_client.get_or_create_collection(name="curriculos", embedding_function=embedding_function)

def process_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    
    for j, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            metadatas=[{"source": file.name, "chunk": j}],
            ids=[f"{file.name}_chunk_{j}"]
        )
    
    st.success(f"Currículo {file.name} processado com sucesso!")

def semantic_search(query):
    results = collection.query(query_texts=[query], n_results=5)
    return results


def query_gemini(system_prompt, user_prompt):
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt
    )
    response = model.generate_content(user_prompt)
    return response.text

def generate_response(query, context):
    system_prompt = "Você é um assistente especializado em análise de currículos."
    user_prompt = f"Contexto: {context}\nPergunta: {query}"
    
    response = query_gemini(system_prompt, user_prompt)
    return response
    

st.title("Análise de Currículos")

uploaded_files = st.file_uploader("Faça upload de múltiplos currículos (PDF)", accept_multiple_files=True, type="pdf")

if st.button("Processar Currículos"):
    if uploaded_files:
        with st.spinner("Processando currículos..."):
            for uploaded_file in uploaded_files:
                process_pdf(uploaded_file)
    else:
        st.warning("Por favor, faça o upload de pelo menos um currículo.")

query = st.text_input("Faça uma pergunta sobre os candidatos:")
if query:
    with st.spinner("Buscando informações..."):
        search_results = semantic_search(query)
    
    context = "\n".join(search_results['documents'][0])
    
    if context:
        with st.spinner("Gerando resposta..."):
            response = generate_response(query, context)
        
        st.subheader("Resposta:")
        st.write(response)
    else:
        st.warning("Nenhum resultado encontrado para a consulta.")
