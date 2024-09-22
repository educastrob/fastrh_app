import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader


def process_pdf_directory(pdf_directory, collection):
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    print(f"Encontrados {len(pdf_files)} arquivos PDF no diretório.")

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"Processando PDF {i}/{len(pdf_files)}: {pdf_file}")
        pdf_path = os.path.join(pdf_directory, pdf_file)
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

            for j, chunk in enumerate(chunks):
                collection.add(
                    documents=[chunk],
                    metadatas=[{"source": pdf_file, "chunk": j}],
                    ids=[f"{pdf_file}_chunk_{j}"]
                )
            
            print(f"- Documento {pdf_file} processado e armazenado.")
        except Exception as e:
            print(f"Erro ao processar {pdf_file}: {str(e)}")

    print("Processamento de todos os PDFs concluído.")

def interactive_query_loop(collection):
    print("Processamento concluído. Iniciando modo de consulta.")
    print("Digite sua consulta ou 'sair' para encerrar.")

    while True:
        query = input("\nConsulta: ")
        if query.lower() == 'sair':
            break

        results = collection.query(
            query_texts=[query],
            n_results=5
        )

        print("\nResultados:")
        for i, (document, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            print(f"Resultado {i}:")
            print(f"Documento: {metadata['source']}")
            print(f"Trecho: {document[:200]}...")
            print()

def main():
    persist_directory = "./chroma_data"
    pdf_directory = "../Currículos"

    chroma_client = chromadb.PersistentClient(path=persist_directory)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    collection = chroma_client.get_or_create_collection(name="curriculos", embedding_function=embedding_function)
    process_pdf_directory(pdf_directory, collection)
    interactive_query_loop(collection)

if __name__ == "__main__":
    main()