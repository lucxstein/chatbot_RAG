from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import traceback
import os

# Inicializar o app Flask
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# Configuração da API OpenAI
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Carregar o índice FAISS
def carregar_indice():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    faiss_index = FAISS.load_local("indice_brasileirao", embeddings, allow_dangerous_deserialization=True)
    return faiss_index

# Inicializar o QA Chain
def criar_qa_chain():
    faiss_index = carregar_indice()
    chat_model = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=faiss_index.as_retriever()
    )
    return qa_chain

qa_chain = criar_qa_chain()

# Endpoint para processar perguntas
@app.route('/', methods=['GET', 'POST'])
def chat():
    try:
        data = request.json
        if not data or 'query' not in data:
            raise ValueError("Campo 'query' ausente ou inválido.")
        
        # Extrair a consulta do cliente
        query = data['query']
        
        # Passar a consulta para o QA Chain
        resposta_qa = qa_chain.run(query)
        
        # Retornar a resposta ao cliente
        resposta = {"resposta": resposta_qa}
        return jsonify(resposta), 200
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print("Erro detalhado:", error_trace)  # Log do traceback completo
        return jsonify({"error": "Erro ao processar a solicitação"}), 500


# Executar o servidor Flask
if __name__ == '__main__':
    # Obtém a porta obrigatoriamente da variável de ambiente
    port = int(os.getenv("PORT"))
    app.run(debug=False, host='0.0.0.0', port=port)

