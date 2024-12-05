from flask import Flask, request, jsonify
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# Inicializar o app Flask
app = Flask(__name__)

# Configuração da API OpenAI
OPENAI_API_KEY = 'OPEN_AI_API_KEY'

# Carregar o índice FAISS
def carregar_indice():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    faiss_index = FAISS.load_local("indice_brasileirao", embeddings)
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
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "A consulta está vazia"}), 400

    try:
        resposta = qa_chain.run(query)
        return jsonify({"resposta": resposta})
    except Exception as e:
        print(f"Erro: {e}")
        return jsonify({"error": "Erro ao processar a solicitação"}), 500

# Executar o servidor Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)
