import os
os.environ["HF_HOME"] = "/tmp"

from flask import Flask, jsonify, request
import os
import uuid
import json
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import torch
import io
import PaginaInicial

app = Flask(__name__)

REGISTROS_FILE = 'registros.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Carregar o modelo e o feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('viniFiedler/vit-base-patch16-224-finetuned-eurosat')
model = ViTModel.from_pretrained('viniFiedler/vit-base-patch16-224-finetuned-eurosat')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_feature_vector(image):
    image = image.convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    feature_vector = outputs.last_hidden_state[0].mean(dim=0).numpy().tolist()
    return feature_vector

# Função para carregar registros existentes
def load_registros():
    if os.path.exists(REGISTROS_FILE):
        with open(REGISTROS_FILE, 'r') as f:
            return json.load(f)
    return []

# Função para salvar registros atualizados
def save_registros(registros):
    with open(REGISTROS_FILE, 'w') as f:
        json.dump(registros, f, indent=4)

# Função para calcular a distância euclidiana entre dois vetores
def euclidean_distance(v1, v2):
    return sum((i - j) ** 2 for i, j in zip(v1, v2)) ** 0.5

@app.route('/registrar', methods=['POST'])
def registrar():
    if 'file' not in request.files or not request.form.get('nome'):
        return jsonify({"error": "Nenhuma imagem ou nome enviado."}), 400

    file = request.files['file']
    nome = request.form.get('nome')

    if file and allowed_file(file.filename):
        image = Image.open(io.BytesIO(file.read()))
        feature_vector = extract_feature_vector(image)

        registro_id = str(uuid.uuid4())
        registros = load_registros()
        novo_registro = {"id": registro_id, "nome": nome, "feature_vector": feature_vector}
        registros.append(novo_registro)
        save_registros(registros)

        return jsonify({"message": "Registro salvo!", "id": registro_id, "nome": nome}), 200
    else:
        return jsonify({"error": "Extensão de arquivo não permitida."}), 400

@app.route('/identificar', methods=['POST'])
def identificar():
    if 'file' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada."}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        image = Image.open(io.BytesIO(file.read()))
        feature_vector = extract_feature_vector(image)

        registros = load_registros()
        if not registros:
            return jsonify({"error": "Nenhum registro encontrado."}), 400

        menor_distancia = float('inf')
        registro_mais_proximo = None

        for registro in registros:
            distancia = euclidean_distance(feature_vector, registro['feature_vector'])
            if distancia < menor_distancia:
                menor_distancia = distancia
                registro_mais_proximo = registro

        return jsonify({"message": "Registro mais próximo encontrado", "registro": registro_mais_proximo}), 200
    else:
        return jsonify({"error": "Extensão de arquivo não permitida."}), 400

# Novo endpoint para verificar uma imagem com base no ID
@app.route('/verificar', methods=['GET', 'POST'])
def verificar():
    if request.method == 'POST':
        if 'file' not in request.files or 'id' not in request.form:
            return jsonify({"error": "Imagem ou ID não fornecidos."}), 400
        
        file = request.files['file']
        registro_id = request.form.get('id')

        # Verifique se a imagem foi enviada e se tem uma extensão permitida
        if file and allowed_file(file.filename):
            # Extraia o vetor de características da imagem enviada
            image = Image.open(io.BytesIO(file.read()))
            feature_vector = extract_feature_vector(image)

            # Carregue os registros existentes
            registros = load_registros()

            # Procure o registro pelo ID fornecido
            registro = next((r for r in registros if r["id"] == registro_id), None)

            if registro:
                # Compare o vetor extraído com o vetor do registro pelo ID
                distancia = euclidean_distance(feature_vector, registro['feature_vector'])

                # Definir um limite de distância para considerar uma correspondência (opcional)
                if distancia < 500:  # ajuste esse valor conforme necessário
                    return jsonify({
                        "message": "Imagem corresponde ao ID fornecido.",
                        "distancia": distancia,
                        "registro": registro
                    }), 200
                else:
                    return jsonify({
                        "message": "A imagem não corresponde ao ID fornecido.",
                        "distancia": distancia
                    }), 200
            else:
                return jsonify({"error": "ID não encontrado."}), 404
        else:
            return jsonify({"error": "Extensão de arquivo não permitida."}), 400
    else:
        return jsonify({"error": "Método não permitido, use POST."}), 405

@app.route("/")
def hello():
    return PaginaInicial.render_pagina_inicial()

if __name__ == "__main__":
    app.run()
