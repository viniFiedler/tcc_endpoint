import os
os.environ["HF_HOME"] = "/tmp"

from flask import Flask, jsonify, request
import uuid
import json
from PIL import Image
import torch
import io
import PaginaInicial
from transformers import ViTFeatureExtractor, ViTModel
from TrataImagems import segmenta_imagem_yolo

app = Flask(__name__)

REGISTROS_FILE = 'registros.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Carregar o modelo e o feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('viniFiedler/vit-base-patch16-224-finetuned-eurosat')
model = ViTModel.from_pretrained('viniFiedler/vit-base-patch16-224-finetuned-eurosat')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_feature_vector(image):
    """
    Extrai o vetor de características de uma imagem usando o modelo ViT.
    """
    image = image.convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    feature_vector = outputs.last_hidden_state[0].mean(dim=0).numpy().tolist()
    return feature_vector

@app.route('/registrar', methods=['POST'])
def registrar():
    """
    Rota para registrar uma imagem segmentada e seu vetor de características.
    """
    if 'file' not in request.files or not request.form.get('nome'):
        return jsonify({"error": "Nenhuma imagem ou nome enviado."}), 400

    file = request.files['file']
    nome = request.form.get('nome')

    if file and allowed_file(file.filename):
        image = Image.open(io.BytesIO(file.read()))

        # Segmenta a imagem com YOLO
        segmentada = segmenta_imagem_yolo(image)

        if segmentada is None:
            return jsonify({"error": "Nenhum cachorro detectado na foto."}), 400

        # Extrair o vetor de características da imagem segmentada
        feature_vector = extract_feature_vector(segmentada)

        # Gerar um ID aleatório e salvar o registro
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
    """
    Rota para identificar o registro mais próximo da imagem segmentada.
    """
    if 'file' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada."}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        image = Image.open(io.BytesIO(file.read()))

        # Segmenta a imagem com YOLO
        segmentada = segmenta_imagem_yolo(image)

        if segmentada is None:
            return jsonify({"error": "Nenhum cachorro detectado na foto."}), 400

        # Extrair o vetor de características da imagem segmentada
        feature_vector = extract_feature_vector(segmentada)

        registros = load_registros()
        if not registros:
            return jsonify({"error": "Nenhum registro encontrado."}), 400

        # Comparar com os registros armazenados
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

@app.route('/verificar', methods=['POST'])
def verificar():
    """
    Rota para verificar se uma imagem corresponde a um registro específico informado por ID.
    """
    if 'file' not in request.files or 'id' not in request.form:
        return jsonify({"error": "Imagem e ID do registro são necessários."}), 400

    file = request.files['file']
    registro_id = request.form.get('id')

    if file and allowed_file(file.filename):
        image = Image.open(io.BytesIO(file.read()))

        # Segmenta a imagem com YOLO
        segmentada = segmenta_imagem_yolo(image)

        if segmentada is None:
            return jsonify({"error": "Nenhum cachorro detectado na foto."}), 400

        # Extrair o vetor de características da imagem segmentada
        feature_vector = extract_feature_vector(segmentada)

        registros = load_registros()
        registro_encontrado = next((r for r in registros if r["id"] == registro_id), None)

        if not registro_encontrado:
            return jsonify({"error": f"Nenhum registro encontrado com o ID: {registro_id}."}), 400

        # Comparar com o vetor de características do registro
        distancia = euclidean_distance(feature_vector, registro_encontrado['feature_vector'])

        return jsonify({
            "message": "Verificação realizada",
            "id": registro_id,
            "distancia": distancia,
            "verificacao": "Correspondente" if distancia < 1.0 else "Não correspondente"
        }), 200
    else:
        return jsonify({"error": "Extensão de arquivo não permitida."}), 400

def load_registros():
    """
    Carrega os registros existentes do arquivo JSON.
    """
    if os.path.exists(REGISTROS_FILE):
        with open(REGISTROS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_registros(registros):
    """
    Salva os registros no arquivo JSON.
    """
    with open(REGISTROS_FILE, 'w') as f:
        json.dump(registros, f, indent=4)

def euclidean_distance(v1, v2):
    """
    Calcula a distância euclidiana entre dois vetores.
    """
    return sum((i - j) ** 2 for i, j in zip(v1, v2)) ** 0.5

@app.route("/")
def hello():
    """
    Página inicial que explica o funcionamento da API.
    """
    return PaginaInicial.render_pagina_inicial()

if __name__ == "__main__":
    app.run()
