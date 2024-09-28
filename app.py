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

app = Flask(__name__)

REGISTROS_FILE = 'registros.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

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

def load_registros():
    if os.path.exists(REGISTROS_FILE):
        with open(REGISTROS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_registros(registros):
    with open(REGISTROS_FILE, 'w') as f:
        json.dump(registros, f, indent=4)

def euclidean_distance(v1, v2):
    return sum((i - j) ** 2 for i, j in zip(v1, v2)) ** 0.5

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run()

