from PIL import Image
from ultralytics import YOLO

# Carrega o modelo YOLO pré-treinado
model = YOLO("best.pt")

def segmenta_imagem_yolo(image):
    """
    Função para receber uma imagem e segmentá-la usando o modelo YOLO.
    Retorna a primeira imagem segmentada sem salvá-la.
    :param image: Imagem de entrada (PIL.Image).
    :return: A primeira imagem segmentada (PIL.Image) ou None se não houver detecção.
    """
    # Converte a imagem PIL para um formato compatível se necessário
    if not isinstance(image, Image.Image):
        return None

    # Faz a predição sem salvar no disco
    results = model.predict(source=image, save=False, verbose=False)

    # Verifica se houve detecções e recortes
    if results and len(results) > 0 and len(results[0].boxes) > 0:
        # Acessa os resultados da primeira predição
        first_result = results[0]

        # Acessa as coordenadas das caixas de detecção (bounding boxes)
        for box in first_result.boxes.xyxy:
            x_min, y_min, x_max, y_max = [int(coord.item()) for coord in box]

            # Recorta a primeira detecção da imagem
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            
            return cropped_image  # Retorna a primeira imagem recortada

    return None  # Retorna None caso não tenha detecções
