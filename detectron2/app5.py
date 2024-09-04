# %%
import json
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import cv2
import numpy as np
import os
from PIL import Image
import openai

# Configure sua chave API do OpenAI
openai.api_key = "sk-vki3PWWrY7-XZ12z_bnRBuDjBD2CWNsrHPZWyUNQRaT3BlbkFJhr0pKqV5AVix6FF3wqnnLJBg86TkzSVL8GDqIJyY0A"

# Função para carregar o arquivo JSON
def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# Função para mapear as categorias e imagens
def parse_json(data):
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    return categories

# Caminho para o arquivo JSON
json_path = 'train_annotations.coco.json'
data = load_json(json_path)

# Extraindo categorias
categories = parse_json(data)

# Registra as instâncias COCO
register_coco_instances("train_dataset", {}, json_path, "train")

# Defina manualmente as classes com base no JSON carregado
thing_classes = [categories[i] for i in sorted(categories.keys())]
MetadataCatalog.get("train_dataset").thing_classes = thing_classes

# Configurações do modelo
cfg = get_cfg()
config_path = os.path.join(os.getcwd(), "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.merge_from_file(config_path)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # Definindo um threshold padrão

# Carregar manualmente os pesos do modelo
model = build_model(cfg)
weights_path = os.path.join(os.getcwd(), "model_final.pth")
checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model"])

# Coloque o modelo no modo de inferência
model.eval()

# Criar o preditor manualmente após todas as configurações
predictor = DefaultPredictor(cfg)
predictor.model = model

# Verificar os metadados do conjunto de dados de treino
metadata = MetadataCatalog.get("train_dataset")
print(f"Classes detectadas: {metadata.thing_classes}")

# Carregar a imagem localmente para teste
image_path = 'henrique_rezer_mosquer_da_silva.png'
image = Image.open(image_path)
image_np = np.array(image)

# Converter para formato que o OpenCV entende (BGR)
image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# Fazer predições
outputs = predictor(image_cv2)

# Captura das inferências e confidências
instances = outputs["instances"]
pred_classes = instances.pred_classes.tolist()
scores = instances.scores.tolist()

# Mapeando as classes preditas para os nomes das categorias
predicted_labels = [metadata.thing_classes[i] for i in pred_classes]

# Preparar a mensagem para o GPT com os resultados da inferência
message_content = f"As classes preditas são: {predicted_labels} com probabilidades correspondentes: {scores}. O que você pode inferir sobre essa imagem?"

# Chamando a API do GPT para gerar o diagnóstico
response = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Você é um especialista em radiologia odontológica."},
        {"role": "user", "content": message_content}
    ],
    max_tokens=300
)

# Extrair o texto gerado pelo GPT
diagnostic_text = response.choices[0].message["content"].strip()

# Exibir o pré-diagnóstico
print("Pré-Diagnóstico Gerado pelo GPT:")
print(diagnostic_text)

# Visualizar as predições usando os metadados corretos
v = Visualizer(image_cv2[:, :, ::-1], metadata)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Converter a imagem resultante para exibição no OpenCV (RGB)
result_image = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)

# Exibir a imagem original e a imagem com as predições
cv2.imshow("Imagem Original", image_cv2)
cv2.imshow("Resultado da Detecção", result_image)

# Salvar a imagem processada
cv2.imwrite("output_predictions.jpg", out.get_image()[:, :, ::-1])

cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
