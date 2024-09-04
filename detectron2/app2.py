import streamlit as st
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

# Interface do Streamlit
st.title("Aplicação de Detecção de Instâncias com Detectron2")

# Controle deslizante para ajustar SCORE_THRESH_TEST
score_thresh = st.slider("Ajuste o Threshold de Confiança", 0.0, 1.0, 0.6)  # Valor inicial ajustado para 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

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
st.write(f"Classes detectadas: {metadata.thing_classes}")

uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Carrega a imagem do usuário
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Converter para formato que o OpenCV entende (BGR)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Fazer predições
    outputs = predictor(image_cv2)

    # Visualizar as predições usando os metadados corretos
    v = Visualizer(image_cv2[:, :, ::-1], metadata)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Converter a imagem resultante para exibição no Streamlit (RGB)
    result_image = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    
    # Exibir a imagem original e a imagem com as predições
    st.image(image, caption='Imagem Carregada', use_column_width=True)
    st.image(result_image, caption='Resultado da Detecção', use_column_width=True)

    # Salvar a imagem processada
    cv2.imwrite("output_predictions.jpg", out.get_image()[:, :, ::-1])

    # Botão para baixar a imagem
    st.download_button(
        label="Baixar imagem com predições",
        data=open("output_predictions.jpg", "rb").read(),
        file_name="output_predictions.jpg",
        mime="image/jpg"
    )

    st.success("Processamento concluído!")