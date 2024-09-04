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
import openai  # Para integração com GPT
import base64  # Para codificar a imagem

# Configure sua chave API do OpenAI
openai.api_key = "YOUR-KEY"

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

# Botões para selecionar o threshold de confiança
st.subheader("Selecione o nível de criticidade:")
col1, col2, col3, col4 = st.columns(4)

# Valor padrão inicial
score_thresh = 0.675  # Intermediário como padrão

with col1:
    if st.button("Não Crítico"):
        score_thresh = 0.225

with col2:
    if st.button("Pouco Crítico"):
        score_thresh = 0.45

with col3:
    if st.button("Intermediário"):
        score_thresh = 0.6

with col4:
    if st.button("Muito Crítico"):
        score_thresh = 0.8

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
    
    # Captura das inferências e confidências
    instances = outputs["instances"]
    pred_classes = instances.pred_classes.tolist()
    scores = instances.scores.tolist()
    
    # Mapeando as classes preditas para os nomes das categorias
    predicted_labels = [metadata.thing_classes[i] for i in pred_classes]
    
    # Codificar a imagem em base64
    buffered = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    buffered = buffered.convert("RGB")
    buffered.save("image_to_send.jpg", format="JPEG")
    with open("image_to_send.jpg", "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    # Preparando a mensagem para o GPT
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Dada a imagem de raio-X odontológico, o que você vê nela?"},
                {"type": "image_base64", "image_base64": image_base64},
            ],
        }
    ]

    # Chamando a API do GPT para gerar o diagnóstico
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300,
    )
    
    # Extrair o texto gerado
    diagnostic_text = response.choices[0].message["content"].strip()
    
    # Exibir o pré-diagnóstico no Streamlit
    st.subheader("Pré-Diagnóstico Gerado pelo GPT:")
    st.write(diagnostic_text)

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
