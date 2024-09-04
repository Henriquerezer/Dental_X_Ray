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
import matplotlib.pyplot as plt
import base64
# %%
