import base64
import io
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import ViTConfig, ViTModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BATCH_SIZE = 64
N_HIDDEN_LAYERS = 16
MAX_LENGTH = 16
EVAL_INTERVAL = 10
LEARNING_RATE = 9e-4
EPOCHS = 6
N_EMBD = 128
N_HEAD = 8
N_LAYER = 8
DROPOUT = 0.4
IMG_SIZE = 640
PATCH_SIZE = 16
IMAGE_EMBED_DIM = 512
N_CHANNELS = 3
MAX_POSITION_EMBEDDINGS = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = LlamaTokenizer.from_pretrained('NousResearch/Llama-2-7b-chat-hf')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "trainCountries")

