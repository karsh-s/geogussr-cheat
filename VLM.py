import base64
import io
import os
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
image_dir = os.path.join(BASE_DIR, "trainEurope")

# Fixed function to properly handle file paths
def image_file_to_base64(image_filename):
    # Remove leading slash if present and join with image_dir
    if image_filename.startswith('/'):
        image_filename = image_filename[1:]
    image_path = os.path.join(image_dir, image_filename)
    with open(image_path, 'rb') as img_file:
        b64_str = base64.b64encode(img_file.read()).decode('utf-8')
    return b64_str

# Load the CSV file
csv_path = os.path.join(image_dir, 'inputs.csv')
df = pd.read_csv(csv_path, sep=";").dropna(axis=1, how="all")
df['b64string_images'] = df['file'].apply(image_file_to_base64)
print(f"Loaded {len(df)} images")
print(df.head())

config = ViTConfig(
    image_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    num_channels=N_CHANNELS,
    hidden_size=IMAGE_EMBED_DIM,
    num_attention_heads=N_HEAD,
    num_hidden_layers=N_HIDDEN_LAYERS,
    intermediate_size=4 * IMAGE_EMBED_DIM,
    hidden_dropout_prob=DROPOUT,
    attention_probs_dropout_prob=DROPOUT,
)

testvit = ViTModel(config)
vit_input = torch.zeros(BATCH_SIZE, N_CHANNELS, IMG_SIZE, IMG_SIZE)
testvit_out = testvit(vit_input).last_hidden_state[:, 0]
print(f"ViT output shape: {testvit_out.shape}")

class VisionLanguageModel(nn.Module):
    def __init__(
        self,
        n_embed,
        image_embed_dim,
        vocab_size,
        n_layer,
        n_head,
        img_size,
        patch_size,
        n_hidden_layers,
        dropout,
        pad_token_id,
        max_position_embeddings,
        n_channels,
    ):
        super().__init__()
        vit_config = ViTConfig(
            image_size=img_size,
            patch_size=patch_size,
            num_channels=n_channels,
            hidden_size=image_embed_dim,
            num_attention_heads=n_head,
            num_hidden_layers=n_hidden_layers,
            intermediate_size=4 * image_embed_dim,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.vision_encoder = ViTModel(vit_config)
        self.image_projector = nn.Linear(image_embed_dim, n_embed)

        llama_config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=n_embed,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=int(pad_token_id),
        )
        self.llama = LlamaForCausalLM(llama_config)
        self.llama = self.llama.to(dtype=torch.bfloat16)

    def forward(self, img_array, input_ids, targets=None):
        image_embeds = self.vision_encoder(img_array).last_hidden_state[:, 0]
        image_embeds_proj = self.image_projector(image_embeds).to(dtype=torch.bfloat16)
        image_embeds_proj = image_embeds_proj.unsqueeze(1)

        text_embeds = self.llama.model.embed_tokens(input_ids).to(dtype=torch.bfloat16)

        input_embeds = torch.cat([image_embeds_proj, text_embeds], dim=1)

        attention_mask = torch.ones(input_embeds.shape[:2], dtype=torch.long, device=input_embeds.device)

        if targets is not None:
            targets = torch.cat([torch.full((targets.size(0), 1), -100, dtype=targets.dtype, device=targets.device), targets], dim=1)
            outputs = self.llama(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                labels=targets,
            )
            return outputs.logits, outputs.loss
        else:
            outputs = self.llama(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
            )
            return outputs.logits

    @torch.no_grad()
    def generate(self, img_array, input_ids, max_new_tokens=20):
        image_embeds = self.vision_encoder(img_array).last_hidden_state[:, 0]
        image_embeds_proj = self.image_projector(image_embeds).unsqueeze(1).to(dtype=torch.bfloat16)

        input_embeds = self.llama.model.embed_tokens(input_ids).to(dtype=torch.bfloat16)
        inputs_embeds = torch.cat([image_embeds_proj, input_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
       
        generated = self.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.llama.config.pad_token_id,
            eos_token_id=self.llama.config.eos_token_id,
        )
        return generated

model = VisionLanguageModel(
        N_EMBD,
        IMAGE_EMBED_DIM,
        tokenizer.vocab_size,
        N_LAYER,
        N_HEAD,
        IMG_SIZE,
        PATCH_SIZE,
        N_HIDDEN_LAYERS,
        DROPOUT,
        tokenizer.pad_token_id,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        n_channels=N_CHANNELS,
)
model.to(device)

dummy_img = torch.randn(1, N_CHANNELS, IMG_SIZE, IMG_SIZE).to(device)
dummy_idx = torch.randint(0, tokenizer.vocab_size, (1, MAX_LENGTH)).to(device)
output = model(dummy_img, dummy_idx)
print(f"Model output shape: {output.shape}")

def base64_to_tensor(base64_str, img_size=640):
    image = Image.open(io.BytesIO(base64.b64decode(base64_str)))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

class VLMDataset(Dataset):
    def __init__(self, df, img_size=640, tokenizer=None):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_b64 = self.df.loc[idx, 'b64string_images']
        caption = self.df.loc[idx, 'caption']
        image = base64_to_tensor(img_b64, self.img_size).squeeze(0)
        encoding = self.tokenizer(
            caption,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )
        input_ids = encoding.input_ids.squeeze(0)
        targets = input_ids.clone()
        targets[:-1] = input_ids[1:]
        targets[-1] = self.tokenizer.pad_token_id
        return image, input_ids, targets

# Prepare train/val split - using only caption column
df_work = df[['b64string_images', 'caption']].copy()
n = int(0.9 * len(df_work))
df_train = df_work.iloc[:n]
df_val = df_work.iloc[n:]

print(f"\nDataset split:")
print(f"Training samples: {len(df_train)}")
print(f"Validation samples: {len(df_val)}")

train_dataset = VLMDataset(df_train, img_size=IMG_SIZE, tokenizer=tokenizer)
val_dataset = VLMDataset(df_val, img_size=IMG_SIZE, tokenizer=tokenizer)

print(f"\nSample from dataset:")
sample_img, sample_input, sample_target = train_dataset[0]
print(f"Image shape: {sample_img.shape}")
print(f"Input IDs shape: {sample_input.shape}")
print(f"Target shape: {sample_target.shape}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=True)

@torch.no_grad()
def estimate_loss(model, val_loader):
    losses = []
    model.eval()
    for images, input_ids, targets in val_loader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        _, loss = model(images, input_ids, targets)
        losses.append(loss.item())
    return sum(losses) / len(losses)

def train_model(model, train_loader, val_loader, epochs, learning_rate, eval_interval):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, input_ids, targets) in loop:
            images = images.to(device)
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits, loss = model(images, input_ids, targets)
            loss.backward()
            optimizer.step()
            if batch_idx % eval_interval == 0:
                loop.set_postfix(loss=loss.item())
        val_loss = estimate_loss(model, val_loader)
        print(f"Validation Loss after epoch {epoch+1}: {val_loss}")

print("\n" + "="*50)
print("Starting training...")
print("="*50)
train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, EVAL_INTERVAL)

# Save the model
model_save_path = os.path.join(BASE_DIR, 'vlm_model.pth')
torch.save(model.state_dict(), model_save_path)
print(f"\nModel saved to: {model_save_path}")

# Test inference on a sample image
print("\n" + "="*50)
print("Testing inference...")
print("="*50)

# Use test image from testEurope folder
test_dir = os.path.join(BASE_DIR, 'testEurope')
img_path = os.path.join(test_dir, 'Albania', 'img_0.jpg')  # Add .jpg extension

if os.path.exists(img_path):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    prompt = "A photo of"
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(img_tensor, input_ids, max_new_tokens=30)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("Generated description of the given picture:")
    print(generated_text)
else:
    print(f"Test image not found at: {img_path}")
    print("Please update the path to a valid test image.")