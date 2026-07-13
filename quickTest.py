import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from transformers import ViTConfig, ViTModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-chat-hf')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# Load saved config
config_path = os.path.join(BASE_DIR, 'vlm_config.pt')
if not os.path.exists(config_path):
    print("ERROR: vlm_config.pt not found! Please run train_vlm.py first.")
    exit(1)

config = torch.load(config_path)
print("Loaded model configuration")

# Define the model class (same as training)
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

# Create model with saved config
model = VisionLanguageModel(
    config['N_EMBD'],
    config['IMAGE_EMBED_DIM'],
    config['vocab_size'],
    config['N_LAYER'],
    config['N_HEAD'],
    config['IMG_SIZE'],
    config['PATCH_SIZE'],
    config['N_HIDDEN_LAYERS'],
    config['DROPOUT'],
    config['pad_token_id'],
    max_position_embeddings=config['MAX_POSITION_EMBEDDINGS'],
    n_channels=config['N_CHANNELS'],
)

# Load trained weights
model_path = os.path.join(BASE_DIR, 'vlm_model.pth')
if not os.path.exists(model_path):
    print("ERROR: vlm_model.pth not found! Please run train_vlm.py first.")
    exit(1)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Loaded trained model weights")

# Function to test an image
def test_image(image_path, prompt="This is a street view from"):
    IMG_SIZE = config['IMG_SIZE']
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return
    
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(img_tensor, input_ids, max_new_tokens=30)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"\nImage: {image_path}")
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    return generated_text

# Test on a sample image
print("\n" + "="*60)
print("Testing model on sample images...")
print("="*60)

# Test image 1
test_dir = os.path.join(BASE_DIR, 'testEurope')
test_img1 = os.path.join(test_dir, 'Germany', 'img_420.jpg')
test_image(test_img1)

# You can add more test images here
# test_img2 = os.path.join(test_dir, 'France', 'img_0.jpg')
# test_image(test_img2)

# Or test with custom path
# custom_path = input("\nEnter path to test image (or press Enter to skip): ")
# if custom_path:
#     test_image(custom_path)

print("\n" + "="*60)
print("Testing complete!")
print("="*60)