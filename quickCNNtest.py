import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'cnn_model_best.pth')
if not os.path.exists(model_path):
    exit(1)

checkpoint = torch.load(model_path, map_location=device)
country_to_idx = checkpoint['country_to_idx']
idx_to_country = checkpoint['idx_to_country']

print(f"Loaded model trained on {len(country_to_idx)} countries")
print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")

model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(country_to_idx))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_country(image_path, top_k=5):
    if not os.path.exists(image_path):
        return None
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = probabilities.topk(top_k, dim=1)
    
    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        country = idx_to_country[idx.item()]

        country_display = country.replace('_', ' ')
        confidence = prob.item() * 100
        results.append((country_display, confidence))
    
    return results

def test_image(image_path):
    
    results = predict_country(image_path, top_k=5)
    
    if results:
        for i, (country, confidence) in enumerate(results, 1):
            print(f"{i:<6} {country:<25} {confidence:>6.2f}%")
        
        top_country, top_conf = results[0]
        print(f"{top_country} ({top_conf:.2f}% confident)")
   
        
        return top_country
    return None

test_dir = os.path.join(BASE_DIR, 'testEurope')
test_img1 = os.path.join(test_dir, 'Finland', 'img_005.jpg')
if os.path.exists(test_img1):
    test_image(test_img1)
else:
    print(f"\nTest image not found: {test_img1}")
    print("Please update the path to a valid test image.")
