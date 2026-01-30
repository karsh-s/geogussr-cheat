import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model_path = os.path.join(BASE_DIR, 'cnn_model_best.pth')
if not os.path.exists(model_path):
    print("ERROR: cnn_model_best.pth not found! Please run train_cnn.py first.")
    exit(1)

checkpoint = torch.load(model_path, map_location=device)
country_to_idx = checkpoint['country_to_idx']
idx_to_country = checkpoint['idx_to_country']

print(f"Loaded model trained on {len(country_to_idx)} countries")
print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")

# Recreate model architecture
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(country_to_idx))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_country(image_path, top_k=5):
    """
    Predict the country from an image
    
    Args:
        image_path: Path to the image file
        top_k: Number of top predictions to return
    
    Returns:
        List of (country, confidence) tuples
    """
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return None
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = probabilities.topk(top_k, dim=1)
    
    # Convert to readable format
    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        country = idx_to_country[idx.item()]
        # Convert underscore to space for display
        country_display = country.replace('_', ' ')
        confidence = prob.item() * 100
        results.append((country_display, confidence))
    
    return results

def test_image(image_path):
    """Test an image and print results"""
    print(f"\nTesting image: {image_path}")
    print("-" * 60)
    
    results = predict_country(image_path, top_k=5)
    
    if results:
        print(f"\n{'Rank':<6} {'Country':<25} {'Confidence':<10}")
        print("-" * 60)
        for i, (country, confidence) in enumerate(results, 1):
            print(f"{i:<6} {country:<25} {confidence:>6.2f}%")
        
        # Highlight top prediction
        top_country, top_conf = results[0]
        print("\n" + "="*60)
        print(f"🎯 PREDICTION: {top_country} ({top_conf:.2f}% confident)")
        print("="*60)
        
        return top_country
    return None

# Test on sample images
print("\n" + "="*60)
print("Testing CNN Country Classifier")
print("="*60)

# Test image 1
test_dir = os.path.join(BASE_DIR, 'testEurope')
test_img1 = os.path.join(test_dir, 'Finland', 'img_005.jpg')
if os.path.exists(test_img1):
    test_image(test_img1)
else:
    print(f"\nTest image not found: {test_img1}")
    print("Please update the path to a valid test image.")

# You can test more images here
# test_img2 = os.path.join(test_dir, 'France', 'img_5.jpg')
# test_image(test_img2)

# Interactive testing
print("\n" + "="*60)
while True:
    custom_path = input("\nEnter path to test image (or 'quit' to exit): ").strip()
    if custom_path.lower() in ['quit', 'q', 'exit', '']:
        break
    test_image(custom_path)

print("\nTesting complete!")