import mss
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import tkinter as tk
import numpy as np
import threading
import time

# --- CONFIGURATION ---
MODEL_PATH = 'cnn_model_best.pth'
ALPHA = 0.3  # Smoothing factor
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- MODEL LOADING ---
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
country_to_idx = checkpoint['country_to_idx']
idx_to_country = checkpoint['idx_to_country']
num_classes = len(country_to_idx)

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- GLOBAL STATE & LOCKS ---
# We use a lock to prevent the AI thread and UI thread from touching the array at the same time
data_lock = threading.Lock()
running_probabilities = np.zeros(num_classes)

class OverlayHUD:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GeoGuessr AI")
        
        # UI Styling
        self.root.geometry("250x80+1600+50") 
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.8)
        self.root.configure(bg='black')

        self.label_country = tk.Label(self.root, text="Waiting...", font=("Arial", 16, "bold"), fg="white", bg="black")
        self.label_country.pack(pady=(10, 0))
        
        self.label_acc = tk.Label(self.root, text="Press Space to Reset", font=("Arial", 10), fg="#888888", bg="black")
        self.label_acc.pack()

        # BIND SPACE KEY
        # Note: This works when the HUD window has "focus". 
        # If you want it to work globally (even when clicking the game), 
        # you would need the 'pynput' or 'keyboard' library.
        self.root.bind('<space>', self.reset_logic)
        
        # Focus the window so it catches keys initially
        self.root.focus_force()

    def reset_logic(self, event=None):
        global running_probabilities
        with data_lock:
            running_probabilities = np.zeros(num_classes)
        self.label_country.config(text="RESETTING...", fg="yellow")
        print("Memory Reset!")

    def update_text(self, country, confidence):
        self.label_country.config(text=country.upper(), fg="white")
        self.label_acc.config(text=f"Confidence: {confidence:.1f}%", fg="#00FF00")

def inference_loop(hud):
    global running_probabilities
    
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        region = {
            "top": (monitor["height"] - 640) // 2,
            "left": (monitor["width"] - 640) // 2,
            "width": 640,
            "height": 640
        }

        while True:
            sct_img = sct.grab(region)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            
            img_t = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = model(img_t)
                probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]

            with data_lock:
                if np.sum(running_probabilities) == 0:
                    running_probabilities = probs
                else:
                    running_probabilities = (ALPHA * probs) + ((1 - ALPHA) * running_probabilities)

                top_idx = np.argmax(running_probabilities)
                top_country = idx_to_country[top_idx].replace('_', ' ')
                confidence = running_probabilities[top_idx] * 100

            hud.update_text(top_country, confidence)
            time.sleep(0.1)

# --- MAIN ---
if __name__ == "__main__":
    hud = OverlayHUD()
    threading.Thread(target=inference_loop, args=(hud,), daemon=True).start()
    hud.root.mainloop()
