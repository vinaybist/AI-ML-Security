from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.vgg16(pretrained=True).to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

try:
    with open('imagenet_classes.txt', 'r') as f:
        IMAGENET_CLASSES = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("Warning: imagenet_classes.txt not found.")

try:
    with open('gtsrb_classes.txt', 'r') as f:
        GTSRB_CLASSES = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("Warning: gtsrb_classes.txt not found.")

# Model management
MODELS = {}

def get_model(model_name):
    if model_name not in MODELS:
        if model_name == 'vgg16':
            print("Loading VGG16...")
            model = models.vgg16(pretrained=True).to(device)
            model.eval()
            MODELS['vgg16'] = model
        elif model_name == 'gtsrb':
            print("Loading GTSRB (ResNet18)...")
            # ResNet18 for 43 traffic sign classes
            # NOTE: This uses random weights as no pre-trained GTSRB weights are available in torchvision
            model = models.resnet18(pretrained=False, num_classes=43).to(device)
            model.eval()
            MODELS['gtsrb'] = model
        else:
            raise ValueError(f"Unknown model: {model_name}")
    return MODELS[model_name]

def get_classes(model_name):
    if model_name == 'vgg16':
        return IMAGENET_CLASSES
    elif model_name == 'gtsrb':
        return GTSRB_CLASSES
    return []

# Initialize default model
get_model('vgg16')

# Check for custom weights
GTSRB_WEIGHTS_PATH = 'gtsrb_model.pth'
if os.path.exists(GTSRB_WEIGHTS_PATH):
    try:
        MODELS['gtsrb'] = get_model('gtsrb') # Initialize structure
        MODELS['gtsrb'].load_state_dict(torch.load(GTSRB_WEIGHTS_PATH, map_location=device))
        print(f"Loaded custom GTSRB weights from {GTSRB_WEIGHTS_PATH}")
    except Exception as e:
        print(f"Failed to load GTSRB weights: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img_io = io.BytesIO()
    img.save(img_io, 'PNG', quality=95)
    img_io.seek(0)
    return base64.b64encode(img_io.getvalue()).decode()


def classify_image(image_tensor, model_name='vgg16'):
    """Classify image and return predictions"""
    model = get_model(model_name)
    classes = get_classes(model_name)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top5_probs, top5_indices = torch.topk(probabilities, min(5, len(classes)))
    
    results = []
    for i in range(len(top5_indices[0])):
        class_idx = top5_indices[0][i].item()
        prob = top5_probs[0][i].item()
        
        if class_idx < len(classes):
            class_name = classes[class_idx]
        else:
            class_name = f'Class {class_idx}'
            
        results.append({
            'class': class_name,
            'probability': round(prob * 100, 2),
            'index': class_idx
        })
    
    return results


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    """Classify uploaded image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    model_name = request.form.get('model', 'vgg16')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Load and preprocess image
        image = Image.open(file.stream).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Classify
        predictions = classify_image(image_tensor, model_name)
        
        # Convert to base64 for display
        img_array = np.array(image) / 255.0
        img_base64 = image_to_base64(img_array)
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}',
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate-adversarial', methods=['POST'])
def generate_adversarial():
    """Generate adversarial example using FGSM: x_adv = x + ε * sign(∇xJ(θ,x,y))"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    epsilon = float(request.form.get('epsilon', 0.03))  # Perturbation magnitude
    model_name = request.form.get('model', 'vgg16')
    file = request.files['file']
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Load model
        model = get_model(model_name)
        
        # Load image
        image = Image.open(file.stream).convert('RGB')
        img_array = np.array(image) / 255.0
        
        # Preprocess
        image_tensor = transform(image).unsqueeze(0).to(device)
        image_tensor.requires_grad = True
        
        # Forward pass
        outputs = model(image_tensor)
        
        # Get the predicted class
        _, predicted = torch.max(outputs, 1)
        target_class = predicted.item()
        
        # Compute loss (maximize loss to move away from predicted class)
        loss = F.cross_entropy(outputs, torch.tensor([target_class]).to(device))
        
        # Compute gradients: ∇xJ(θ,x,y)
        model.zero_grad()
        loss.backward()
        
        # Get the sign of gradients
        gradient = image_tensor.grad.data
        sign_gradient = torch.sign(gradient)
        
        # Apply FGSM: x_adv = x + ε * sign(∇xJ(θ,x,y))
        adversarial_tensor = image_tensor.detach() + epsilon * sign_gradient
        adversarial_tensor = torch.clamp(adversarial_tensor, 0, 1)
        
        # Extract perturbation
        perturbation = (adversarial_tensor - image_tensor.detach()).squeeze(0).cpu().numpy()
        perturbation = np.transpose(perturbation, (1, 2, 0))
        # Normalize to [0, 1] for visualization (min-max scaling)
        perturbation_normalized = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)
        
        # Convert adversarial to image
        adversarial_np = adversarial_tensor.squeeze(0).detach().cpu().numpy()
        adversarial_np = np.transpose(adversarial_np, (1, 2, 0))
        adversarial_np = np.clip(adversarial_np, 0, 1)
        
        # Classify original and adversarial
        # Note: We pass model_name to classify_image to ensure it uses the correct model/classes
        with torch.no_grad():
            orig_preds = classify_image(transform(image).unsqueeze(0).to(device), model_name)
            adv_preds = classify_image(adversarial_tensor, model_name)
        
        # Convert to base64
        orig_base64 = image_to_base64(img_array)
        adv_base64 = image_to_base64(adversarial_np)
        pert_base64 = image_to_base64(perturbation_normalized)
        
        return jsonify({
            'success': True,
            'original_image': f'data:image/png;base64,{orig_base64}',
            'adversarial_image': f'data:image/png;base64,{adv_base64}',
            'perturbation': f'data:image/png;base64,{pert_base64}',
            'original_predictions': orig_preds,
            'adversarial_predictions': adv_preds,
            'epsilon': epsilon
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download-adversarial', methods=['POST'])
def download_adversarial():
    """Download adversarial image"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        epsilon = float(request.form.get('epsilon', 0.03))
        model_name = request.form.get('model', 'vgg16')
        file = request.files['file']
        
        # Load model
        model = get_model(model_name)
        
        # Load image
        image = Image.open(file.stream).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        image_tensor.requires_grad = True
        
        # Generate adversarial
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        loss = F.cross_entropy(outputs, torch.tensor([predicted.item()]).to(device))
        
        model.zero_grad()
        loss.backward()
        
        gradient = image_tensor.grad.data
        sign_gradient = torch.sign(gradient)
        adversarial_tensor = image_tensor.detach() + epsilon * sign_gradient
        adversarial_tensor = torch.clamp(adversarial_tensor, 0, 1)
        
        # Convert to PIL Image
        adversarial_np = adversarial_tensor.squeeze(0).detach().cpu().numpy()
        adversarial_np = np.transpose(adversarial_np, (1, 2, 0))
        adversarial_np = np.clip(adversarial_np, 0, 1)
        adversarial_img = Image.fromarray((adversarial_np * 255).astype(np.uint8))
        
        # Save to bytes
        img_io = io.BytesIO()
        adversarial_img.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png', as_attachment=True, 
                        download_name='adversarial_example.png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
