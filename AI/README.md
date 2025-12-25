# 🛡️ AI Hacking: FGSM Adversarial Attack Demo

This project demonstrates how to generate **Adversarial Examples** using the **Fast Gradient Sign Method (FGSM)**. It uses a pre-trained **VGG-16** neural network to classify images and then attacks it by adding imperceptible noise that tricks the model into making incorrect predictions.

## 📂 Project Structure

- **`app.py`**: The Flask backend that handles image processing, model inference, and adversarial generation.
- **`templates/index.html`**: The main user interface.
- **`static/script.js`**: Handles user interactions, file uploads, and real-time updates.
- **`imagenet_classes.txt`**: A list of 1000 ImageNet class labels used for interpretable predictions.

---

## 💻 Code Explanation

### 1. Backend (`app.py`)

The backend is built with **Flask** and **PyTorch**.

#### **Model Setup**
We load a pre-trained **VGG-16** model from `torchvision`. This model has been trained on the ImageNet dataset.
```python
model = models.vgg16(pretrained=True).to(device)
model.eval()  # Set to evaluation mode (important for batchnorm/dropout)
```

#### **Image Preprocessing**
Images are resized to `224x224`, converted to tensors, and normalized using standard ImageNet mean and standard deviation.
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

#### **Endpoint: `/classify`**
1. Receives an uploaded image.
2. Preprocesses it into a tensor.
3. Passes it through VGG-16 to get the top 5 predicted classes.
4. Returns the results to the frontend.

#### **Endpoint: `/generate-adversarial` (The Core Logic)**
This simulates the attack. It implements the **FGSM** formula:
$$
\text{adversarial\_image} = \text{image} + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))
$$

**Steps:**
1. **Forward Pass**: The image is fed into the model to get a prediction.
2. **Calculate Loss**: We compute the loss between the prediction and the *current* class.
3. **Backward Pass**: We calculate the gradients of the loss with respect to the **input image** (not the model weights!).
    ```python
    loss.backward()
    gradient = image_tensor.grad.data
    ```
4. **Create Perturbation**: We take the `sign()` of the gradient to determine the direction to change each pixel to *maximize* the loss.
    ```python
    sign_gradient = torch.sign(gradient)
    ```
5. **Apply Attack**: We add this noise multiplied by `epsilon` (the attack strength) to the original image.
    ```python
    adversarial_tensor = image_tensor.detach() + epsilon * sign_gradient
    ```

### 2. Frontend (`index.html` & `script.js`)

#### **Interface**
The UI has two main tabs:
- **Classify**: Simple inference to see what the model thinks.
- **Generate Adversarial**: The attack interface with the Epsilon slider.

#### **Real-time Updates**
We use **Debouncing** to make the epsilon slider responsive. As you drag the slider:
1. The displayed value updates immediately.
2. The `generateAdversarial` API call is delayed by 300ms so we don't crash the server with requests for every pixel of movement.

```javascript
// From script.js
const debouncedGenerate = debounce(() => {
    // ... call API ...
}, 300);

epsilonSlider.addEventListener('input', function(e) {
    updateEpsilonValue(e.target.value);
    debouncedGenerate(); // Triggers the safe, delayed call
});
```

### 3. Data (`imagenet_classes.txt`)
The app reads this text file at startup to map the numeric model outputs (0-999) to human-readable names (e.g., "Goldfish", "Tabby Cat").

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the Server**:
   ```bash
   python app.py
   ```
3. **Open Browser**:
   Visit `http://127.0.0.1:5000`

---
*Created for educational purposes to demonstrate AI safety concepts.*
