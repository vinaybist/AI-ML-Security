/* Tab Switching */
function switchTab(tabName) {
    // Hide all tabs
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.classList.remove('active'));

    // Remove active state from all buttons
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(btn => btn.classList.remove('active'));

    // Show selected tab
    document.getElementById(tabName).classList.add('active');

    // Add active state to clicked button
    event.target.classList.add('active');
}

/* Update Epsilon Display */
function updateEpsilonValue(value) {
    document.getElementById('epsilonValue').textContent = parseFloat(value).toFixed(2);
}

/* Classify Image Upload Handler */
function handleClassifyUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    classifyImage(file);
}

/* Adversarial Upload Handler */
function handleAdversarialUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    generateAdversarial(file);
}

/* Classify Image */
async function classifyImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', document.getElementById('modelSelect').value);

    showElement('classifyLoading');
    hideElement('classifyResults');
    hideElement('classifyError');

    try {
        const response = await fetch('/classify', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            document.getElementById('classifyImage').src = data.image;
            displayPredictions('classifyPredictions', data.predictions);
            hideElement('classifyLoading');
            showElement('classifyResults');
        } else {
            showError('classifyError', data.error);
        }
    } catch (error) {
        showError('classifyError', 'Error classifying image: ' + error.message);
    }
}

/* Generate Adversarial Example */
async function generateAdversarial(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('epsilon', document.getElementById('epsilon').value);
    formData.append('model', document.getElementById('modelSelect').value);

    showElement('adversarialLoading');
    hideElement('adversarialResults');
    hideElement('adversarialError');

    try {
        const response = await fetch('/generate-adversarial', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            document.getElementById('originalImage').src = data.original_image;
            document.getElementById('perturbationImage').src = data.perturbation;
            document.getElementById('adversarialImage').src = data.adversarial_image;

            // Display predictions in HTML instead of using displayPredictions
            const originalPredDiv = document.getElementById('originalPredictions');
            originalPredDiv.innerHTML = '';
            data.original_predictions.forEach(pred => {
                const item = createPredictionItem(pred);
                originalPredDiv.appendChild(item);
            });

            const adversarialPredDiv = document.getElementById('adversarialPredictions');
            adversarialPredDiv.innerHTML = '';
            data.adversarial_predictions.forEach(pred => {
                const item = createPredictionItem(pred);
                adversarialPredDiv.appendChild(item);
            });

            hideElement('adversarialLoading');
            showElement('adversarialResults');
        } else {
            showError('adversarialError', data.error);
        }
    } catch (error) {
        showError('adversarialError', 'Error generating adversarial example: ' + error.message);
    }
}

/* Create Prediction Item Element */
function createPredictionItem(pred) {
    const div = document.createElement('div');
    div.className = 'prediction-item';

    const percentage = pred.probability;
    const barFill = percentage / 100;

    div.innerHTML = `
        <div style="flex: 1; text-align: left;">
            <span>${pred.class}</span>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${percentage}%"></div>
            </div>
        </div>
        <span>${percentage.toFixed(1)}%</span>
    `;

    return div;
}

/* Display Predictions */
function displayPredictions(elementId, predictions) {
    const container = document.getElementById(elementId);
    container.innerHTML = '';

    predictions.forEach(pred => {
        const item = createPredictionItem(pred);
        container.appendChild(item);
    });
}

/* Download Adversarial Image */
async function downloadAdversarial() {
    const file = document.getElementById('adversarialInput').files[0];
    if (!file) {
        alert('Please upload an image first');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('epsilon', document.getElementById('epsilon').value);
    formData.append('model', document.getElementById('modelSelect').value);

    try {
        const response = await fetch('/download-adversarial', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'adversarial_example.png';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            alert('Error downloading image');
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

/* Utility Functions */
function showElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.remove('hidden');
    }
}

function hideElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.add('hidden');
    }
}

function showError(elementId, message) {
    const errorDiv = document.getElementById(elementId);
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.classList.add('show');
        hideElement('classifyLoading');
        hideElement('adversarialLoading');
    }
}

/* Drag and Drop Support */
document.addEventListener('DOMContentLoaded', function () {
    setupDragAndDrop('classifyInput');
    setupDragAndDrop('adversarialInput');

    // Epsilon Slider Setup
    const epsilonSlider = document.getElementById('epsilon');
    if (epsilonSlider) {
        // Debounce generation to avoid spamming the server
        const debouncedGenerate = debounce(() => {
            const fileInput = document.getElementById('adversarialInput');
            if (fileInput.files.length > 0) {
                generateAdversarial(fileInput.files[0]);
            }
        }, 300); // 300ms delay

        epsilonSlider.addEventListener('input', function (e) {
            updateEpsilonValue(e.target.value);
            debouncedGenerate();
        });
    }
});

// Debounce Utility
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function setupDragAndDrop(inputId) {
    const input = document.getElementById(inputId);
    const label = input.nextElementSibling;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        label.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        label.addEventListener(eventName, () => {
            label.classList.add('active');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        label.addEventListener(eventName, () => {
            label.classList.remove('active');
        }, false);
    });

    label.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        input.files = files;

        if (inputId === 'classifyInput') {
            handleClassifyUpload({ target: { files: files } });
        } else {
            handleAdversarialUpload({ target: { files: files } });
        }
    }, false);
}
