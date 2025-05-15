/**
 * Main application logic for Naira Detection webapp
 */
// DOM elements
const captureBtn = document.getElementById('capture-btn');
const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('file-input');
const resultContainer = document.getElementById('result-container');
const closeResultBtn = document.getElementById('close-result-btn');
const resultImage = document.getElementById('result-image');
const denominationText = document.getElementById('denomination-text');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceText = document.getElementById('confidence-text');
const speakResultBtn = document.getElementById('speak-result-btn');
const newScanBtn = document.getElementById('new-scan-btn');
const toast = document.getElementById('toast');
const loadingIndicator = document.getElementById('loading-indicator');
const videoElement = document.getElementById('camera-feed');
const cameraContainer = document.getElementById('camera-container');

// App state
let lastDetectionResult = null;
let currentMode = 'camera'; // 'camera' or 'result'
let mediaStream = null;

// Speech synthesis
const synth = window.speechSynthesis;

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    // Initialize camera
    initCamera();

    // Initialize event listeners
    setupEventListeners();

    // Load saved preferences
    loadPreferences();

    // Show welcome toast
    showToast('Welcome to Naira Note Detector');
});

/**
 * Set up event listeners for all interactive elements
 */
function setupEventListeners() {
    // Capture button
    captureBtn.addEventListener('click', handleCapture);

    // Upload button
    uploadBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Close result button
    closeResultBtn.addEventListener('click', showCameraView);

    // Speak result button
    speakResultBtn.addEventListener('click', speakResult);

    // New scan button
    newScanBtn.addEventListener('click', showCameraView);

    // Add touch gestures for accessibility
    setupTouchGestures();

    // Add keyboard shortcuts
    setupKeyboardShortcuts();
}

/**
 * Initialize camera access
 */
async function initCamera() {
    try {
        // Request camera permissions with rear camera preference
        const constraints = {
            video: {
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };

        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = mediaStream;
        videoElement.play();

        showCameraView();
    } catch (error) {
        console.error('Error accessing camera:', error);
        showToast('Camera access denied. Please enable camera permissions.', 'error');
    }
}

/**
 * Handle image capture from camera
 */
async function handleCapture() {
    showLoading(true);

    try {
        // Create canvas to capture frame from video
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

        // Convert to blob for upload
        const imageBlob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg', 0.9);
        });

        // Process the image
        await processImage(imageBlob);
    } catch (error) {
        console.error('Error capturing image:', error);
        showToast('Failed to capture image. Please try again.', 'error');
        showLoading(false);
    }
}

/**
 * Handle file selection from upload
 * @param {Event} event - File input change event
 */
async function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Check if file is an image
    if (!file.type.startsWith('image/')) {
        showToast('Please select an image file.', 'error');
        return;
    }

    showLoading(true);

    try {
        await processImage(file);
    } catch (error) {
        console.error('Error processing uploaded image:', error);
        showToast('Failed to process image. Please try again.', 'error');
        showLoading(false);
    }

    // Reset file input
    fileInput.value = "";
}

/**
 * Process image and detect naira notes
 * @param {Blob} imageBlob - Image blob to process
 */
async function processImage(imageBlob) {
    try {
        // Create form data for API request
        const formData = new FormData();
        formData.append('file', imageBlob);

        // Call detection API
        const response = await fetch('http://127.0.0.1:8000/api/detection/detect/image', {
            method: 'POST',
            body: formData
        });

        console.log({ response })

        // if (!response.ok) {
        //     throw new Error(`Server returned ${response.status}`);
        // }

        // const result = await response.json();
        // console.log({ result })
        // // Store result and display
        // lastDetectionResult = result;
        // console.log({ result })
        // displayResult(result);
    } catch (error) {
        console.error('Detection failed:', error);
        showToast('Detection failed. Please try again.', 'error');
        showLoading(false);
    }
}

/**
 * Display detection result
 * @param {Object} result - Detection result from API
 */
function displayResult(result) {
    // Update result image
    resultImage.src = URL.createObjectURL(new Blob([result.processedImage], { type: 'image/jpeg' }));

    // Update text information
    if (result.denominations && result.denominations.length > 0) {
        // Sort by confidence
        const sortedDenominations = [...result.denominations].sort((a, b) => b.confidence - a.confidence);
        const topResult = sortedDenominations[0];

        // Update UI
        denominationText.textContent = `${topResult.value} Naira`;
        confidenceText.textContent = `${Math.round(topResult.confidence * 100)}% Confidence`;
        confidenceBar.style.width = `${topResult.confidence * 100}%`;

        // Color code based on confidence
        if (topResult.confidence > 0.85) {
            confidenceBar.className = 'confidence-bar high';
        } else if (topResult.confidence > 0.6) {
            confidenceBar.className = 'confidence-bar medium';
        } else {
            confidenceBar.className = 'confidence-bar low';
        }
    } else {
        denominationText.textContent = 'No Naira note detected';
        confidenceText.textContent = '';
        confidenceBar.style.width = '0%';
    }

    // Switch to result view
    showResultView();
    showLoading(false);

    // Automatically speak result for accessibility
    const shouldAutoSpeak = getPreference('autoSpeak', true);
    if (shouldAutoSpeak) {
        speakResult();
    }

    // Vibrate device for tactile feedback
    if ('vibrate' in navigator) {
        navigator.vibrate(200);
    }
}

/**
 * Speak the detection result using text-to-speech
 */
function speakResult() {
    if (!lastDetectionResult) return;

    // Stop any ongoing speech
    if (synth.speaking) {
        synth.cancel();
    }

    let speechText = '';

    if (lastDetectionResult.denominations && lastDetectionResult.denominations.length > 0) {
        // Sort by confidence
        const sortedDenominations = [...lastDetectionResult.denominations].sort((a, b) => b.confidence - a.confidence);
        const topResult = sortedDenominations[0];

        // Format text for speech
        const confidencePercent = Math.round(topResult.confidence * 100);
        speechText = `Detected ${topResult.value} Naira note with ${confidencePercent} percent confidence.`;

        // Add extra detail for multiple detections
        if (sortedDenominations.length > 1) {
            speechText += ` Also possibly detecting ${sortedDenominations[1].value} Naira.`;
        }
    } else {
        speechText = 'No Naira note detected. Please try again with better lighting or positioning.';
    }

    // Create and play speech
    const utterance = new SpeechSynthesisUtterance(speechText);
    utterance.rate = getPreference('speechRate', 1);
    utterance.pitch = getPreference('speechPitch', 1);
    synth.speak(utterance);
}

/**
 * Show the camera view (scanning mode)
 */
function showCameraView() {
    currentMode = 'camera';
    cameraContainer.style.display = 'block';
    resultContainer.style.display = 'none';

    // Clear previous results
    if (resultImage.src) {
        URL.revokeObjectURL(resultImage.src);
        resultImage.src = '';
    }
}

/**
 * Show the result view
 */
function showResultView() {
    currentMode = 'result';
    cameraContainer.style.display = 'none';
    resultContainer.style.display = 'block';
}

/**
 * Toggle loading indicator
 * @param {boolean} isLoading - Whether to show or hide loading
 */
function showLoading(isLoading) {
    loadingIndicator.style.display = isLoading ? 'flex' : 'none';
}

/**
 * Display toast notification
 * @param {string} message - Message to display
 * @param {string} type - Toast type ('info', 'success', 'error')
 */
function showToast(message, type = 'info') {
    toast.textContent = message;
    toast.className = `toast toast-${type} visible`;

    // Hide toast after 3 seconds
    setTimeout(() => {
        toast.className = toast.className.replace(' visible', '');
    }, 3000);
}

/**
 * Add touch gestures for better accessibility
 */
function setupTouchGestures() {
    // Double tap anywhere on screen to capture
    let lastTap = 0;
    document.addEventListener('touchend', function (event) {
        const currentTime = new Date().getTime();
        const tapLength = currentTime - lastTap;

        if (tapLength < 500 && tapLength > 0 && currentMode === 'camera') {
            handleCapture();
            event.preventDefault();
        }

        lastTap = currentTime;
    });

    // Swipe left to go back to camera from result
    document.addEventListener('touchstart', handleSwipe);
    document.addEventListener('touchmove', handleSwipe);
    document.addEventListener('touchend', handleSwipe);
}

/**
 * Handle swipe gestures
 */
let xDown = null;
let yDown = null;

function handleSwipe(evt) {
    if (currentMode !== 'result') return;

    if (evt.type === 'touchstart') {
        const firstTouch = evt.touches[0];
        xDown = firstTouch.clientX;
        yDown = firstTouch.clientY;
    } else if (evt.type === 'touchend' && xDown && yDown) {
        const xUp = evt.changedTouches[0].clientX;
        const yUp = evt.changedTouches[0].clientY;

        const xDiff = xDown - xUp;
        const yDiff = yDown - yUp;

        if (Math.abs(xDiff) > Math.abs(yDiff) && Math.abs(xDiff) > 50) {
            if (xDiff > 0) {
                // Swiped left, go back to camera
                showCameraView();
            }
        }

        xDown = null;
        yDown = null;
    }
}

/**
 * Setup keyboard shortcuts for accessibility
 */
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (event) => {
        // Space or Enter to capture
        if ((event.code === 'Space' || event.code === 'Enter') && currentMode === 'camera') {
            handleCapture();
            event.preventDefault();
        }

        // Escape to go back to camera
        if (event.code === 'Escape' && currentMode === 'result') {
            showCameraView();
            event.preventDefault();
        }

        // S to speak result
        if (event.code === 'KeyS' && currentMode === 'result') {
            speakResult();
            event.preventDefault();
        }

        // U to upload
        if (event.code === 'KeyU' && currentMode === 'camera') {
            fileInput.click();
            event.preventDefault();
        }
    });
}

/**
 * Get user preference from localStorage
 * @param {string} key - Preference key
 * @param {any} defaultValue - Default value if not found
 * @returns {any} The preference value
 */
function getPreference(key, defaultValue) {
    const prefString = localStorage.getItem('nairaScanPrefs');
    if (!prefString) return defaultValue;

    try {
        const prefs = JSON.parse(prefString);
        return prefs[key] !== undefined ? prefs[key] : defaultValue;
    } catch {
        return defaultValue;
    }
}

/**
 * Save user preference to localStorage
 * @param {string} key - Preference key
 * @param {any} value - Value to save
 */
function savePreference(key, value) {
    const prefString = localStorage.getItem('nairaScanPrefs');
    let prefs = {};

    if (prefString) {
        try {
            prefs = JSON.parse(prefString);
        } catch {
            prefs = {};
        }
    }

    prefs[key] = value;
    localStorage.setItem('nairaScanPrefs', JSON.stringify(prefs));
}

/**
 * Load user preferences
 */
function loadPreferences() {
    // Load speech settings
    const speechRate = getPreference('speechRate', 1);
    const speechPitch = getPreference('speechPitch', 1);
    const autoSpeak = getPreference('autoSpeak', true);

    // Apply any settings to UI if needed
    // (Settings UI would be implemented separately)
}

/**
 * Clean up resources
 */
function cleanup() {
    // Stop camera stream
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }

    // Stop any ongoing speech
    if (synth.speaking) {
        synth.cancel();
    }

    // Revoke any object URLs
    if (resultImage.src) {
        URL.revokeObjectURL(resultImage.src);
    }
}

// Clean up when page unloads
window.addEventListener('beforeunload', cleanup);