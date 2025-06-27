let stream = null;
let isProcessing = false;
let isRequesting = false;
let canvas, ctx;
let overlayCanvas, overlayCtx;
let lastFrameTime = 0;
const frameRate = 15;

// Sequence management
let labelSequence = [];
let lastPrediction = '';

const videoElement = document.getElementById('videoElement');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const resetBtn = document.getElementById('resetBtn');
const clearAllBtn = document.getElementById('clearAllBtn');
const removeLastBtn = document.getElementById('removeLastBtn');
const errorMessage = document.getElementById('errorMessage');
const loadingMessage = document.getElementById('loadingMessage');
const predictionOverlay =
    document.getElementById('predictionOverlay');
const sequenceContent = document.getElementById('sequenceContent');
const sequenceCounter = document.getElementById('sequenceCounter');

// Canvas setup
canvas = document.createElement('canvas');
ctx = canvas.getContext('2d');
overlayCanvas = document.getElementById('processedCanvas');
overlayCtx = overlayCanvas.getContext('2d', { willReadFrequently: false });

// MediaPipe hand connections
const HAND_CONNECTIONS = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
    [5, 9],
    [9, 13],
    [13, 17],
];

const constraints = {
    video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user',
    },
};

async function startCamera() {
    try {
        showLoading('Đang khởi tạo camera...');

        stream = await navigator.mediaDevices.getUserMedia(
            constraints
        );
        videoElement.srcObject = stream;

        videoElement.onloadedmetadata = () => {
            hideLoading();
            startBtn.disabled = true;
            stopBtn.disabled = false;
            setupCanvas();
            startProcessing();
        };
    } catch (error) {
        console.error('Error accessing camera:', error);
        showError(
            'Không thể truy cập camera. Vui lòng kiểm tra quyền truy cập.'
        );
        hideLoading();
    }
}

function setupCanvas() {
    // Đảm bảo overlayCanvas luôn đúng tỉ lệ 4:3 và khớp video
    const vw = videoElement.videoWidth;
    const vh = videoElement.videoHeight;
    if (vw && vh) {
        overlayCanvas.width = vw;
        overlayCanvas.height = vh;
        overlayCanvas.style.width = '100%';
        overlayCanvas.style.height = '100%';
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
    }
    videoElement.srcObject = null;
    isProcessing = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    clearCanvas();
}

function clearCanvas() {
    if (overlayCtx) {
        overlayCtx.clearRect(
            0,
            0,
            overlayCanvas.width,
            overlayCanvas.height
        );
    }
}

function startProcessing() {
    isProcessing = true;
    isRequesting = false;
    processFrame();
}

async function processFrame() {
    if (!isProcessing || !videoElement.videoWidth || isRequesting) {
        setTimeout(processFrame, 1000 / frameRate);
        return;
    }
    isRequesting = true;
    try {
        // Resize frame nhỏ trước khi gửi (320x240)
        const targetW = 320;
        const targetH = 240;
        canvas.width = targetW;
        canvas.height = targetH;
        ctx.drawImage(videoElement, 0, 0, targetW, targetH);
        const frameData = canvas.toDataURL('image/jpeg', 0.7);

        const response = await fetch('/process_frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ frame: frameData }),
        });

        const result = await response.json();

        if (result.success) {
            updateUI(result);
            drawLandmarks(result.landmarks);
            if (
                result.prediction &&
                result.prediction !== lastPrediction &&
                result.prediction !== 'non-action'
            ) {
                addLabelToSequence(result.prediction);
                lastPrediction = result.prediction;
            }
        } else {
            console.error('Processing error:', result.error);
        }
    } catch (error) {
        console.error('Frame processing error:', error);
    }
    isRequesting = false;
    setTimeout(processFrame, 1000 / frameRate);
}

function drawLandmarks(landmarksData) {
    // Tối ưu: chỉ clear và vẽ khi có landmarks
    if (!overlayCtx) return;
    if (!landmarksData || landmarksData.length === 0) {
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        return;
    }
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    landmarksData.forEach((handData) => {
        const landmarks = handData.landmarks;
        const isLeftHand = handData.hand_type === 'Left';
        const landmarkColor = isLeftHand ? '#00FF00' : '#FF0000';
        const connectionColor = isLeftHand ? '#00CC00' : '#CC0000';
        // Draw connections
        overlayCtx.save();
        overlayCtx.strokeStyle = connectionColor;
        overlayCtx.lineWidth = 2;
        overlayCtx.beginPath();
        HAND_CONNECTIONS.forEach((connection) => {
            const start = landmarks[connection[0]];
            const end = landmarks[connection[1]];
            if (start && end) {
                const startX = (1 - start.x) * overlayCanvas.width;
                const startY = start.y * overlayCanvas.height;
                const endX = (1 - end.x) * overlayCanvas.width;
                const endY = end.y * overlayCanvas.height;
                overlayCtx.moveTo(startX, startY);
                overlayCtx.lineTo(endX, endY);
            }
        });
        overlayCtx.stroke();
        overlayCtx.restore();
        // Draw landmarks
        overlayCtx.save();
        overlayCtx.fillStyle = landmarkColor;
        for (let i = 0; i < landmarks.length; i++) {
            const landmark = landmarks[i];
            const x = (1 - landmark.x) * overlayCanvas.width;
            const y = landmark.y * overlayCanvas.height;
            overlayCtx.beginPath();
            overlayCtx.arc(x, y, 3, 0, 2 * Math.PI);
            overlayCtx.fill();
        }
        overlayCtx.restore();
    });
}

function addLabelToSequence(label) {
    labelSequence.push(label);
    updateSequenceDisplay();
}

function updateSequenceDisplay() {
    sequenceCounter.textContent = labelSequence.length;

    if (labelSequence.length === 0) {
        sequenceContent.innerHTML = '';
        return;
    }

    const sequenceText = labelSequence.join(' → ');
    sequenceContent.innerHTML = sequenceText;

    // Scroll to bottom
    sequenceContent.scrollTop = sequenceContent.scrollHeight;
}

function clearAllLabels() {
    labelSequence = [];
    lastPrediction = '';
    updateSequenceDisplay();
}

function removeLastLabel() {
    if (labelSequence.length > 0) {
        labelSequence.pop();
        updateSequenceDisplay();
    }
}

function updateUI(result) {
    const prediction = result.prediction || '...';
    predictionOverlay.textContent = prediction;

    const collectingStatus =
        document.getElementById('collectingStatus');
    if (result.collecting) {
        collectingStatus.textContent = 'Đang thu thập';
        collectingStatus.className =
            'status-value status-collecting';
        document
            .getElementById('videoContainer')
            .classList.add('collecting');
    } else {
        collectingStatus.textContent = 'Chờ tín hiệu';
        collectingStatus.className = 'status-value status-idle';
        document
            .getElementById('videoContainer')
            .classList.remove('collecting');
    }
}

async function resetSequence() {
    try {
        await fetch('/reset_sequence', { method: 'POST' });
        predictionOverlay.textContent = '...';
        clearCanvas();
        lastPrediction = '';
    } catch (error) {
        console.error('Reset error:', error);
    }
}

async function checkStatus() {
    try {
        const response = await fetch('/get_status');
        const status = await response.json();

        const modelStatus = document.getElementById('modelStatus');
        if (status.model_loaded) {
            modelStatus.textContent = '✅ Sẵn sàng';
            modelStatus.style.color = '#28a745';
        } else {
            modelStatus.textContent = '❌ Lỗi';
            modelStatus.style.color = '#dc3545';
        }
    } catch (error) {
        console.error('Status check error:', error);
    }
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

function hideError() {
    errorMessage.style.display = 'none';
}

function showLoading(message) {
    loadingMessage.textContent = message;
    loadingMessage.style.display = 'block';
}

function hideLoading() {
    loadingMessage.style.display = 'none';
}

// Handle window resize
window.addEventListener('resize', () => {
    if (videoElement.videoWidth > 0) {
        setupCanvas();
    }
});

// Check model status on page load
checkStatus();

// Handle page visibility change
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        isProcessing = false;
    } else if (stream && videoElement.srcObject) {
        isProcessing = true;
        processFrame();
    }
});

// TỰ ĐỘNG BẬT CAMERA KHI LOAD TRANG
window.addEventListener('DOMContentLoaded', startCamera);

// Thêm event listener cho nút Xóa tất cả và Xóa cuối
clearAllBtn.addEventListener('click', clearAllLabels);
removeLastBtn.addEventListener('click', removeLastLabel);