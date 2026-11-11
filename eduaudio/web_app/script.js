// --- CONFIGURATION ---
// The exact names the AI learned during Python training
const RAW_LABELS = ['cat', 'cow', 'crow', 'dog', 'frog', 'hen', 'pig', 'rooster', 'sheep'];

// A map to turn boring text names into fun emojis for the kids
const EMOJI_MAP = {
    'cat': '🐱 Cat', 'cow': '🐮 Cow', 'crow': '🐦 Crow',
    'dog': '🐶 Dog', 'frog': '🐸 Frog', 'hen': '🐔 Hen',
    'pig': '🐷 Pig', 'rooster': '🐓 Rooster', 'sheep': '🐑 Sheep'
};

// Technical settings that must match the Python training code
const MODEL_PATH = './model.tflite'; // Path to your AI model
const EXPECTED_TIMESTEPS = 130;      // The AI expects a 3-second clip (130 slices)
const MFCC_FEATURES = 40;            // The level of detail (High Definition)
const TARGET_SAMPLE_RATE = 22050;    // The audio speed the AI expects

// --- UI ELEMENTS ---
// Grab the buttons and text boxes from the HTML so we can control them
const statusPill = document.getElementById('status_pill');
const recordBtn = document.getElementById('recordBtn');
const predText = document.getElementById('predictionText');
const confFill = document.getElementById('confidence_fill');
const resultCard = document.getElementById('result_card');
const placeholderMsg = document.getElementById('placeholder_msg');
const feedbackSec = document.getElementById('feedback_section');
const correctionUI = document.getElementById('correction_ui');
const visualizer = document.getElementById('visualizer');
const canvasCtx = visualizer.getContext('2d');
const audioPlayer = document.getElementById('player');

// --- STATE VARIABLES ---
// Temporary storage for the AI, the microphone connection, and sound history
let tfliteModel, audioCtx, meyda, mfccHistory = [], elementSource = null, referenceAudio = null;

// --- INITIALIZATION ---
// This function "wakes up" the AI when first open the website
async function init() {
    try {
        // Tell the browser where to find the TFLite support files
        tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/');
        // Load the AI model file into the browser's memory
        tfliteModel = await tflite.loadTFLiteModel(MODEL_PATH);
        // Show a green "Ready" badge
        statusPill.className = "badge ready";
        statusPill.innerHTML = '<i class="fas fa-check-circle"></i> I am ready!';
        recordBtn.disabled = false; // Enable the microphone button
    } catch (e) {
        // Show a red error badge if something goes wrong
        statusPill.className = "badge error";
        statusPill.innerHTML = "❌ AI Error";
    }
}
init();

// Start the browser's audio engine (AudioContext)
function getAudioContext() {
    if (!audioCtx) {
        // Force the audio speed to 22050Hz to match the AI training
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 22050 });
    }
    return audioCtx;
}

// --- ANALYZER SETUP ---
// Configures the "Meyda" tool to turn sound waves into numbers
function setupAnalyzer(sourceNode) {
    mfccHistory = []; // Clear any old recording data
    if (meyda) meyda.stop(); // Stop any previous analysis

    meyda = Meyda.createMeydaAnalyzer({
        "audioContext": sourceNode.context,
        "source": sourceNode,
        "bufferSize": 512,
        "featureExtractors": ["mfcc", "rms"], // Get sound texture (MFCC) and volume (RMS)
        "numberOfMFCCCoefficients": MFCC_FEATURES, // 40 features
        "callback": (features) => {
            // Drop the 0th piece of data (volume) to match Python's 1-39 feature logic
            const rawMfcc = Array.from(features.mfcc);
            const correctedMfcc = rawMfcc.slice(1);

            // Save the sound slices until have a full 3-second clip
            if (mfccHistory.length < EXPECTED_TIMESTEPS) {
                mfccHistory.push({ mfcc: correctedMfcc, rms: features.rms });
            }
            // Draw the bouncy green bars on the screen
            drawWaveform(features.rms);
        }
    });
    meyda.start();
}

// Visualizer: Paints the green volume bars on the black canvas box
function drawWaveform(rms) {
    const w = visualizer.width, h = visualizer.height;
    const img = canvasCtx.getImageData(1, 0, w - 1, h); // Capture existing bars
    canvasCtx.putImageData(img, 0, 0); // Shift them to the left
    canvasCtx.clearRect(w - 1, 0, 1, h); // Clear space for the new bar

    const bar = Math.min(rms * 1000, h); // Calculate bar height based on loudness
    canvasCtx.fillStyle = bar > 10 ? '#66BB6A' : '#CFD8DC'; // Green if loud, Grey if quiet
    canvasCtx.fillRect(w - 2, (h - bar) / 2, 2, bar); // Draw the bar in the center
}

// --- PRE-PROCESSING ---
// Standardize: Scales numbers so the AI isn't confused by quiet or loud mics
function standardize(flatArray) {
    const mean = flatArray.reduce((a, b) => a + b) / flatArray.length;
    const std = Math.sqrt(flatArray.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / flatArray.length);
    return flatArray.map(x => (x - mean) / (std + 0.000001));
}

// TrimSilence: Cuts out "dead air" from the start and end of a recording
function trimSilence(history) {
    if (history.length === 0) return [];
    const maxVol = Math.max(...history.map(h => h.rms));
    if (maxVol < 0.001) return history.map(h => h.mfcc); // If all silent, return all
    const threshold = maxVol * 0.15; // Set volume threshold to 15%
    let start = 0, end = history.length - 1;

    // Find where the sound actually begins and ends
    for (let i = 0; i < history.length; i++) if (history[i].rms > threshold) { start = i; break; }
    for (let i = history.length - 1; i >= 0; i--) if (history[i].rms > threshold) { end = i; break; }

    start = Math.max(0, start - 2);
    end = Math.min(history.length - 1, end + 2);
    return (end <= start) ? history.map(h => h.mfcc) : history.slice(start, end + 1).map(h => h.mfcc);
}

// --- PREDICTION LOGIC ---
// This is the main function where the AI thinks and reveals the answer
async function predict() {
    statusPill.innerHTML = '<i class="fas fa-magic fa-spin"></i> Thinking...';
    feedbackSec.classList.add('hidden'); // Hide feedback loop
    correctionUI.classList.add('hidden'); // Hide correction menu

    if (mfccHistory.length < 5) return; // Stop if the recording is too short

    // Prepare data: Trim dead air first
    const activeAudio = trimSilence(mfccHistory);
    const TARGET_COLS = MFCC_FEATURES - 1; // 39 features

    // Standardize only the active sound
    const flatActive = activeAudio.flat();
    const standardizedActive = standardize(flatActive);

    // Build the slices (frames) for padding
    let processedFrames = [];
    for (let i = 0; i < standardizedActive.length; i += TARGET_COLS) {
        processedFrames.push(standardizedActive.slice(i, i + TARGET_COLS));
    }

    //Pad with Zeros: Add empty space so every clip is exactly 3 seconds
    while (processedFrames.length < EXPECTED_TIMESTEPS) {
        processedFrames.push(new Array(TARGET_COLS).fill(0));
    }

    // Convert to Tensor: The specific math format the AI understands
    const inputFlat = processedFrames.slice(0, EXPECTED_TIMESTEPS).flat();
    const inputTensor = tf.tensor(inputFlat, [1, EXPECTED_TIMESTEPS, TARGET_COLS, 1], 'float32');

    try {
        // RUN THE AI!
        const output = tfliteModel.predict(inputTensor);
        const data = await output.data();
        inputTensor.dispose(); output.dispose(); // Clean up memory to keep it fast

        // Find the winning animal
        const maxProb = Math.max(...data);
        const maxIndex = data.indexOf(maxProb);

        placeholderMsg.classList.add('hidden'); // Hide the "Waiting" icon
        resultCard.classList.remove('hidden'); // Show the results card

        if (maxProb > 0.25) {
            const rawName = RAW_LABELS[maxIndex];
            // Display the emoji and fun name
            predText.innerText = EMOJI_MAP[rawName] || rawName.toUpperCase();
            // Fill the green progress bar based on certainty
            confFill.style.width = (maxProb * 100).toFixed(0) + "%";
            // Show the "Was I right?" buttons after a tiny delay
            setTimeout(() => { feedbackSec.classList.remove('hidden'); }, 400);
        } else {
            // Show this if the AI is confused
            predText.innerText = "🤔 Unsure";
            confFill.style.width = "10%";
        }
    } catch (e) { console.error(e); }
    statusPill.innerHTML = '<i class="fas fa-check-circle"></i> Ready';
}

// --- EVENT HANDLERS ---

// Microphone: Capture sound while the button is held down
recordBtn.addEventListener('mousedown', async () => {
    if (audioCtx) await audioCtx.close();
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 22050 });
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const source = audioCtx.createMediaStreamSource(stream);
        setupAnalyzer(source);
        recordBtn.classList.add('recording');
        statusPill.innerText = "👂 Listening...";
    } catch (e) { console.error(e); }
});

// Microphone: Start thinking when the user lets go of the button
recordBtn.addEventListener('mouseup', () => {
    if (meyda) meyda.stop();
    recordBtn.classList.remove('recording');
    predict();
});

// File Upload: Load a file from the computer and show the player
document.getElementById('fileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    audioPlayer.src = url; // Connect file to the UI player
    audioPlayer.style.display = "block"; // Make the player visible
    audioPlayer.load(); // Force the browser to refresh the player
    statusPill.innerText = "Click Play on the bar!";
    predText.innerText = "---";
    resultCard.classList.add('hidden');
    placeholderMsg.classList.remove('hidden');
});

// Player: Analyze the sound while the file is playing
audioPlayer.addEventListener('play', async () => {
    const ctx = getAudioContext();
    if (ctx.state === 'suspended') await ctx.resume();
    if (!elementSource) {
        elementSource = ctx.createMediaElementSource(audioPlayer);
        elementSource.connect(ctx.destination); // Route sound to speakers
    }
    setupAnalyzer(elementSource);
});

// Player: Start thinking when the file finishes playing
audioPlayer.addEventListener('ended', () => {
    if (meyda) meyda.stop();
    predict();
});

// Footer Library: Play reference sounds when requested
document.getElementById('playBtn').addEventListener('click', () => {
    const txt = document.getElementById('textInput').value.toLowerCase().trim();
    const playIcon = document.getElementById('playIcon');
    if (!RAW_LABELS.includes(txt)) return alert("Not found!");
    const src = `assets/${txt}.wav`;

    // Toggle between Play and Pause icons
    if (referenceAudio && referenceAudio.src.includes(src)) {
        referenceAudio.paused ? referenceAudio.play() : referenceAudio.pause();
        playIcon.className = referenceAudio.paused ? "fas fa-play" : "fas fa-pause";
        return;
    }
    if (referenceAudio) referenceAudio.pause();
    referenceAudio = new Audio(src);
    referenceAudio.play();
    playIcon.className = "fas fa-pause";
    referenceAudio.onended = () => playIcon.className = "fas fa-play";
});

// Simulate Feedback Loop Logic: Simple functions to handle user corrections
function toggleCorrectionUI() { correctionUI.classList.remove('hidden'); }
function sendFeedback(isCorrect) {
    const animal = isCorrect ? "correct" : document.getElementById('correction_select').value;
    alert(`Thanks! Learning that it was ${animal}! 📚`);
    feedbackSec.classList.add('hidden');
    correctionUI.classList.add('hidden');
}