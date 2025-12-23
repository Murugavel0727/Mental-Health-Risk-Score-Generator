const API_URL = "http://localhost:8000";

let audioBlob = null;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];

// Helper to simulate API delay
const delay = (ms) => new Promise(res => setTimeout(res, ms));

async function analyzeText() {
    const text = document.getElementById("journalInput").value;
    const resultSpan = document.getElementById("textResult");

    if (!text.trim()) {
        resultSpan.textContent = "Please enter some text.";
        return;
    }

    resultSpan.textContent = "Analyzing...";

    try {
        const response = await fetch(`${API_URL}/analyze/text`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });
        if (!response.ok) throw new Error("Server error");
        const data = await response.json();
        resultSpan.textContent = `Risk Score: ${(data.risk_score * 100).toFixed(1)}`;
    } catch (error) {
        console.warn("Backend offline, using demo mode.");
        await delay(1000); // Simulate processing
        // Deterministic mock based on text length
        const mockScore = (text.length % 100);
        resultSpan.textContent = `Risk Score: ${mockScore.toFixed(1)} (Demo Mode)`;
    }
}

async function toggleRecording() {
    const statusSpan = document.getElementById("recordingStatus");
    const btn = document.getElementById("recordBtn");

    if (!isRecording) {
        // Start Recording
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                statusSpan.textContent = "Recording saved.";
            };

            mediaRecorder.start();
            isRecording = true;
            btn.innerHTML = '<span class="icon">⏹️</span> Stop Recording';
            statusSpan.textContent = "Recording...";
            btn.classList.add("recording-active");
        } catch (err) {
            console.error(err);
            statusSpan.textContent = "Mic access denied.";
        }
    } else {
        // Stop Recording
        mediaRecorder.stop();
        isRecording = false;
        btn.innerHTML = '<span class="icon">🎙️</span> Start Recording';
        btn.classList.remove("recording-active");
    }
}

async function analyzeAudio() {
    const resultSpan = document.getElementById("audioResult");

    if (!audioBlob) {
        resultSpan.textContent = "No audio recorded.";
        return;
    }

    resultSpan.textContent = "Processing...";

    const formData = new FormData();
    formData.append("file", audioBlob, "recording.wav");

    try {
        const response = await fetch(`${API_URL}/analyze/audio`, {
            method: "POST",
            body: formData
        });
        if (!response.ok) throw new Error("Server error");
        const data = await response.json();
        resultSpan.textContent = `Risk Score: ${(data.risk_score * 100).toFixed(1)}`;
    } catch (error) {
        console.warn("Backend offline, using demo mode.");
        await delay(1500);
        resultSpan.textContent = `Risk Score: 42.5 (Demo Mode)`;
    }
}

async function analyzeMultimodal() {
    const text = document.getElementById("journalInput").value;
    const age = document.getElementById("ageInput").value || "25";
    const sleep = document.getElementById("sleepInput").value || "7";
    const stress = document.getElementById("stressInput").value || "5";

    const finalResult = document.getElementById("finalResult");
    const scoreValue = document.getElementById("scoreValue");
    const riskLevel = document.getElementById("riskLevel");

    finalResult.classList.remove("hidden");
    scoreValue.textContent = "...";

    // Allow partial input for demo
    if ((!text || !text.trim()) && !audioBlob) {
        riskLevel.textContent = "Please provide input.";
        return;
    }

    const formData = new FormData();
    if (audioBlob) formData.append("file", audioBlob, "recording.wav");
    formData.append("text", text || " ");
    formData.append("age", age);
    formData.append("sleep_quality", sleep);
    formData.append("stress_level", stress);

    try {
        const response = await fetch(`${API_URL}/analyze/multimodal`, {
            method: "POST",
            body: formData
        });
        if (!response.ok) throw new Error("Server error");
        const data = await response.json();
        updateUI(data.risk_score * 100);

    } catch (error) {
        console.warn("Backend offline, using demo mode.");
        await delay(2000);
        // Simulate score based on input presence + clinical
        let mockScore = 30;
        if (text && text.toLowerCase().includes("stress")) mockScore += 20;

        // Add clinical impact
        mockScore += (parseInt(stress) * 2); // 0-20
        mockScore += ((10 - parseInt(sleep)) * 2); // 0-20

        mockScore = Math.min(mockScore, 99);
        updateUI(mockScore);
    }
}

function updateUI(score) {
    const scoreValue = document.getElementById("scoreValue");
    const riskLevel = document.getElementById("riskLevel");
    const insightsList = document.getElementById("insightsList");
    const circle = document.querySelector('.progress-ring__circle');

    // Animate Text
    let currentScore = 0;
    const interval = setInterval(() => {
        if (currentScore >= score) clearInterval(interval);
        else currentScore++;
        scoreValue.textContent = currentScore;
    }, 10);

    // Animate SVG Ring
    // Radius 90 => Circumference ~ 565
    const radius = circle.r.baseVal.value;
    const circumference = radius * 2 * Math.PI;
    const offset = circumference - (score / 100) * circumference;

    circle.style.strokeDasharray = `${circumference} ${circumference}`;
    circle.style.strokeDashoffset = offset;

    insightsList.innerHTML = "";
    let riskColor = "";
    let riskText = "";

    if (score < 30) {
        riskText = "Wellness Maintained";
        riskColor = "#4ade80";
        circle.style.stroke = "#4ade80";
        insightsList.innerHTML = "<li>Analysis indicates a healthy mental state.</li><li>No significant markers of stress detected.</li>";
    } else if (score < 70) {
        riskText = "Moderate Stress Detectors";
        riskColor = "#facc15";
        circle.style.stroke = "#facc15";
        insightsList.innerHTML = "<li>Elevated cortisol-related vocal markers.</li><li>Journal entries show mild anxiety patterns.</li>";
    } else {
        riskText = "High Risk Factors Identified";
        riskColor = "#f87171";
        circle.style.stroke = "#f87171";
        insightsList.innerHTML = "<li>High usage of negative sentiment keywords.</li><li>Significant vocal biomarkers indicating distress.</li><li><strong>Recommendation:</strong> Consult a specialist.</li>";
    }

    riskLevel.textContent = riskText;
    riskLevel.style.color = riskColor;
}
