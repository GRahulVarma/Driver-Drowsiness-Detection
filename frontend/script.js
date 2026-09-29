const video = document.getElementById('video');
const statusDiv = document.getElementById('status');

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => { video.srcObject = stream; })
    .catch(err => console.error(err));

// Capture and send frames to backend
async function sendFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, 224, 224);

    const dataUrl = canvas.toDataURL('image/jpeg');
    const base64 = dataUrl.split(',')[1];

    try {
        const res = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64 })
        });

        const result = await res.json();
        statusDiv.innerText = 'Status: ' + result.status.toUpperCase();
        statusDiv.className = 'status ' + result.status;

        if(result.status === 'drowsy') {
            const audio = new Audio('alert.wav'); // relative to index.html
            audio.play();
        }

    } catch(err) {
        console.error(err);
    }
}

setInterval(sendFrame, 1000); // every 1 second
