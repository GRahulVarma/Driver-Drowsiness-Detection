import numpy as np
import wave
import struct
import os

# Folder path where you'll store it (inside your project)
sound_folder = "sounds"
os.makedirs(sound_folder, exist_ok=True)

file_path = os.path.join(sound_folder, "alert.wav")

# Generate a simple beep tone
frequency = 1000  # 1000 Hz beep
duration = 0.7    # seconds
volume = 0.5      # volume 0.0 to 1.0
sampling_rate = 44100

t = np.linspace(0, duration, int(sampling_rate * duration), False)
tone = np.sin(frequency * t * 2 * np.pi) * volume

# Save to WAV file
with wave.open(file_path, 'w') as wav_file:
    n_channels = 1
    sampwidth = 2
    n_frames = len(tone)
    comptype = "NONE"
    compname = "not compressed"
    wav_file.setparams((n_channels, sampwidth, sampling_rate, n_frames, comptype, compname))

    for s in tone:
        wav_file.writeframes(struct.pack('h', int(s * 32767.0)))

print(f"✅ Alert sound created successfully at: {file_path}")
