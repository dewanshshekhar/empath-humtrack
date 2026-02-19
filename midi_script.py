import sounddevice as sd
import numpy as np
import torch
import torchcrepe
import pretty_midi
import time
from scipy.signal import medfilt

# ==========================
# CONFIG
# ==========================
DURATION = 8
SR = 16000              # torchcrepe expects 16kHz
HOP_LENGTH = 160        # 10ms hop (160 samples at 16kHz)
CONF_THRESHOLD = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

print("🎤 Recording in 2 seconds...")
time.sleep(2)
print("Recording... Hum clearly!")

audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
sd.wait()
audio = audio.flatten().astype(np.float32)

print("Recording complete.")

# ==========================
# TORCHCREPE PITCH DETECTION
# ==========================
print("🎼 Detecting pitch with torchcrepe...")

audio_tensor = torch.tensor(audio).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pitch = torchcrepe.predict(
        audio_tensor,
        SR,
        HOP_LENGTH,
        fmin=50,
        fmax=1000,
        model='full',
        batch_size=1024,
        device=DEVICE,
        return_periodicity=True
    )

frequency = pitch[0].cpu().numpy().flatten()
confidence = pitch[1].cpu().numpy().flatten()

# Remove low-confidence frames
frequency[confidence < CONF_THRESHOLD] = 0

# Smooth frequency
frequency = medfilt(frequency, kernel_size=5)

# ==========================
# CONVERT Hz → MIDI
# ==========================
midi_notes = []
for f in frequency:
    if f > 0:
        midi = int(round(69 + 12 * np.log2(f / 440.0)))
        midi_notes.append(midi)

# Remove duplicates (basic segmentation)
clean_notes = []
for note in midi_notes:
    if len(clean_notes) == 0 or note != clean_notes[-1]:
        clean_notes.append(note)

print("Detected MIDI Notes:")
print(clean_notes)

# ==========================
# CREATE MIDI FILE
# ==========================
print("🎹 Creating MIDI file...")

pm = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)

start = 0
note_duration = 0.4

for note in clean_notes:
    midi_note = pretty_midi.Note(
        velocity=100,
        pitch=note,
        start=start,
        end=start + note_duration
    )
    instrument.notes.append(midi_note)
    start += 0.5

pm.instruments.append(instrument)
pm.write("humming_output.mid")

print("✅ MIDI saved as humming_output.mid")
