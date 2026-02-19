import sounddevice as sd
import numpy as np
import librosa
import pretty_midi
import scipy.io.wavfile as wav
import time
from collections import Counter
import simpleaudio as sa

# --------------------------
# CONFIG
# --------------------------
DURATION = 8  # recording duration in seconds
SR = 22050    # sample rate

print("🎙️ Recording in 2 seconds...")
time.sleep(2)
print("🎤 Hum now!")
audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
sd.wait()
audio = audio.flatten()

wav.write("hum.wav", SR, audio)
print("📁 Saved as hum.wav")

# --------------------------
# PITCH DETECTION
# --------------------------
print("🎧 Detecting pitch…")
f0 = librosa.yin(audio,
                 fmin=librosa.note_to_hz("C2"),
                 fmax=librosa.note_to_hz("C7"),
                 sr=SR)

f0_clean = f0[f0 > 0]
if len(f0_clean) == 0:
    print("⚠️ No voiced notes detected — try again!")
    exit()

midi_notes = librosa.hz_to_midi(f0_clean)
rounded = np.round(midi_notes)
note_names = [pretty_midi.note_number_to_name(int(n)) for n in rounded]
filtered_notes = []
for n in note_names:
    if not filtered_notes or filtered_notes[-1] != n:
        filtered_notes.append(n)

# --------------------------
# BPM
# --------------------------
print("⏱️ Estimating BPM…")
tempo, beats = librosa.beat.beat_track(y=audio, sr=SR)
tempo_val = float(tempo)

# --------------------------
# SCALE & KEY
# --------------------------
NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
MAJOR = [2,2,1,2,2,2,1]
MINOR = [2,1,2,2,1,2,2]

def build_scale(root, pattern):
    scale = [NOTES[root]]
    idx = root
    for step in pattern[:-1]:
        idx = (idx + step) % 12
        scale.append(NOTES[idx])
    return scale

SCALES = {}
for i,n in enumerate(NOTES):
    SCALES[f"{n} Major"] = build_scale(i, MAJOR)
    SCALES[f"{n} Minor"] = build_scale(i, MINOR)

pitch_classes = [n[:-1] for n in filtered_notes]
counts = Counter(pitch_classes)
best_key, best_score = None, 0
for key,scale in SCALES.items():
    score = sum(counts[p] for p in scale)
    if score > best_score:
        best_score = score
        best_key = key

# --------------------------
# RESULTS
# --------------------------
print("\n🎵 Analysis")
print("---------------------")
print("Notes:", filtered_notes)
print("BPM:", round(tempo_val,2))
print("Key:", best_key if best_key else "Unknown")

# --------------------------
# PLAYBACK
# --------------------------
print("\n🔊 Playing detected notes…")

def note_to_freq(note):
    return pretty_midi.note_number_to_hz(pretty_midi.note_name_to_number(note))

for note in filtered_notes:
    freq = note_to_freq(note)

    duration = 0.4  # constant duration
    t = np.linspace(0, duration, int(SR * duration), False)
    wave = 0.4 * np.sin(freq * t * 2 * np.pi)  # lower amplitude

    audio_data = (wave * 32767).astype(np.int16)
    play_obj = sa.play_buffer(audio_data, 1, 2, SR)
    play_obj.wait_done()
