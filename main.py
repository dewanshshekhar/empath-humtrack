#!/usr/bin/env python3
"""
humtrack_pipeline.py
─────────────────────────────────────────────────────────────────────────────
HumTrack AI — Full Pipeline Orchestrator
Optimized for RTX 4060 (8GB VRAM)

Strategy: Sequential model loading with aggressive VRAM cleanup between stages.
Only ONE model lives in GPU memory at any time.

Pipeline stages:
  Stage 0 ── Record hum from microphone
  Stage 1 ── Kimi-Audio-7B-Q4  →  key, BPM, mood, note sequence
  Stage 2 ── ChatMusician-Q4   →  chord progression (ABC notation → MIDI)
  Stage 3 ── MusicGen-Chord    →  optional 30s preview audio
  Stage 4 ── FL Studio MCP     →  send MIDI to Piano Roll

Requirements:
  pip install sounddevice numpy scipy torch transformers accelerate
              bitsandbytes symusic audiocraft mido python-rtmidi rich

Usage:
  python humtrack_pipeline.py                    # full interactive run
  python humtrack_pipeline.py --skip-preview     # skip MusicGen (saves ~3 min)
  python humtrack_pipeline.py --duration 15      # record 15 seconds
  python humtrack_pipeline.py --bpm 140          # set project BPM
  python humtrack_pipeline.py --genre trap       # genre hint for ChatMusician
  python humtrack_pipeline.py --check            # VRAM / dependency check only
"""

import argparse
import gc
import json
import os
import sys
import time
import threading
import tempfile
import wave
from pathlib import Path
from datetime import datetime

# ── Rich console (pretty output) ─────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    RICH = True
except ImportError:
    RICH = False
    class Console:
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw): print("─" * 60)
    def rprint(*a, **kw): print(*a)

console = Console()

# ── Dependency checks ─────────────────────────────────────────────────────────
MISSING = []
try:
    import torch
except ImportError:
    MISSING.append("torch")
try:
    import numpy as np
except ImportError:
    MISSING.append("numpy")
try:
    import sounddevice as sd
except ImportError:
    MISSING.append("sounddevice")
try:
    from scipy.signal import butter, lfilter
except ImportError:
    MISSING.append("scipy")
try:
    import mido
    MIDO_OK = True
except ImportError:
    MIDO_OK = False
    MISSING.append("mido python-rtmidi")
try:
    import symusic
    SYMUSIC_OK = True
except ImportError:
    SYMUSIC_OK = False

if MISSING:
    console.print(f"[bold red]❌ Missing packages:[/bold red] {', '.join(MISSING)}")
    console.print(f"[yellow]Run:[/yellow] pip install {' '.join(MISSING)}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# VRAM MANAGER — the key to fitting on 8GB
# ══════════════════════════════════════════════════════════════════════════════

class VRAMManager:
    """
    Tracks and aggressively frees GPU memory between pipeline stages.
    On RTX 4060 (8GB), we MUST unload each model before loading the next.
    """

    def __init__(self):
        self.current_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            console.print("[yellow]⚠  No CUDA GPU found — running on CPU (very slow)[/yellow]")

    def free(self, label: str = ""):
        """Nuclear VRAM cleanup — call between every stage."""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        freed_label = f" [{label}]" if label else ""
        vram = self._vram_used_gb()
        console.print(f"[dim]🧹 VRAM freed{freed_label} → {vram:.2f} GB used[/dim]")

    def _vram_used_gb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0

    def _vram_free_gb(self) -> float:
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            used  = torch.cuda.memory_allocated() / 1e9
            return total - used
        return 0.0

    def status(self) -> str:
        if not torch.cuda.is_available():
            return "CPU mode"
        used  = self._vram_used_gb()
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        pct   = (used / total) * 100
        bar   = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        return f"{bar} {used:.1f}/{total:.1f}GB ({pct:.0f}%)"

    def report(self):
        if not torch.cuda.is_available():
            console.print("[dim]Running on CPU — no VRAM stats[/dim]")
            return
        props = torch.cuda.get_device_properties(0)
        console.print(f"[bold cyan]GPU:[/bold cyan] {props.name}")
        console.print(f"[bold cyan]VRAM:[/bold cyan] {props.total_memory / 1e9:.1f} GB total")
        console.print(f"[bold cyan]Used:[/bold cyan] {self._vram_used_gb():.2f} GB")
        console.print(f"[bold cyan]Free:[/bold cyan] {self._vram_free_gb():.2f} GB")


vram = VRAMManager()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — MICROPHONE RECORDING
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE  = 16000   # Kimi-Audio expects 16kHz
BLOCK_SIZE   = 1024
SILENCE_RMS  = 0.004

def record_hum(duration: float = 10.0) -> str:
    """
    Record from microphone for `duration` seconds.
    Returns path to WAV file (16kHz mono).
    """
    console.rule("[bold magenta]🎙  STAGE 0 — RECORDING[/bold magenta]")
    console.print(f"[bold]Recording for {duration:.0f} seconds...[/bold]")
    console.print("[dim]Hum or sing your melody clearly. Keep gaps between notes.[/dim]\n")

    frames = []
    stop_event = threading.Event()

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    # Countdown
    for i in range(3, 0, -1):
        console.print(f"[yellow]  Starting in {i}...[/yellow]", end="\r")
        time.sleep(1)
    console.print("[bold green]  ▶ RECORDING NOW — Hum your melody![/bold green]     ")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=BLOCK_SIZE, callback=callback):
        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            elapsed = time.perf_counter() - start
            remaining = duration - elapsed
            pct = int((elapsed / duration) * 30)
            bar = "█" * pct + "░" * (30 - pct)
            rms_val = np.abs(frames[-1]).mean() if frames else 0
            level = "█" * min(10, int(rms_val * 200))
            console.print(f"  [{bar}] {remaining:.1f}s  | Level: {level:<10}", end="\r")
            time.sleep(0.1)

    console.print("\n[bold green]  ✓ Recording complete![/bold green]")

    # Save to WAV
    audio = np.concatenate(frames, axis=0).flatten()
    audio_int16 = (audio * 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="humtrack_")
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    duration_secs = len(audio) / SAMPLE_RATE
    console.print(f"[dim]  Saved: {tmp.name} ({duration_secs:.1f}s, {len(audio):,} samples)[/dim]")
    return tmp.name


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — KIMI-AUDIO ANALYSIS
# 8GB budget: ~4.5GB for Kimi-Audio Q4 → leaves 3.5GB free
# ══════════════════════════════════════════════════════════════════════════════

KIMI_MODEL_ID = "moonshotai/Kimi-Audio-7B-Instruct"

KIMI_PROMPT = """You are analysing a hummed melody recorded from a microphone.
This is NOT speech — it is a musical hum or sung melody with no lyrics.

Please analyse this audio and return a JSON object with EXACTLY these fields:
{
  "key": "e.g. C minor, G major, F# minor",
  "bpm_estimate": 120,
  "mood": "e.g. melancholic, uplifting, dark, energetic",
  "genre_suggestion": "e.g. lo-fi hip hop, trap, R&B, jazz, cinematic",
  "notes": [
    {"pitch": "C4", "midi": 60, "duration_beats": 0.5, "position_beats": 0.0},
    ...
  ],
  "melodic_description": "brief description of the melodic contour and feel"
}

Notes array rules:
- pitch: scientific notation (C4, D#4, Bb3, etc.)
- midi: MIDI note number (0-127), middle C = 60
- duration_beats and position_beats: floating point beat values at detected BPM
- Return ONLY the JSON object, no explanation, no markdown fences.
"""

def analyse_with_kimi(wav_path: str, bpm_hint: float = None) -> dict:
    """
    Load Kimi-Audio-7B at Q4 precision, analyse the hum, return structured JSON.
    VRAM usage: ~4.5GB with Q4_K_M
    """
    console.rule("[bold cyan]🎧  STAGE 1 — KIMI-AUDIO ANALYSIS[/bold cyan]")
    console.print(f"[dim]VRAM before load: {vram.status()}[/dim]")

    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch

        console.print("[yellow]  Loading Kimi-Audio-7B (Q4 — ~4.5GB VRAM)...[/yellow]")
        t0 = time.time()

        # Load with 4-bit quantization to fit in 8GB
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        processor = AutoProcessor.from_pretrained(
            KIMI_MODEL_ID,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            KIMI_MODEL_ID,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        vram.current_model = model
        console.print(f"[green]  ✓ Model loaded in {time.time()-t0:.1f}s[/green]")
        console.print(f"[dim]  VRAM after load: {vram.status()}[/dim]")

        # Build the prompt with audio
        prompt_hint = f"\n\nNote: the project BPM is approximately {bpm_hint}." if bpm_hint else ""
        full_prompt = KIMI_PROMPT + prompt_hint

        # Format depends on Kimi-Audio's chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": wav_path},
                    {"type": "text", "text": full_prompt}
                ]
            }
        ]

        console.print("[yellow]  Running inference...[/yellow]")
        t1 = time.time()
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        response = processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        console.print(f"[green]  ✓ Analysis complete in {time.time()-t1:.1f}s[/green]")

    except Exception as e:
        console.print(f"[red]  ⚠ Kimi-Audio failed: {e}[/red]")
        console.print("[yellow]  → Falling back to autocorrelation pitch detector[/yellow]")
        vram.free("kimi-audio-failed")
        return _fallback_pitch_detection(wav_path, bpm_hint or 120.0)
    finally:
        vram.free("kimi-audio")

    # Parse JSON from response
    return _parse_kimi_response(response, bpm_hint)


def _parse_kimi_response(raw: str, bpm_hint: float = None) -> dict:
    """Extract and validate JSON from Kimi-Audio's response."""
    import re
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    # Find JSON block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        console.print("[red]  ⚠ Could not parse JSON from Kimi-Audio response[/red]")
        console.print(f"[dim]  Raw output: {raw[:300]}[/dim]")
        return _fallback_pitch_detection("", bpm_hint or 120.0)
    try:
        data = json.loads(match.group(0))
        # Normalise fields
        data.setdefault("key", "C major")
        data.setdefault("bpm_estimate", bpm_hint or 120.0)
        data.setdefault("mood", "neutral")
        data.setdefault("genre_suggestion", "lo-fi")
        data.setdefault("notes", [])
        data.setdefault("melodic_description", "")
        console.print(f"[green]  ✓ Detected key: [bold]{data['key']}[/bold]  BPM: {data['bpm_estimate']}  Mood: {data['mood']}[/green]")
        console.print(f"[green]  ✓ {len(data['notes'])} notes detected[/green]")
        return data
    except json.JSONDecodeError as e:
        console.print(f"[red]  ⚠ JSON parse error: {e}[/red]")
        return _fallback_pitch_detection("", bpm_hint or 120.0)


def _fallback_pitch_detection(wav_path: str, bpm: float) -> dict:
    """
    Pure Python autocorrelation fallback — no GPU needed.
    Used when Kimi-Audio is unavailable or fails.
    """
    console.print("[dim]  Using autocorrelation fallback detector...[/dim]")
    if not wav_path or not Path(wav_path).exists():
        return {"key": "C major", "bpm_estimate": bpm, "mood": "neutral",
                "genre_suggestion": "lo-fi", "notes": [], "melodic_description": "fallback"}

    with wave.open(wav_path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        sr  = wf.getframerate()
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    BLOCK = 2048
    HOP   = 512
    MIN_F, MAX_F = 60.0, 1200.0
    SILENCE = 0.005

    def autocorr_freq(frame):
        if np.sqrt(np.mean(frame**2)) < SILENCE:
            return None
        b, a = butter(4, [MIN_F / (sr/2), MAX_F / (sr/2)], btype="band")
        frame = lfilter(b, a, frame)
        N = len(frame)
        min_lag = int(sr / MAX_F)
        max_lag = int(sr / MIN_F)
        corr = np.correlate(frame, frame, "full")[N-1:]
        corr = corr[min_lag:max_lag]
        if not len(corr):
            return None
        peak = np.argmax(corr) + min_lag
        conf = corr[peak-min_lag] / (corr[0] + 1e-9)
        if conf < 0.3:
            return None
        return sr / peak

    def freq_to_midi(f):
        return int(round(12 * np.log2(f / 440.0) + 69))

    def midi_to_name(m):
        names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        return f"{names[m%12]}{m//12-1}"

    pitch_stream = []
    for i in range(0, len(audio)-BLOCK, HOP):
        f = autocorr_freq(audio[i:i+BLOCK])
        m = freq_to_midi(f) if f else None
        pitch_stream.append((i / sr, m))

    # Median smooth
    def med_smooth(seq, w=5):
        out = []
        for i in range(len(seq)):
            chunk = [v for _, v in seq[max(0,i-w//2):i+w//2+1] if v is not None]
            out.append(sorted(chunk)[len(chunk)//2] if chunk else None)
        return out

    smoothed = med_smooth(pitch_stream)
    beats_ps = bpm / 60.0
    notes, cur, start = [], None, 0.0

    for i, ((ts, _), midi) in enumerate(zip(pitch_stream, smoothed)):
        if midi != cur:
            if cur and (ts - start) >= 0.08:
                lb = (ts - start) * beats_ps
                pb = start * beats_ps
                grid = 0.125
                notes.append({
                    "pitch": midi_to_name(cur),
                    "midi": cur,
                    "duration_beats": round(round(lb/grid)*grid, 4),
                    "position_beats": round(round(pb/grid)*grid, 4),
                })
            cur, start = midi, ts

    console.print(f"[dim]  Fallback detected {len(notes)} notes[/dim]")
    return {
        "key": "C minor",
        "bpm_estimate": bpm,
        "mood": "neutral",
        "genre_suggestion": "lo-fi",
        "notes": notes,
        "melodic_description": "Autocorrelation fallback — no AI key/mood analysis"
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — CHATMUSICIAN HARMONY
# 8GB budget: ~4GB for ChatMusician Q4 → safe after Kimi unloaded
# ══════════════════════════════════════════════════════════════════════════════

CHATMUSICIAN_ID = "m-a-p/ChatMusician"

def generate_harmony(analysis: dict, genre: str = None) -> dict:
    """
    Load ChatMusician at Q4, generate chord progression + ABC notation.
    VRAM usage: ~4GB with Q4_K_M
    Returns dict with 'abc_notation', 'chords', 'midi_notes'
    """
    console.rule("[bold green]🎵  STAGE 2 — CHATMUSICIAN HARMONY[/bold green]")
    console.print(f"[dim]VRAM before load: {vram.status()}[/dim]")

    key    = analysis.get("key", "C minor")
    mood   = analysis.get("mood", "neutral")
    bpm    = analysis.get("bpm_estimate", 120)
    g      = genre or analysis.get("genre_suggestion", "lo-fi hip hop")
    desc   = analysis.get("melodic_description", "")
    notes  = analysis.get("notes", [])

    # Build note string from detected melody
    note_str = " ".join(f"{n['pitch']}({n['duration_beats']}b)" for n in notes[:16])

    prompt = f"""You are an expert music composer. Generate a chord progression and harmonised arrangement.

Input melody:
- Key: {key}
- BPM: {bpm}
- Mood: {mood}
- Genre: {g}
- Melodic notes: {note_str}
- Description: {desc}

Task: Generate a 4-bar chord progression that harmonises this melody, then write it as ABC notation.

Return ONLY a JSON object:
{{
  "chords": ["Cm", "Ab", "Eb", "Bb"],
  "chord_description": "i–VI–III–VII in C minor, classic lo-fi progression",
  "abc_notation": "X:1\\nT:HumTrack Harmony\\nM:4/4\\nL:1/8\\nQ:{int(bpm)}\\nK:{key}\\n| ...notes... |",
  "arrangement_notes": "brief note on how to use this in production"
}}"""

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        console.print("[yellow]  Loading ChatMusician (Q4 — ~4GB VRAM)...[/yellow]")
        t0 = time.time()

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        tokenizer = AutoTokenizer.from_pretrained(CHATMUSICIAN_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            CHATMUSICIAN_ID,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        vram.current_model = model
        console.print(f"[green]  ✓ Model loaded in {time.time()-t0:.1f}s[/green]")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        console.print("[yellow]  Generating harmony...[/yellow]")
        t1 = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        console.print(f"[green]  ✓ Harmony generated in {time.time()-t1:.1f}s[/green]")

    except Exception as e:
        console.print(f"[red]  ⚠ ChatMusician failed: {e}[/red]")
        console.print("[yellow]  → Using rule-based harmony fallback[/yellow]")
        vram.free("chatmusician-failed")
        return _fallback_harmony(analysis, genre)
    finally:
        vram.free("chatmusician")

    return _parse_harmony_response(response, analysis)


def _parse_harmony_response(raw: str, analysis: dict) -> dict:
    import re
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return _fallback_harmony(analysis)
    try:
        data = json.loads(match.group(0))
        data.setdefault("chords", ["Cm", "Ab", "Eb", "Bb"])
        data.setdefault("chord_description", "")
        data.setdefault("abc_notation", "")
        data.setdefault("arrangement_notes", "")
        # Convert ABC notation to MIDI notes
        data["midi_notes"] = _abc_to_midi_notes(data["abc_notation"], analysis.get("bpm_estimate", 120))
        console.print(f"[green]  ✓ Chords: [bold]{' – '.join(data['chords'])}[/bold][/green]")
        console.print(f"[green]  ✓ {len(data['midi_notes'])} MIDI notes generated[/green]")
        return data
    except Exception as e:
        console.print(f"[red]  ⚠ Parse error: {e}[/red]")
        return _fallback_harmony(analysis)


def _fallback_harmony(analysis: dict, genre: str = None) -> dict:
    """Rule-based chord generation when ChatMusician unavailable."""
    console.print("[dim]  Using rule-based harmony fallback...[/dim]")
    key = analysis.get("key", "C minor").lower()
    bpm = analysis.get("bpm_estimate", 120)

    # Common progressions by key type
    minor_progressions = {
        "lo-fi": (["Cm", "Ab", "Eb", "Bb"], "i–VI–III–VII — classic lo-fi"),
        "trap":  (["Cm", "Gm", "Ab", "Bb"], "i–v–VI–VII — dark trap"),
        "rnb":   (["Cm", "Fm", "Bb", "Eb"], "i–iv–VII–III — R&B minor"),
        "jazz":  (["Cm7", "Fm7", "Bb7", "Ebmaj7"], "i7–iv7–VII7–IIImaj7 — jazz"),
    }
    major_progressions = {
        "lo-fi": (["C", "Am", "F", "G"], "I–vi–IV–V — classic major"),
        "trap":  (["C", "F", "Am", "G"], "I–IV–vi–V — pop trap"),
        "rnb":   (["Cmaj7", "Am7", "Fmaj7", "G7"], "Imaj7–vi7–IVmaj7–V7 — R&B"),
        "jazz":  (["Cmaj7", "Dm7", "G7", "Cmaj7"], "Imaj7–ii7–V7–I — jazz ii-V-I"),
    }

    mode = "minor" if "minor" in key else "major"
    g = (genre or analysis.get("genre_suggestion", "lo-fi")).lower().split()[0]
    pool = minor_progressions if mode == "minor" else major_progressions
    chords, desc = pool.get(g, pool.get("lo-fi"))

    # Build simple MIDI chord notes
    midi_notes = []
    chord_roots = {"C":60,"D":62,"E":64,"F":65,"G":67,"A":69,"B":71,
                   "Cm":60,"Dm":62,"Em":64,"Fm":65,"Gm":67,"Am":69,"Bm":71}

    for i, chord in enumerate(chords):
        root = chord_roots.get(chord.replace("maj7","").replace("7","").replace("m","m"), 60)
        for note_offset in [0, 4, 7]:  # root, third, fifth
            midi_notes.append({
                "note": root + note_offset,
                "velocity": 65,
                "length_beats": 3.75,
                "position_beats": float(i * 4),
            })

    return {
        "chords": chords,
        "chord_description": desc,
        "abc_notation": "",
        "arrangement_notes": "Rule-based fallback harmony",
        "midi_notes": midi_notes,
    }


def _abc_to_midi_notes(abc: str, bpm: float) -> list:
    """Convert ABC notation to MIDI note list using symusic if available."""
    if not SYMUSIC_OK or not abc:
        return []
    try:
        import symusic
        # symusic can parse ABC notation directly
        score = symusic.Score.from_abc(abc)
        notes = []
        beats_per_sec = bpm / 60.0
        for track in score.tracks:
            for note in track.notes:
                pos_beats  = note.time  / score.tpq
                len_beats  = note.duration / score.tpq
                notes.append({
                    "note":           int(note.pitch),
                    "velocity":       int(note.velocity),
                    "length_beats":   round(len_beats, 4),
                    "position_beats": round(pos_beats, 4),
                })
        return notes
    except Exception as e:
        console.print(f"[dim]  ABC→MIDI conversion error: {e}[/dim]")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — MUSICGEN-CHORD PREVIEW (OPTIONAL)
# 8GB budget: ~3.5GB for MusicGen-small → fits after others unloaded
# Skip this stage if --skip-preview or VRAM is tight
# ══════════════════════════════════════════════════════════════════════════════

def generate_preview(harmony: dict, analysis: dict, output_path: str = "humtrack_preview.wav") -> str | None:
    """
    Generate a 30-second backing track using MusicGen-Chord.
    Uses the small model (~3.5GB) to stay within 8GB VRAM.
    Returns path to WAV file, or None on failure.
    """
    console.rule("[bold yellow]🔊  STAGE 3 — MUSICGEN PREVIEW[/bold yellow]")
    console.print(f"[dim]VRAM before load: {vram.status()}[/dim]")
    console.print("[dim]Using MusicGen-small (~3.5GB) for RTX 4060 compatibility[/dim]")

    chords  = " ".join(harmony.get("chords", ["Cm", "Ab", "Eb", "Bb"]) * 2)
    mood    = analysis.get("mood", "chill")
    genre   = analysis.get("genre_suggestion", "lo-fi hip hop")
    bpm     = int(analysis.get("bpm_estimate", 120))
    prompt  = f"{genre} instrumental, {mood}, {bpm} BPM, chords: {chords}, no vocals, high quality"

    try:
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write
        import torch

        console.print("[yellow]  Loading MusicGen-small (chord-conditioned)...[/yellow]")
        t0 = time.time()
        # Use sakemin's chord-conditioned fork if available, else fall back to small
        try:
            model = MusicGen.get_pretrained("sakemin/musicgen-chord")
        except Exception:
            console.print("[dim]  sakemin/musicgen-chord not found, using facebook/musicgen-small[/dim]")
            model = MusicGen.get_pretrained("facebook/musicgen-small")

        model.set_generation_params(duration=30, use_sampling=True, top_k=250)
        vram.current_model = model
        console.print(f"[green]  ✓ Model loaded in {time.time()-t0:.1f}s[/green]")
        console.print(f"[yellow]  Generating 30s preview: \"{prompt[:80]}\"...[/yellow]")
        t1 = time.time()

        with torch.no_grad():
            wav = model.generate([prompt])

        out_base = output_path.replace(".wav", "")
        audio_write(out_base, wav[0].cpu(), model.sample_rate,
                    strategy="loudness", loudness_compressor=True)

        actual_path = out_base + ".wav"
        console.print(f"[green]  ✓ Preview generated in {time.time()-t1:.1f}s → {actual_path}[/green]")
        return actual_path

    except Exception as e:
        console.print(f"[red]  ⚠ MusicGen failed: {e}[/red]")
        console.print("[dim]  Skipping audio preview — MIDI will still be sent to FL Studio[/dim]")
        return None
    finally:
        vram.free("musicgen")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — FL STUDIO MCP DELIVERY
# No GPU needed — pure Python MIDI over LoopMIDI
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_PORT = "loopMIDI Port 1"   # ← change to your port name
ENCRYPT_CH  = 0

def _pack_int(value: int, cc_hi: int, cc_lo: int) -> list:
    value = max(0, min(16383, int(round(value))))
    return [
        mido.Message("control_change", channel=ENCRYPT_CH, control=cc_hi, value=(value>>7)&0x7F),
        mido.Message("control_change", channel=ENCRYPT_CH, control=cc_lo, value=value&0x7F),
    ]

def _note_to_messages(note, velocity, length_beats, position_beats) -> list:
    msgs  = _pack_int(int(note)*100, 20, 21)
    msgs += _pack_int(int(velocity),  22, 23)
    msgs += _pack_int(int(length_beats*1000), 24, 25)
    msgs += _pack_int(int(position_beats*1000), 26, 27)
    msgs.append(mido.Message("note_on", channel=ENCRYPT_CH, note=60, velocity=64))
    return msgs

def send_to_flstudio(analysis: dict, harmony: dict, transpose: int = 0) -> bool:
    """
    Send melody + chord MIDI tracks to FL Studio via MCP/LoopMIDI.
    Melody on channel 0, chords on channel 1.
    """
    console.rule("[bold magenta]🎹  STAGE 4 — FL STUDIO DELIVERY[/bold magenta]")

    if not MIDO_OK:
        console.print("[red]  ⚠ mido/python-rtmidi not installed — cannot send MIDI[/red]")
        return False

    # Check port exists
    available = mido.get_output_names()
    if OUTPUT_PORT not in available:
        console.print(f"[red]  ⚠ MIDI port '{OUTPUT_PORT}' not found![/red]")
        console.print(f"[yellow]  Available ports: {available or ['(none)']}")
        console.print("[yellow]  → Install LoopMIDI (Windows) or use IAC Driver (Mac)[/yellow]")
        return False

    melody_notes = analysis.get("notes", [])
    chord_notes  = harmony.get("midi_notes", [])
    bpm          = analysis.get("bpm_estimate", 120)

    if not melody_notes and not chord_notes:
        console.print("[red]  ⚠ No notes to send![/red]")
        return False

    console.print(f"[yellow]  Sending {len(melody_notes)} melody + {len(chord_notes)} chord notes...[/yellow]")
    console.print("[dim]  Make sure FL Studio is open and an instrument is selected![/dim]")

    try:
        with mido.open_output(OUTPUT_PORT) as port:
            # ── Melody track ──────────────────────────────────────────────
            if melody_notes:
                console.print(f"[cyan]  → Melody track ({len(melody_notes)} notes)[/cyan]")
                for n in melody_notes:
                    midi_note = max(0, min(127, int(n.get("midi", 60)) + transpose))
                    for msg in _note_to_messages(midi_note, n.get("velocity",80),
                                                 n.get("duration_beats",0.5),
                                                 n.get("position_beats",0.0)):
                        port.send(msg)
                        time.sleep(0.005)

            time.sleep(0.1)  # brief pause between tracks

            # ── Chord track ───────────────────────────────────────────────
            if chord_notes:
                console.print(f"[cyan]  → Chord track ({len(chord_notes)} notes)[/cyan]")
                for n in chord_notes:
                    midi_note = max(0, min(127, int(n.get("note", 60)) + transpose))
                    for msg in _note_to_messages(midi_note, n.get("velocity",65),
                                                 n.get("length_beats",3.75),
                                                 n.get("position_beats",0.0)):
                        port.send(msg)
                        time.sleep(0.005)

        console.print("[bold green]  ✓ MIDI sent successfully to FL Studio![/bold green]")
        console.print("[green]  → Check your Piano Roll — melody + chords should be there[/green]")
        return True

    except Exception as e:
        console.print(f"[red]  ⚠ MIDI send failed: {e}[/red]")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SAVE SESSION
# ══════════════════════════════════════════════════════════════════════════════

def save_session(analysis: dict, harmony: dict, preview_path: str | None,
                 wav_path: str, args) -> str:
    """Save full session JSON for later re-use or debugging."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"humtrack_session_{ts}.json"
    session = {
        "timestamp":    ts,
        "wav_input":    wav_path,
        "preview_wav":  preview_path,
        "args": {
            "duration": args.duration,
            "bpm":      args.bpm,
            "genre":    args.genre,
            "transpose":args.transpose,
        },
        "analysis":  analysis,
        "harmony":   harmony,
    }
    Path(out_path).write_text(json.dumps(session, indent=2))
    console.print(f"[dim]💾 Session saved: {out_path}[/dim]")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY / VRAM CHECK
# ══════════════════════════════════════════════════════════════════════════════

def run_check():
    """Print a full system readiness report."""
    console.print(Panel("[bold]HumTrack AI — System Check[/bold]", style="cyan"))

    # GPU
    vram.report()
    console.print()

    # Models — estimated VRAM per stage
    table = Table(title="Pipeline VRAM Budget (RTX 4060 — 8GB)")
    table.add_column("Stage", style="cyan")
    table.add_column("Model", style="white")
    table.add_column("VRAM (Q4)", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Fits 8GB?", style="bold")

    rows = [
        ("Stage 1", "Kimi-Audio-7B-Instruct", "~4.5GB", "Sequential — others unloaded", "✅ Yes"),
        ("Stage 2", "ChatMusician (LLaMA2-7B)", "~4.0GB", "Sequential — others unloaded", "✅ Yes"),
        ("Stage 3", "MusicGen-small", "~3.5GB", "Sequential — optional", "✅ Yes"),
        ("Stage 4", "No model (MIDI only)", "0GB", "Always available", "✅ Yes"),
        ("Fallback 1", "Whisper + autocorrelation", "~2.0GB", "If Kimi-Audio fails", "✅ Yes"),
    ]
    for r in rows:
        table.add_row(*r)
    console.print(table)

    console.print()

    # Dependencies
    deps = [
        ("sounddevice", "sounddevice"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("bitsandbytes", "bitsandbytes"),
        ("symusic", "symusic"),
        ("mido", "mido"),
        ("audiocraft", "audiocraft"),
        ("rich", "rich"),
    ]
    dep_table = Table(title="Python Dependencies")
    dep_table.add_column("Package")
    dep_table.add_column("Status")
    for pkg, imp in deps:
        try:
            __import__(imp)
            dep_table.add_row(pkg, "[green]✅ Installed[/green]")
        except ImportError:
            dep_table.add_row(pkg, "[red]❌ Missing[/red]")
    console.print(dep_table)

    console.print()
    console.print("[bold]MIDI Ports:[/bold]")
    if MIDO_OK:
        ports = mido.get_output_names()
        if ports:
            for p in ports:
                marker = "[green]✅[/green]" if p == OUTPUT_PORT else "  "
                console.print(f"  {marker} {p}")
        else:
            console.print("  [red]No MIDI output ports found[/red]")
            console.print("  [yellow]→ Install LoopMIDI (Windows) or enable IAC Driver (Mac)[/yellow]")
    else:
        console.print("  [red]mido not installed[/red]")

    console.print()
    console.print("[bold]Install command for all missing packages:[/bold]")
    console.print("[yellow]pip install sounddevice numpy scipy torch torchvision torchaudio transformers accelerate bitsandbytes symusic mido python-rtmidi audiocraft rich[/yellow]")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(analysis: dict, harmony: dict, preview_path: str | None, session_path: str):
    """Print a clean results summary table."""
    console.rule("[bold]📊  RESULTS SUMMARY[/bold]")
    t = Table(show_header=False, box=None, padding=(0,2))
    t.add_column("Field", style="cyan", width=24)
    t.add_column("Value", style="white")
    t.add_row("Detected Key",     analysis.get("key", "—"))
    t.add_row("BPM",              str(analysis.get("bpm_estimate", "—")))
    t.add_row("Mood",             analysis.get("mood", "—"))
    t.add_row("Genre Suggestion", analysis.get("genre_suggestion", "—"))
    t.add_row("Melody Notes",     str(len(analysis.get("notes", []))))
    t.add_row("Chord Progression"," – ".join(harmony.get("chords", [])))
    t.add_row("Chord Notes (MIDI)",str(len(harmony.get("midi_notes", []))))
    t.add_row("Preview Audio",    preview_path or "skipped")
    t.add_row("Session File",     session_path)
    console.print(t)


def main():
    parser = argparse.ArgumentParser(
        description="HumTrack AI — Hum to FL Studio Pipeline (RTX 4060 optimised)"
    )
    parser.add_argument("--duration",     type=float, default=10.0,  help="Recording duration in seconds (default: 10)")
    parser.add_argument("--bpm",          type=float, default=None,  help="Project BPM hint (default: auto-detect)")
    parser.add_argument("--genre",        type=str,   default=None,  help="Genre hint: lofi, trap, rnb, jazz, cinematic")
    parser.add_argument("--transpose",    type=int,   default=0,     help="Semitones to transpose before sending (default: 0)")
    parser.add_argument("--skip-preview", action="store_true",       help="Skip MusicGen audio preview (saves ~3 min + VRAM)")
    parser.add_argument("--check",        action="store_true",       help="Run system check and exit")
    parser.add_argument("--no-flstudio",  action="store_true",       help="Skip FL Studio delivery (output JSON only)")
    parser.add_argument("--load-session", type=str,   default=None,  help="Re-send a saved session JSON to FL Studio")
    parser.add_argument("--midi-port",    type=str,   default=None,  help=f"MIDI output port name (default: '{OUTPUT_PORT}')")
    args = parser.parse_args()

    # ── override midi port ─────────────────────────────────────────────────────
    global OUTPUT_PORT
    if args.midi_port:
        OUTPUT_PORT = args.midi_port

    # ── check mode ────────────────────────────────────────────────────────────
    if args.check:
        run_check()
        return

    # ── re-send existing session ───────────────────────────────────────────────
    if args.load_session:
        console.print(f"[cyan]Loading session: {args.load_session}[/cyan]")
        session = json.loads(Path(args.load_session).read_text())
        send_to_flstudio(session["analysis"], session["harmony"], transpose=args.transpose)
        return

    # ── Banner ─────────────────────────────────────────────────────────────────
    console.print(Panel(
        "[bold white]HumTrack AI[/bold white]  [dim]— RTX 4060 Optimised Pipeline[/dim]\n"
        "[dim]Sequential model loading · 1 model in VRAM at a time · Max 5GB per stage[/dim]",
        style="cyan", expand=False
    ))
    vram.report()
    console.print()

    t_total = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 0: Record
    # ─────────────────────────────────────────────────────────────────────────
    wav_path = record_hum(duration=args.duration)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1: Kimi-Audio Analysis
    # ─────────────────────────────────────────────────────────────────────────
    analysis = analyse_with_kimi(wav_path, bpm_hint=args.bpm)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2: ChatMusician Harmony
    # ─────────────────────────────────────────────────────────────────────────
    harmony = generate_harmony(analysis, genre=args.genre)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3: MusicGen Preview (optional)
    # ─────────────────────────────────────────────────────────────────────────
    preview_path = None
    if not args.skip_preview:
        preview_path = generate_preview(harmony, analysis)
    else:
        console.print("[dim]  Skipping audio preview (--skip-preview)[/dim]")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4: FL Studio Delivery
    # ─────────────────────────────────────────────────────────────────────────
    if not args.no_flstudio:
        send_to_flstudio(analysis, harmony, transpose=args.transpose)
    else:
        console.print("[dim]  Skipping FL Studio delivery (--no-flstudio)[/dim]")

    # ─────────────────────────────────────────────────────────────────────────
    # Save & Summarise
    # ─────────────────────────────────────────────────────────────────────────
    session_path = save_session(analysis, harmony, preview_path, wav_path, args)
    print_summary(analysis, harmony, preview_path, session_path)

    elapsed = time.time() - t_total
    console.print(f"\n[bold green]✅  Total pipeline time: {elapsed:.1f}s[/bold green]")
    console.print("[dim]  Tip: use --skip-preview to drop 60–180s off total time[/dim]")


if __name__ == "__main__":
    main()