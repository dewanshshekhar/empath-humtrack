// --- State ---
let isRecording = false;
let hasRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let audioBlob = null;
let audioContext = null;
let analyser = null;
let animFrameId = null;
let timerInterval = null;
let seconds = 0;
let selectedGenre = 'lo-fi';
let isPlaying = false;
let playInterval = null;
let playProgress = 0;

// --- DOM ---
const btnRecord = document.getElementById('btnRecord');
const btnClear = document.getElementById('btnClear');
const btnPlayPreview = document.getElementById('btnPlayPreview');
const btnGenerate = document.getElementById('btnGenerate');
const timerValue = document.getElementById('timerValue');
const waveformIdle = document.getElementById('waveformIdle');
const waveformCanvas = document.getElementById('waveformCanvas');
const recDot = document.getElementById('recDot');
const waveformLabel = document.getElementById('waveformLabel');
const outputCard = document.getElementById('outputCard');
const audioPlayer = document.getElementById('audioPlayer');
const progressBar = document.getElementById('progressBar');
const btnPlay = document.getElementById('btnPlay');
const seekFill = document.getElementById('seekFill');
const timeDisplay = document.getElementById('timeDisplay');
const trackMeta = document.getElementById('trackMeta');

// Genre chips
document.querySelectorAll('.genre-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    document.querySelectorAll('.genre-chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    selectedGenre = chip.dataset.genre;
  });
});

// Format time
function fmt(s) {
  return `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;
}

// --- Recording ---
btnRecord.addEventListener('click', async () => {
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      startRecording(stream);
    } catch(e) {
      alert('Microphone access denied. Please allow mic access to record your hum.');
    }
  } else {
    stopRecording();
  }
});

function startRecording(stream) {
  isRecording = true;
  audioChunks = [];
  btnRecord.classList.add('recording');
  recDot.style.display = 'block';
  waveformIdle.style.display = 'none';
  waveformCanvas.style.display = 'block';
  waveformLabel.style.display = 'none';

  // Timer
  seconds = 0;
  timerValue.textContent = fmt(0);
  timerInterval = setInterval(() => {
    seconds++;
    timerValue.textContent = fmt(seconds);
    if (seconds >= 30) stopRecording();
  }, 1000);

  // Analyser
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 256;
  const source = audioContext.createMediaStreamSource(stream);
  source.connect(analyser);
  drawWaveform();

  // MediaRecorder
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.addEventListener('dataavailable', e => audioChunks.push(e.data));
  mediaRecorder.addEventListener('stop', () => {
    audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    hasRecording = true;
    btnGenerate.disabled = false;
    btnClear.disabled = false;
    btnPlayPreview.disabled = false;
  });
  mediaRecorder.start();
}

function stopRecording() {
  isRecording = false;
  btnRecord.classList.remove('recording');
  recDot.style.display = 'none';
  clearInterval(timerInterval);
  cancelAnimationFrame(animFrameId);
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
  }
  if (audioContext) audioContext.close();
  waveformLabel.textContent = '● RECORDED';
  waveformLabel.style.color = 'var(--accent)';
  waveformLabel.style.display = 'block';
}

function drawWaveform() {
  const canvas = waveformCanvas;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth;
  const H = canvas.offsetHeight;
  canvas.width = W;
  canvas.height = H;

  const bufLen = analyser.frequencyBinCount;
  const dataArr = new Uint8Array(bufLen);

  function draw() {
    animFrameId = requestAnimationFrame(draw);
    analyser.getByteTimeDomainData(dataArr);
    ctx.fillStyle = '#060810';
    ctx.fillRect(0, 0, W, H);

    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(232,255,71,0.8)';
    ctx.beginPath();

    const slice = W / bufLen;
    let x = 0;
    for (let i = 0; i < bufLen; i++) {
      const v = dataArr[i] / 128.0;
      const y = (v * H) / 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      x += slice;
    }
    ctx.lineTo(W, H/2);
    ctx.stroke();
  }
  draw();
}

// Preview
btnPlayPreview.addEventListener('click', () => {
  if (!audioBlob) return;
  const url = URL.createObjectURL(audioBlob);
  const a = new Audio(url);
  a.play();
  btnPlayPreview.textContent = '▶ Playing...';
  a.onended = () => { btnPlayPreview.textContent = 'Preview'; };
});

// Clear
btnClear.addEventListener('click', () => {
  hasRecording = false;
  audioBlob = null;
  audioChunks = [];
  seconds = 0;
  timerValue.textContent = '0:00';
  btnGenerate.disabled = true;
  btnClear.disabled = true;
  btnPlayPreview.disabled = true;
  waveformCanvas.style.display = 'none';
  waveformIdle.style.display = 'flex';
  waveformLabel.textContent = 'HUM YOUR MELODY';
  waveformLabel.style.color = '';
  waveformLabel.style.display = 'block';
  outputCard.classList.remove('visible');
  outputCard.style.display = 'none';
});

// --- Generation Simulation ---
btnGenerate.addEventListener('click', () => {
  outputCard.classList.add('visible');
  outputCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  audioPlayer.classList.remove('visible');

  const genre = document.querySelector('.genre-chip.active').textContent;
  const mood = document.getElementById('moodSelect').value;
  const dur = document.getElementById('durSelect').value;

  // Update track meta
  trackMeta.textContent = `${genre} · ${mood} · ${dur}`;
  document.getElementById('trackName').textContent = `HumTrack_${Date.now().toString(36).toUpperCase()}`;

  const steps = ['step1','step2','step3','step4','step5'];
  const durations = [1800, 2200, 2400, 2800, 2000];
  let currentStep = 0;
  progressBar.style.width = '0%';
  steps.forEach(id => {
    const el = document.getElementById(id);
    el.className = 'step';
    el.querySelector('.step-icon').textContent = '◌';
  });

  function runStep(i) {
    if (i >= steps.length) {
      progressBar.style.width = '100%';
      setTimeout(() => { audioPlayer.classList.add('visible'); }, 400);
      return;
    }
    const el = document.getElementById(steps[i]);
    el.className = 'step active';
    el.querySelector('.step-icon').innerHTML = '↻';
    progressBar.style.width = `${((i+1)/steps.length * 100)}%`;

    setTimeout(() => {
      el.className = 'step done';
      el.querySelector('.step-icon').textContent = '✓';
      runStep(i + 1);
    }, durations[i]);
  }
  runStep(0);
});

// Fake audio player
btnPlay.addEventListener('click', () => {
  isPlaying = !isPlaying;
  const playIcon = document.getElementById('playIcon');
  if (isPlaying) {
    playIcon.setAttribute('d', 'M6 19h4V5H6v14zm8-14v14h4V5h-4z');
    playIcon.setAttribute('points', '');
    playInterval = setInterval(() => {
      playProgress = Math.min(playProgress + 0.5, 100);
      seekFill.style.width = playProgress + '%';
      const total = 60;
      const elapsed = Math.round(total * playProgress / 100);
      timeDisplay.textContent = `${fmt(elapsed)} / ${fmt(total)}`;
      if (playProgress >= 100) { clearInterval(playInterval); isPlaying = false; playIcon.setAttribute('d', ''); playIcon.setAttribute('points', '5,3 19,12 5,21'); playProgress = 0; }
    }, 300);
  } else {
    clearInterval(playInterval);
    playIcon.setAttribute('d', '');
    playIcon.setAttribute('points', '5,3 19,12 5,21');
  }
});

document.getElementById('btnDownload').addEventListener('click', () => {
  if (audioBlob) {
    const url = URL.createObjectURL(audioBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'humtrack.webm';
    a.click();
  } else {
    alert('In a real integration, your generated track would download here.');
  }
});
