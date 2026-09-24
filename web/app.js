/**
 * Voice AI Assistant Web Application - Robust Hybrid Engine with Voice Barge-In / Interruption
 * 1. Hardware Mic Capture (getUserMedia) with live AudioContext volume/VAD analysis.
 * 2. Active Voice Barge-In: Talking while audio is playing instantly cuts off the assistant and starts capturing user's new question.
 * 3. Fast failover Gemini + local Piper TTS pipeline with AbortController cancellation.
 * 4. Dual fallback: Web Speech API interim transcripts + local Whisper base offline transcription.
 */

document.addEventListener('DOMContentLoaded', () => {
  const chatContainer = document.getElementById('chatContainer');
  const chatForm = document.getElementById('chatForm');
  const messageInput = document.getElementById('messageInput');
  const micBtn = document.getElementById('micBtn');
  const statusBadge = document.getElementById('statusBadge');
  const statusText = document.getElementById('statusText');
  const recordingBanner = document.getElementById('recordingBanner');
  const liveTranscriptText = document.getElementById('liveTranscriptText');

  let isRecording = false;
  let isProcessing = false;
  let isPlayingAudio = false;

  let currentAudioSource = null;
  let currentAbortController = null;
  let currentPlaceholderMsg = null;

  let mediaStream = null;
  let mediaRecorder = null;
  let audioChunks = [];

  let audioCtx = null;
  let vadAudioCtx = null;
  let vadInterval = null;
  let vadAnalyser = null;
  let vadDataArray = null;

  let recognition = null;
  let liveSpokenText = '';
  let speechDetected = false;
  let silenceStartTime = null;
  let consecutiveSpeechFrames = 0;
  let noiseFloor = 0.002;
  let frameCount = 0;
  let captureStartTime = 0;

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  // Initialize Mic Stream & Continuous VAD
  async function ensureMicrophoneStream() {
    if (mediaStream && mediaStream.active) {
      return mediaStream;
    }
    try {
      console.log("[Audio] Requesting microphone access with echo cancellation...");
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });

      // Setup permanent Web Audio VAD Analyser on this stream
      vadAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const sourceNode = vadAudioCtx.createMediaStreamSource(mediaStream);
      vadAnalyser = vadAudioCtx.createAnalyser();
      vadAnalyser.fftSize = 512;
      sourceNode.connect(vadAnalyser);

      const bufferLength = vadAnalyser.frequencyBinCount;
      vadDataArray = new Float32Array(bufferLength);

      startContinuousVadLoop();
      return mediaStream;
    } catch (err) {
      console.error("[Audio] Microphone access error:", err);
      alert("Microphone access failed: " + err.message);
      return null;
    }
  }

  // Continuous VAD loop for silence detection and barge-in / interruption
  function startContinuousVadLoop() {
    if (vadInterval) return;

    vadInterval = setInterval(() => {
      if (!vadAnalyser || !vadDataArray) return;

      vadAnalyser.getFloatTimeDomainData(vadDataArray);
      let sumSq = 0;
      for (let i = 0; i < vadDataArray.length; i++) {
        sumSq += vadDataArray[i] * vadDataArray[i];
      }
      const rms = Math.sqrt(sumSq / vadDataArray.length);

      // Noise floor estimation during quiescent periods
      frameCount++;
      if (frameCount < 6) {
        noiseFloor = Math.max(0.001, (noiseFloor + rms) / 2);
        return;
      }

      // 1. VOICE BARGE-IN / INTERRUPTION MONITOR
      // If assistant is currently speaking or processing, listen for user speaking over it
      if (isPlayingAudio || isProcessing) {
        // Interruption threshold with safety margin above speaker acoustic bleed
        const bargeInThreshold = Math.max(0.016, noiseFloor * 3.2);

        if (rms > bargeInThreshold) {
          consecutiveSpeechFrames++;
          // Require 2 consecutive frames (~100ms) to prevent acoustic clicks triggering interruption
          if (consecutiveSpeechFrames >= 2) {
            console.log("[Barge-In] User voice interruption detected (RMS: " + rms.toFixed(4) + ")! Halting assistant...");
            consecutiveSpeechFrames = 0;
            triggerInterruption();
            return;
          }
        } else {
          consecutiveSpeechFrames = 0;
        }
        return;
      }

      // 2. ACTIVE USER RECORDING VAD MONITOR
      if (isRecording) {
        const speechThreshold = Math.max(0.007, noiseFloor * 2.5);
        const silenceThreshold = Math.max(0.0035, noiseFloor * 1.4);

        // Visual waveform feedback
        const bars = document.querySelectorAll('.waveform-mini .bar');
        const level = Math.min(1.0, rms * 40);
        bars.forEach((bar, idx) => {
          const height = Math.max(4, Math.round(level * 24 * (0.6 + 0.4 * Math.sin(idx + Date.now() / 150))));
          bar.style.height = `${height}px`;
        });

        if (rms > speechThreshold) {
          if (!speechDetected) {
            speechDetected = true;
            if (!liveSpokenText) {
              liveTranscriptText.textContent = 'Voice detected... Listening';
            }
          }
          silenceStartTime = null;
        } else if (rms < silenceThreshold && speechDetected) {
          if (silenceStartTime === null) {
            silenceStartTime = Date.now();
          } else if (Date.now() - silenceStartTime >= 1000) {
            // 1.0 second silence after speech -> auto-send
            console.log("[Audio] 1.0s silence detected. Auto-finalizing utterance...");
            liveTranscriptText.textContent = 'Finalizing speech with Whisper...';
            stopCaptureAndSend(false);
            return;
          }
        }

        // Safety cap: auto-finalize after 7 seconds max
        if (speechDetected && (Date.now() - captureStartTime > 7000)) {
          console.log("[Audio] Max utterance duration reached. Finalizing...");
          stopCaptureAndSend(false);
          return;
        }
      }
    }, 50);
  }

  // Stop all active audio playback and abort active HTTP pipeline requests
  function stopAllPlaybackAndProcessing(reason = "interrupted") {
    let wasActive = false;

    if (currentAudioSource) {
      try {
        currentAudioSource.stop();
      } catch (e) {}
      currentAudioSource = null;
      wasActive = true;
    }

    if (currentAbortController) {
      try {
        currentAbortController.abort();
      } catch (e) {}
      currentAbortController = null;
      wasActive = true;
    }

    if (currentPlaceholderMsg) {
      removePlaceholderMessage(currentPlaceholderMsg);
      currentPlaceholderMsg = null;
    }

    isPlayingAudio = false;
    isProcessing = false;
    return wasActive;
  }

  // Voice Barge-in trigger handler
  async function triggerInterruption() {
    stopAllPlaybackAndProcessing("barge_in");

    statusBadge.className = 'status-badge recording interrupted';
    statusText.textContent = 'Interrupted! Listening...';
    recordingBanner.classList.remove('hidden');
    liveTranscriptText.textContent = 'Interrupted! Listening to your voice...';
    micBtn.classList.add('recording');

    // Immediately start recording the new user query
    await startCapture({ isInterrupted: true });
  }

  // Mic Button Click handler
  micBtn.addEventListener('click', async () => {
    // If assistant is speaking or processing, clicking mic interrupts it immediately
    if (isPlayingAudio || isProcessing) {
      console.log("[Audio] Mic clicked during playback/processing. Interrupting...");
      triggerInterruption();
      return;
    }

    if (!isRecording) {
      await startCapture();
    } else {
      stopCaptureAndSend(false);
    }
  });

  // Start capturing audio from user
  async function startCapture(options = {}) {
    const isInterrupted = options.isInterrupted || false;
    const stream = await ensureMicrophoneStream();
    if (!stream) return;

    if (vadAudioCtx && vadAudioCtx.state === 'suspended') {
      await vadAudioCtx.resume();
    }

    audioChunks = [];
    liveSpokenText = '';
    speechDetected = isInterrupted; // If interrupted, voice is already active
    silenceStartTime = null;
    captureStartTime = Date.now();
    consecutiveSpeechFrames = 0;

    // MediaRecorder for high-quality audio recording
    try {
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        try { mediaRecorder.stop(); } catch (e) {}
      }
      const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
          ? 'audio/webm'
          : '';
      mediaRecorder = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) audioChunks.push(e.data);
      };
      mediaRecorder.start(100);
    } catch (err) {
      console.error("[Audio] MediaRecorder start error:", err);
    }

    isRecording = true;
    micBtn.classList.add('recording');
    statusBadge.className = isInterrupted ? 'status-badge recording interrupted' : 'status-badge recording';
    statusText.textContent = isInterrupted ? 'Interrupted! Listening...' : 'Listening (Whisper Base)...';
    recordingBanner.classList.remove('hidden');
    liveTranscriptText.textContent = isInterrupted ? 'Interrupted! Speak your new question...' : 'Speak naturally... Listening';
    messageInput.placeholder = 'Listening... (Auto-sends after 1s silence)';

    setupLiveSpeechRecognition();
  }

  // Setup Web Speech API for real-time word streaming
  function setupLiveSpeechRecognition() {
    if (!SpeechRecognition) return;

    try {
      if (recognition) {
        try { recognition.stop(); } catch (e) {}
      }
      recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onresult = (event) => {
        let interim = '';
        let final = '';
        for (let i = 0; i < event.results.length; ++i) {
          const item = event.results[i];
          if (item.isFinal) final += item[0].transcript + ' ';
          else interim += item[0].transcript;
        }
        const text = (final + interim).trim();
        if (text) {
          // If assistant was speaking, WebSpeech recognized words also trigger instant barge-in
          if (isPlayingAudio || isProcessing) {
            triggerInterruption();
          }

          liveSpokenText = text;
          speechDetected = true;
          liveTranscriptText.textContent = text;
          messageInput.value = text;
          silenceStartTime = Date.now();
        }
      };

      recognition.onerror = () => {};
      recognition.onend = () => {
        if (isRecording && recognition) {
          try { recognition.start(); } catch (e) {}
        }
      };

      recognition.start();
    } catch (e) {}
  }

  // Stop capturing audio and trigger processing
  function stopCaptureAndSend(cancel = false) {
    if (!isRecording) return;
    isRecording = false;

    if (recognition) {
      try { recognition.stop(); } catch (e) {}
      recognition = null;
    }

    micBtn.classList.remove('recording');
    statusBadge.className = 'status-badge';
    recordingBanner.classList.add('hidden');

    if (cancel) {
      statusText.textContent = 'Ready';
      messageInput.value = '';
      messageInput.placeholder = 'Type your message or click microphone...';
      return;
    }

    statusText.textContent = 'Transcribing with Whisper...';

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.onstop = async () => {
        let textToSend = (liveSpokenText || '').trim();

        // If Web Speech didn't supply text (e.g., in Thorium or Firefox), send audio chunk to Whisper
        if (!textToSend && audioChunks.length > 0) {
          try {
            const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
            const resp = await fetch('/transcribe', {
              method: 'POST',
              body: audioBlob
            });
            if (resp.ok) {
              const data = await resp.json();
              textToSend = (data.text || '').trim();
            }
          } catch (err) {
            console.error("[Transcribe Error]", err);
          }
        }

        if (textToSend.length > 0) {
          console.log("[Pipeline] Spoken text:", textToSend);
          await handleTextPrompt(textToSend);
        } else {
          appendSystemNotice("No speech detected.");
          statusBadge.className = 'status-badge';
          statusText.textContent = 'Ready';
          messageInput.placeholder = 'Type your message or click microphone...';
        }
      };
      mediaRecorder.stop();
    }
  }

  // Fast text route: /generate (Gemini 2.5) -> /synthesize (Piper)
  async function handleTextPrompt(text) {
    // If anything active, halt it cleanly
    stopAllPlaybackAndProcessing("new_prompt");

    isProcessing = true;
    currentAbortController = new AbortController();
    const signal = currentAbortController.signal;

    appendMessage({ sender: 'user', name: 'You', text: text, time: getCurrentTime() });
    messageInput.value = '';
    statusBadge.className = 'status-badge';
    statusText.textContent = 'Gemini Thinking...';
    currentPlaceholderMsg = appendPlaceholderMessage();

    try {
      const genRes = await fetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text }),
        signal: signal
      });

      if (!genRes.ok) throw new Error(`Gemini Error (HTTP ${genRes.status})`);
      const genData = await genRes.json();
      const aiResponse = genData.response || 'No response received.';

      if (currentPlaceholderMsg) {
        removePlaceholderMessage(currentPlaceholderMsg);
        currentPlaceholderMsg = null;
      }

      appendMessage({ sender: 'assistant', name: 'Gemini Voice AI', text: aiResponse, time: getCurrentTime() });
      statusText.textContent = 'Speaking (Piper TTS)...';

      const synRes = await fetch('/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: aiResponse }),
        signal: signal
      });

      if (synRes.ok && synRes.status !== 204) {
        const sr = parseInt(synRes.headers.get('X-Sample-Rate') || '22050', 10);
        const buf = await synRes.arrayBuffer();
        if (buf && buf.byteLength > 0) {
          isProcessing = false; // Transition to playing state
          await playFloat32Audio(buf, sr);
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log("[Pipeline] Processing aborted by user interruption.");
      } else {
        console.error("[Pipeline Error]", err);
        appendSystemNotice("Error: " + err.message);
      }
      if (currentPlaceholderMsg) {
        removePlaceholderMessage(currentPlaceholderMsg);
        currentPlaceholderMsg = null;
      }
    } finally {
      if (!isPlayingAudio && !isRecording) {
        isProcessing = false;
        statusBadge.className = 'status-badge';
        statusText.textContent = 'Ready';
        messageInput.placeholder = 'Type your message or click microphone...';
      }
    }
  }

  // Audio Playback via Web Audio API with full interruption support
  async function playFloat32Audio(arrayBuffer, sampleRate) {
    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
      }
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }

      const float32Array = new Float32Array(arrayBuffer);
      const audioBuffer = audioCtx.createBuffer(1, float32Array.length, sampleRate);
      audioBuffer.getChannelData(0).set(float32Array);

      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);

      currentAudioSource = source;
      isPlayingAudio = true;
      statusBadge.className = 'status-badge';
      statusText.textContent = 'Speaking (Piper TTS)...';

      return new Promise((resolve) => {
        source.onended = () => {
          if (currentAudioSource === source) {
            currentAudioSource = null;
            isPlayingAudio = false;
            statusBadge.className = 'status-badge';
            statusText.textContent = 'Ready';
          }
          resolve();
        };
        source.start(0);
      });
    } catch (e) {
      console.error("[Audio Playback Error]", e);
      isPlayingAudio = false;
      currentAudioSource = null;
    }
  }

  // Form text submit
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (isPlayingAudio || isProcessing) {
      stopAllPlaybackAndProcessing("form_submit");
    }
    if (isRecording) {
      stopCaptureAndSend(false);
      return;
    }
    const text = messageInput.value.trim();
    if (text) {
      await handleTextPrompt(text);
    }
  });

  function appendMessage({ sender, name, text, time }) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}-message`;
    const avatarText = sender === 'user' ? 'U' : 'AI';

    msgDiv.innerHTML = `
      <div class="avatar ${sender}-avatar">${avatarText}</div>
      <div class="message-content">
        <div class="sender-name">${escapeHtml(name)}</div>
        <div class="message-text">${escapeHtml(text)}</div>
        <span class="timestamp">${time}</span>
      </div>
    `;
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  function appendPlaceholderMessage() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant-message placeholder-msg';
    msgDiv.innerHTML = `
      <div class="avatar assistant-avatar">AI</div>
      <div class="message-content">
        <div class="sender-name">Gemini Voice AI</div>
        <div class="typing-indicator">
          <span></span><span></span><span></span>
        </div>
      </div>
    `;
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return msgDiv;
  }

  function removePlaceholderMessage(elem) {
    if (elem && elem.parentNode) {
      elem.parentNode.removeChild(elem);
    }
  }

  function appendSystemNotice(text) {
    const notice = document.createElement('div');
    notice.className = 'system-notice';
    notice.style.textAlign = 'center';
    notice.style.fontSize = '0.8rem';
    notice.style.color = '#94a3b8';
    notice.style.margin = '8px 0';
    notice.textContent = text;
    chatContainer.appendChild(notice);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  function getCurrentTime() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function escapeHtml(str) {
    return (str || '').replace(/[&<>"']/g, (m) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }[m]));
  }
});
