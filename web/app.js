/**
 * Voice AI Assistant Web Application - Continuous Conversational Assistant Engine
 * 1. Single Microphone Click: Persistent microphone MediaStream & AudioContext VAD.
 * 2. Automatic Speech Detection: Continuous monitoring with ~700ms VAD silence finalization.
 * 3. Continuous Conversation Loop: LISTENING -> USER_SPEAKING -> PROCESSING -> AI_SPEAKING -> LISTENING.
 * 4. Real-time Voice Barge-In / Interruption: User voice instantly cuts off AI audio and starts capturing new query.
 * 5. State Machine & Turn Cancellation: Epoch / turn ID validation prevents stale responses.
 * 6. Explicit Logging: Standardized tag logging ([MIC], [VAD], [STT], [LLM], [TTS], [AUDIO], [INTERRUPT]).
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
  const newChatBtn = document.getElementById('newChatBtn');

  // Session ID for in-memory multi-turn history (fresh on tab load)
  let currentSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 9);

  // Configurable silence threshold for utterance finalization (ms)
  const SILENCE_DURATION_MS = 700;

  // State Machine definitions
  const States = {
    IDLE: 'IDLE',
    LISTENING: 'LISTENING',
    USER_SPEAKING: 'USER_SPEAKING',
    PROCESSING: 'PROCESSING',
    AI_SPEAKING: 'AI_SPEAKING',
    INTERRUPTED: 'INTERRUPTED'
  };

  let currentState = States.IDLE;
  let isContinuousMode = false;
  let currentTurnId = 0;

  const MAX_HISTORY_TURNS = 20;
  let conversationHistory = [];

  let lastUserPrompt = '';
  let lastAssistantResponseText = '';
  let lastAssistantMessageElem = null;
  let interruptedTurnContext = null;

  function clearSessionHistory() {
    conversationHistory = [];
    interruptedTurnContext = null;
    console.log("[SESSION] Conversation history cleared.");
  }
  window.clearSessionHistory = clearSessionHistory;

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
  let silenceStartTime = null;
  let consecutiveSpeechFrames = 0;
  let noiseFloor = 0.002;
  let frameCount = 0;
  let captureStartTime = 0;

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  function setState(newState, detail = '') {
    const oldState = currentState;
    currentState = newState;
    console.log(`[STATE] ${oldState} -> ${newState}${detail ? ' (' + detail + ')' : ''}`);

    switch (newState) {
      case States.IDLE:
        micBtn.classList.remove('recording');
        statusBadge.className = 'status-badge';
        statusText.textContent = 'Ready';
        recordingBanner.classList.add('hidden');
        messageInput.placeholder = 'Type your message or click microphone...';
        break;

      case States.LISTENING:
        micBtn.classList.add('recording');
        statusBadge.className = 'status-badge recording';
        statusText.textContent = 'Listening...';
        recordingBanner.classList.remove('hidden');
        liveTranscriptText.textContent = 'Speak naturally... Listening';
        messageInput.placeholder = 'Listening continuously... Speak anytime';
        break;

      case States.USER_SPEAKING:
        micBtn.classList.add('recording');
        statusBadge.className = 'status-badge recording';
        statusText.textContent = 'User Speaking...';
        recordingBanner.classList.remove('hidden');
        if (!liveSpokenText) {
          liveTranscriptText.textContent = 'Voice detected... Listening';
        }
        break;

      case States.PROCESSING:
        micBtn.classList.add('recording');
        statusBadge.className = 'status-badge recording';
        statusText.textContent = detail || 'Gemini Thinking...';
        recordingBanner.classList.add('hidden');
        break;

      case States.AI_SPEAKING:
        micBtn.classList.add('recording');
        statusBadge.className = 'status-badge recording';
        statusText.textContent = 'AI Speaking (Piper TTS)...';
        recordingBanner.classList.add('hidden');
        break;

      case States.INTERRUPTED:
        micBtn.classList.add('recording');
        statusBadge.className = 'status-badge recording interrupted';
        statusText.textContent = 'Interrupted! Listening...';
        recordingBanner.classList.remove('hidden');
        liveTranscriptText.textContent = 'Interrupted! Listening to your new question...';
        break;
    }
  }

  // Persistent Microphone MediaStream & VAD Setup
  async function ensureMicrophoneStream() {
    if (mediaStream && mediaStream.active) {
      return mediaStream;
    }
    try {
      console.log("[Audio] Requesting microphone access with noise suppression & echo cancellation...");
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });

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
      setState(States.IDLE);
      return null;
    }
  }

  // Continuous VAD Monitoring Loop
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

      // Adaptively estimate background noise floor
      frameCount++;
      if (frameCount < 6) {
        noiseFloor = Math.max(0.001, (noiseFloor + rms) / 2);
        return;
      }

      // 1. BARGE-IN MONITOR (User speaks while AI is speaking or thinking)
      if (currentState === States.AI_SPEAKING || currentState === States.PROCESSING) {
        const bargeInThreshold = Math.max(0.016, noiseFloor * 3.2);
        if (rms > bargeInThreshold) {
          consecutiveSpeechFrames++;
          if (consecutiveSpeechFrames >= 2) {
            consecutiveSpeechFrames = 0;
            console.log("[INTERRUPT] USER INTERRUPTED AI");
            triggerInterruption();
            return;
          }
        } else {
          consecutiveSpeechFrames = 0;
        }
        return;
      }

      // 2. ACTIVE SPEECH MONITORING (In LISTENING or USER_SPEAKING states)
      if (isContinuousMode && (currentState === States.LISTENING || currentState === States.USER_SPEAKING || currentState === States.INTERRUPTED)) {
        const speechThreshold = Math.max(0.007, noiseFloor * 2.5);
        const silenceThreshold = Math.max(0.0035, noiseFloor * 1.4);

        // Update visual waveform
        const bars = document.querySelectorAll('.waveform-mini .bar');
        const level = Math.min(1.0, rms * 40);
        bars.forEach((bar, idx) => {
          const height = Math.max(4, Math.round(level * 24 * (0.6 + 0.4 * Math.sin(idx + Date.now() / 150))));
          bar.style.height = `${height}px`;
        });

        if (rms > speechThreshold) {
          if (currentState !== States.USER_SPEAKING) {
            console.log("[VAD] SPEECH START");
            setState(States.USER_SPEAKING);
            startChunkRecording();
          }
          silenceStartTime = null;
        } else if (rms < silenceThreshold && currentState === States.USER_SPEAKING) {
          if (silenceStartTime === null) {
            silenceStartTime = Date.now();
          } else if (Date.now() - silenceStartTime >= SILENCE_DURATION_MS) {
            console.log(`[VAD] SPEECH END (${SILENCE_DURATION_MS}ms silence detected)`);
            silenceStartTime = null;
            finalizeUtterance();
            return;
          }
        }

        // Safety cap: auto-finalize after 7 seconds max
        if (currentState === States.USER_SPEAKING && (Date.now() - captureStartTime > 7000)) {
          console.log("[VAD] SPEECH END (7s max duration reached)");
          silenceStartTime = null;
          finalizeUtterance();
          return;
        }
      }
    }, 40);
  }

  // Start recording raw audio chunk for the current user utterance
  function startChunkRecording() {
    audioChunks = [];
    liveSpokenText = '';
    captureStartTime = Date.now();
    silenceStartTime = null;

    if (!mediaStream) return;

    try {
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        try { mediaRecorder.stop(); } catch (e) { }
      }
      const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
          ? 'audio/webm'
          : '';
      mediaRecorder = new MediaRecorder(mediaStream, mime ? { mimeType: mime } : {});
      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) audioChunks.push(e.data);
      };
      mediaRecorder.start(100);
    } catch (err) {
      console.error("[Audio] MediaRecorder creation error:", err);
    }

    setupLiveSpeechRecognition();
  }

  // Finalize the current user utterance and send through STT -> LLM -> TTS
  function finalizeUtterance() {
    if (currentState !== States.USER_SPEAKING) return;
    setState(States.PROCESSING, 'Finalizing utterance...');

    if (recognition) {
      try { recognition.stop(); } catch (e) { }
      recognition = null;
    }

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.onstop = async () => {
        let textToSend = (liveSpokenText || '').trim();
        let audioBlob = null;

        if (audioChunks.length > 0) {
          audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
        }

        audioChunks = [];
        processTurnPipeline(textToSend, audioBlob);
      };
      mediaRecorder.stop();
    } else {
      let textToSend = (liveSpokenText || '').trim();
      processTurnPipeline(textToSend, null);
    }
  }

  // Interruption trigger when user speaks over AI audio/processing
  function triggerInterruption() {
    if (currentState === States.AI_SPEAKING || currentState === States.PROCESSING) {
      interruptedTurnContext = {
        previousUserPrompt: lastUserPrompt,
        previousAssistantText: lastAssistantResponseText,
        targetAssistantElem: lastAssistantMessageElem,
        wasInterrupted: true
      };
    }
    stopAllPlaybackAndProcessing("barge_in");
    currentTurnId++; // Invalidate stale turn responses

    if (recognition) {
      try { recognition.stop(); } catch (e) { }
      recognition = null;
    }
    liveSpokenText = '';
    messageInput.value = '';

    setState(States.INTERRUPTED);

    // Immediately start recording the new user query
    console.log("[VAD] SPEECH START");
    setState(States.USER_SPEAKING, 'Interrupted voice input');
    startChunkRecording();
  }

  // Stop active playback and cancel in-flight HTTP requests
  function stopAllPlaybackAndProcessing(reason = "cancelled") {
    let wasActive = false;

    if (currentAudioSource) {
      try {
        currentAudioSource.stop();
      } catch (e) { }
      currentAudioSource = null;
      console.log("[AUDIO] PLAYBACK STOPPED");
      wasActive = true;
    }

    if (currentAbortController) {
      try {
        currentAbortController.abort();
      } catch (e) { }
      currentAbortController = null;
      wasActive = true;
    }

    if (currentPlaceholderMsg) {
      removePlaceholderMessage(currentPlaceholderMsg);
      currentPlaceholderMsg = null;
    }

    return wasActive;
  }

  // Single Microphone Click Handler (Toggle Continuous Mode ON/OFF)
  micBtn.addEventListener('click', async () => {
    // If AI is speaking or processing, clicking mic interrupts it and keeps continuous mode ON
    if (currentState === States.AI_SPEAKING || currentState === States.PROCESSING) {
      console.log("[MIC] Mic clicked during AI response. Triggering barge-in interruption...");
      triggerInterruption();
      return;
    }

    if (!isContinuousMode) {
      // Turn Continuous Mode ON
      isContinuousMode = true;
      console.log("[MIC] MICROPHONE STARTED");
      const stream = await ensureMicrophoneStream();
      if (!stream) {
        isContinuousMode = false;
        return;
      }

      if (vadAudioCtx && vadAudioCtx.state === 'suspended') {
        await vadAudioCtx.resume();
      }

      setState(States.LISTENING);
    } else {
      // Turn Continuous Mode OFF
      isContinuousMode = false;
      console.log("[MIC] MICROPHONE STOPPED");
      stopAllPlaybackAndProcessing("mic_toggle_off");
      if (recognition) {
        try { recognition.stop(); } catch (e) { }
        recognition = null;
      }
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        try { mediaRecorder.stop(); } catch (e) { }
      }
      setState(States.IDLE);
    }
  });

  // Web Speech API integration for streaming interim text
  function setupLiveSpeechRecognition() {
    if (!SpeechRecognition) return;

    try {
      if (recognition) {
        try { recognition.stop(); } catch (e) { }
        recognition = null;
      }
      liveSpokenText = '';
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
          // If AI is speaking, recognized words trigger instant barge-in
          if (currentState === States.AI_SPEAKING || currentState === States.PROCESSING) {
            console.log("[INTERRUPT] USER INTERRUPTED AI");
            triggerInterruption();
            return;
          }

          liveSpokenText = text;
          liveTranscriptText.textContent = text;
          messageInput.value = text;
          silenceStartTime = Date.now();
        }
      };

      recognition.onerror = () => { };
      recognition.onend = () => {
        if (isContinuousMode && currentState === States.USER_SPEAKING && recognition) {
          try { recognition.start(); } catch (e) { }
        }
      };

      recognition.start();
    } catch (e) { }
  }

  // Turn Execution Pipeline: Whisper STT -> Gemini -> Piper TTS -> Web Audio Playback
  async function processTurnPipeline(transcribedText, audioBlob) {
    const turnId = ++currentTurnId;
    stopAllPlaybackAndProcessing("new_turn");

    currentAbortController = new AbortController();
    const signal = currentAbortController.signal;

    let finalPrompt = (transcribedText || '').trim();

    // 1. Whisper STT if raw audio is provided and no WebSpeech text was captured
    if (!finalPrompt && audioBlob && audioBlob.size > 0) {
      setState(States.PROCESSING, 'Transcribing with Whisper...');
      try {
        const resp = await fetch('/transcribe', {
          method: 'POST',
          body: audioBlob,
          signal: signal
        });
        if (resp.ok) {
          const data = await resp.json();
          finalPrompt = (data.text || '').trim();
        }
      } catch (err) {
        if (err.name !== 'AbortError') console.error("[STT Error]", err);
      }
    }

    // Check race condition / cancellation
    if (turnId !== currentTurnId) {
      console.log(`[Turn ${turnId}] Stale request cancelled before LLM step.`);
      return;
    }

    if (!finalPrompt) {
      console.log("[VAD] Empty transcription / no speech detected.");
      appendSystemNotice("No speech detected.");
      if (isContinuousMode) {
        setState(States.LISTENING);
      } else {
        setState(States.IDLE);
      }
      return;
    }

    console.log(`[STT] TRANSCRIPTION: '${finalPrompt}'`);
    appendMessage({ sender: 'user', name: 'You', text: finalPrompt, time: getCurrentTime() });
    messageInput.value = '';

    // Save current prompt for context tracking and retrieve any active interruption context
    lastUserPrompt = finalPrompt;
    const ctxToSend = interruptedTurnContext;
    interruptedTurnContext = null;

    // 2. Gemini LLM Generation
    setState(States.PROCESSING, 'Gemini Thinking...');
    currentPlaceholderMsg = appendPlaceholderMessage();

    let aiResponse = '';
    try {
      const payload = {
        prompt: finalPrompt,
        session_id: currentSessionId
      };
      if (ctxToSend) {
        payload.interrupted_context = {
          previousUserPrompt: ctxToSend.previousUserPrompt || '',
          previousAssistantText: ctxToSend.previousAssistantText || '',
          wasInterrupted: !!ctxToSend.wasInterrupted
        };
      }

      console.log("[DEBUG] Sending /generate payload:", payload);

      const genRes = await fetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: signal
      });

      if (!genRes.ok) throw new Error(`Gemini Error (HTTP ${genRes.status})`);
      const genData = await genRes.json();
      aiResponse = genData.response || 'No response received.';
      lastAssistantResponseText = aiResponse;
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log(`[Turn ${turnId}] LLM request aborted by user interruption.`);
      } else {
        console.error("[LLM Error]", err);
        appendSystemNotice("Error: " + err.message);
      }
      if (currentPlaceholderMsg) {
        removePlaceholderMessage(currentPlaceholderMsg);
        currentPlaceholderMsg = null;
      }
      if (isContinuousMode && turnId === currentTurnId) {
        setState(States.LISTENING);
      } else if (turnId === currentTurnId) {
        setState(States.IDLE);
      }
      return;
    }

    // Check race condition
    if (turnId !== currentTurnId) {
      console.log(`[Turn ${turnId}] Response invalidated by newer user utterance.`);
      if (currentPlaceholderMsg) {
        removePlaceholderMessage(currentPlaceholderMsg);
        currentPlaceholderMsg = null;
      }
      return;
    }

    console.log(`[LLM] GEMINI RESPONSE: '${aiResponse}'`);
    if (currentPlaceholderMsg) {
      removePlaceholderMessage(currentPlaceholderMsg);
      currentPlaceholderMsg = null;
    }

    // Append new assistant message bubble for every response turn
    lastAssistantResponseText = aiResponse;
    lastAssistantMessageElem = appendMessage({ sender: 'assistant', name: 'Gemini Voice AI', text: aiResponse, time: getCurrentTime() });

    // Add user prompt and assistant response to conversation history
    conversationHistory.push({ role: 'user', content: finalPrompt });
    conversationHistory.push({ role: 'assistant', content: aiResponse });

    if (conversationHistory.length > MAX_HISTORY_TURNS) {
      conversationHistory = conversationHistory.slice(-MAX_HISTORY_TURNS);
    }

    // 3. Piper TTS Synthesis
    setState(States.PROCESSING, 'Synthesizing Piper TTS...');
    try {
      const synRes = await fetch('/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: aiResponse }),
        signal: signal
      });

      if (synRes.ok && synRes.status !== 204) {
        const sr = parseInt(synRes.headers.get('X-Sample-Rate') || '22050', 10);
        const buf = await synRes.arrayBuffer();
        if (buf && buf.byteLength > 0 && turnId === currentTurnId) {
          const sampleCount = Math.floor(buf.byteLength / 4);
          console.log(`[TTS] PIPER SYNTHESIS: ${sampleCount} samples`);
          await playFloat32Audio(buf, sr, turnId);
          return;
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log(`[Turn ${turnId}] TTS synthesis aborted by user interruption.`);
      } else {
        console.error("[TTS Error]", err);
      }
    }

    // Fallback if no audio or completed without audio playback
    if (turnId === currentTurnId) {
      if (isContinuousMode) {
        setState(States.LISTENING);
      } else {
        setState(States.IDLE);
      }
    }
  }

  // Audio Playback via Web Audio API with turn ID validation & interruption handling
  async function playFloat32Audio(arrayBuffer, sampleRate, turnId) {
    if (turnId !== currentTurnId) return;

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
      setState(States.AI_SPEAKING);
      console.log("[AUDIO] PLAYBACK START");

      return new Promise((resolve) => {
        source.onended = () => {
          if (currentAudioSource === source) {
            currentAudioSource = null;
            console.log("[AUDIO] PLAYBACK STOPPED");

            // Automatically return to listening mode if continuous mode remains ON!
            if (turnId === currentTurnId && isContinuousMode) {
              setState(States.LISTENING);
            } else if (turnId === currentTurnId) {
              setState(States.IDLE);
            }
          }
          resolve();
        };
        source.start(0);
      });
    } catch (e) {
      console.error("[Audio Playback Error]", e);
      currentAudioSource = null;
      if (turnId === currentTurnId && isContinuousMode) {
        setState(States.LISTENING);
      } else if (turnId === currentTurnId) {
        setState(States.IDLE);
      }
    }
  }

  // Manual Form Text Submit Handler
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = messageInput.value.trim();
    if (!text) return;

    if (currentState === States.AI_SPEAKING || currentState === States.PROCESSING) {
      console.log("[INTERRUPT] USER INTERRUPTED AI via text submission");
      stopAllPlaybackAndProcessing("form_submit");
    }

    messageInput.value = '';
    processTurnPipeline(text, null);
  });

  // "New Chat" Button Handler - Resets backend session and clears chat UI
  if (newChatBtn) {
    newChatBtn.addEventListener('click', async () => {
      console.log("[Session] Resetting chat session: " + currentSessionId);
      const oldSessionId = currentSessionId;
      currentSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 9);

      try {
        await fetch('/session/reset', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: oldSessionId })
        });
      } catch (e) {
        console.warn("[Session] Reset request error:", e);
      }

      stopAllPlaybackAndProcessing("new_chat");
      interruptedTurnContext = null;
      lastUserPrompt = '';
      lastAssistantResponseText = '';

      // Reset chat feed keeping clean welcome message
      chatContainer.innerHTML = `
        <div class="message assistant-message">
          <div class="avatar assistant-avatar">AI</div>
          <div class="message-content">
            <div class="sender-name">Gemini Voice AI</div>
            <div class="message-text">
              Started a new conversation session. How can I help you today?
            </div>
            <span class="timestamp">Just now</span>
          </div>
        </div>
      `;
      appendSystemNotice("Conversation history reset.");
      setState(States.IDLE);
    });
  }

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
    return msgDiv;
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
