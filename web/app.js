/**
 * Voice AI Assistant Interface Script
 * Automatic microphone recording & Voice AI Pipeline Integration:
 * Browser Mic -> Python Backend (Silero VAD -> Whisper STT -> Gemini -> Piper TTS) -> Browser Speaker
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
  let mediaRecorder = null;
  let audioChunks = [];
  let mediaStream = null;
  let audioCtx = null;
  let isProcessing = false;
  let autoStopTimer = null;

  let speechCheckInterval = null;
  let vadAudioCtx = null;

  // Single click starts automatic mic capture and pipeline processing
  micBtn.addEventListener('click', async () => {
    if (!isRecording) {
      await startAutomaticRecording();
    } else {
      stopRecordingAndSend();
    }
  });

  async function startAutomaticRecording() {
    try {
      console.log("[MIC] MIC STARTED");
      audioChunks = [];
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Use native MediaRecorder
      const options = MediaRecorder.isTypeSupported('audio/webm') ? { mimeType: 'audio/webm' } : {};
      mediaRecorder = new MediaRecorder(mediaStream, options);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.start(100);
      isRecording = true;

      micBtn.classList.add('recording');
      statusBadge.classList.add('recording');
      statusText.textContent = 'Listening...';
      recordingBanner.classList.remove('hidden');
      liveTranscriptText.textContent = 'Listening... Speak now';
      messageInput.placeholder = 'Listening for speech... (Auto-processing on silence)';

      // Set up Web Audio API Analyser for real-time speech / silence detection
      vadAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const sourceNode = vadAudioCtx.createMediaStreamSource(mediaStream);
      const analyser = vadAudioCtx.createAnalyser();
      analyser.fftSize = 512;
      sourceNode.connect(analyser);

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Float32Array(bufferLength);

      let speechStarted = false;
      let silenceStartTime = null;
      let speechStartTime = Date.now();

      speechCheckInterval = setInterval(() => {
        if (!isRecording) return;

        analyser.getFloatTimeDomainData(dataArray);
        let sumSq = 0;
        for (let i = 0; i < dataArray.length; i++) {
          sumSq += dataArray[i] * dataArray[i];
        }
        const rms = Math.sqrt(sumSq / dataArray.length);

        const speechThreshold = 0.02; // Threshold for speech detection
        const elapsedSinceStart = Date.now() - speechStartTime;

        if (rms > speechThreshold) {
          if (!speechStarted) {
            speechStarted = true;
            liveTranscriptText.textContent = 'Speech detected... Speaking...';
          }
          silenceStartTime = null;
        } else {
          if (speechStarted) {
            if (silenceStartTime === null) {
              silenceStartTime = Date.now();
            } else if (Date.now() - silenceStartTime >= 700) { // 700ms of silence after speech
              console.log("[MIC] Silence detected after speech. Finalizing utterance...");
              stopRecordingAndSend();
              return;
            }
          } else if (elapsedSinceStart > 10000) {
            // Max silence timeout if user doesn't speak within 10s
            console.log("[MIC] Timeout reached with no speech. Finalizing recording...");
            stopRecordingAndSend();
            return;
          }
        }
      }, 50);

    } catch (err) {
      console.error("Microphone access error:", err);
      alert("Microphone access failed: " + err.message);
    }
  }

  function stopRecordingAndSend() {
    if (!isRecording || !mediaRecorder) return;
    isRecording = false;

    if (speechCheckInterval) {
      clearInterval(speechCheckInterval);
      speechCheckInterval = null;
    }
    if (vadAudioCtx) {
      vadAudioCtx.close().catch(() => {});
      vadAudioCtx = null;
    }

    micBtn.classList.remove('recording');
    statusBadge.classList.remove('recording');
    statusText.textContent = 'Processing Pipeline...';
    recordingBanner.classList.add('hidden');
    messageInput.placeholder = 'Processing audio (Silero VAD -> Whisper -> Gemini -> Piper)...';

    mediaRecorder.onstop = async () => {
      if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
      }

      const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
      await processAudioWithBackend(audioBlob);
    };

    if (mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
  }

  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (isRecording) {
      stopRecordingAndSend();
    } else {
      const text = messageInput.value.trim();
      if (text) {
        await handleTextMessageSubmit(text);
      }
    }
  });

  async function processAudioWithBackend(audioBlob) {
    if (isProcessing) return;
    isProcessing = true;

    const placeholderMsg = appendPlaceholderMessage();

    try {
      const response = await fetch('/process_speech', {
        method: 'POST',
        headers: {
          'Content-Type': audioBlob.type || 'audio/webm'
        },
        body: audioBlob
      });

      removePlaceholderMessage(placeholderMsg);

      if (response.status === 204) {
        statusText.textContent = 'Ready';
        messageInput.placeholder = 'Type your message or click microphone...';
        appendSystemNotice("No speech or silence detected by VAD.");
        isProcessing = false;
        return;
      }

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const rawTrans = response.headers.get('X-Transcription') || '';
      const rawResp = response.headers.get('X-Response-Text') || '';
      const sampleRateHeader = response.headers.get('X-Sample-Rate') || '16000';

      const transcription = rawTrans ? decodeURIComponent(rawTrans) : 'Speech Input';
      const aiResponse = rawResp ? decodeURIComponent(rawResp) : 'No response text.';
      const sampleRate = parseInt(sampleRateHeader, 10);

      // 1. Display User Message from Whisper STT
      appendMessage({
        sender: 'user',
        name: 'You',
        text: transcription,
        time: getCurrentTime()
      });

      // 2. Display Gemini Response Message
      appendMessage({
        sender: 'assistant',
        name: 'Gemini Voice AI',
        text: aiResponse,
        time: getCurrentTime()
      });

      // 3. Play Piper Audio Response
      const audioArrayBuffer = await response.arrayBuffer();
      if (audioArrayBuffer && audioArrayBuffer.byteLength > 0) {
        await playFloat32Audio(audioArrayBuffer, sampleRate);
      } else {
        statusText.textContent = 'Ready';
        messageInput.placeholder = 'Type your message or click microphone...';
        isProcessing = false;
      }

    } catch (err) {
      console.error("Backend error:", err);
      removePlaceholderMessage(placeholderMsg);
      statusText.textContent = 'Ready';
      messageInput.placeholder = 'Type your message or click microphone...';
      appendSystemNotice("Pipeline Error: " + err.message);
      isProcessing = false;
    }
  }

  async function handleTextMessageSubmit(text) {
    appendMessage({
      sender: 'user',
      name: 'You',
      text: text,
      time: getCurrentTime()
    });

    messageInput.value = '';
    statusText.textContent = 'Gemini Thinking...';
    const placeholderMsg = appendPlaceholderMessage();

    try {
      const genRes = await fetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text })
      });
      const genData = await genRes.json();
      const aiResponse = genData.response || '';

      removePlaceholderMessage(placeholderMsg);

      appendMessage({
        sender: 'assistant',
        name: 'Gemini Voice AI',
        text: aiResponse,
        time: getCurrentTime()
      });

      const synRes = await fetch('/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: aiResponse })
      });

      if (synRes.ok) {
        const srHeader = synRes.headers.get('X-Sample-Rate') || '16000';
        const sampleRate = parseInt(srHeader, 10);
        const audioArrayBuffer = await synRes.arrayBuffer();
        if (audioArrayBuffer && audioArrayBuffer.byteLength > 0) {
          await playFloat32Audio(audioArrayBuffer, sampleRate);
        } else {
          statusText.textContent = 'Ready';
        }
      } else {
        statusText.textContent = 'Ready';
      }

    } catch (err) {
      console.error("Text submission error:", err);
      removePlaceholderMessage(placeholderMsg);
      statusText.textContent = 'Ready';
    }
  }

  async function playFloat32Audio(arrayBuffer, sampleRate) {
    statusText.textContent = 'Speaking...';
    messageInput.placeholder = 'Assistant is speaking...';

    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: sampleRate });
      }

      const float32Array = new Float32Array(arrayBuffer);
      const audioBuffer = audioCtx.createBuffer(1, float32Array.length, sampleRate);
      audioBuffer.getChannelData(0).set(float32Array);

      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);

      source.onended = () => {
        console.log("[AUDIO] PLAYBACK COMPLETE");
        statusText.textContent = 'Ready';
        messageInput.placeholder = 'Type your message or click microphone...';
        isProcessing = false;
      };

      source.start(0);
    } catch (e) {
      console.error("Audio playback error:", e);
      statusText.textContent = 'Ready';
      messageInput.placeholder = 'Type your message or click microphone...';
      isProcessing = false;
    }
  }

  function appendMessage({ sender, name, text, time }) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}-message`;
    const isUser = sender === 'user';
    msgDiv.innerHTML = `
      <div class="avatar ${isUser ? 'user-avatar' : 'assistant-avatar'}">
        ${isUser ? 'YOU' : 'AI'}
      </div>
      <div class="message-content">
        <div class="sender-name">${name}</div>
        <div class="message-text">${escapeHtml(text)}</div>
        <span class="timestamp">${time}</span>
      </div>
    `;
    chatContainer.appendChild(msgDiv);
    scrollToBottom();
  }

  function appendPlaceholderMessage() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant-message placeholder-message';
    msgDiv.innerHTML = `
      <div class="avatar assistant-avatar">AI</div>
      <div class="message-content">
        <div class="sender-name">Gemini Voice AI</div>
        <div class="message-text placeholder-loading">
          <span class="loading-dot"></span>
          <span class="loading-dot"></span>
          <span class="loading-dot"></span>
        </div>
      </div>
    `;
    chatContainer.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
  }

  function removePlaceholderMessage(elem) {
    if (elem && elem.parentNode) {
      elem.parentNode.removeChild(elem);
    }
  }

  function appendSystemNotice(text) {
    const noticeDiv = document.createElement('div');
    noticeDiv.style.textAlign = 'center';
    noticeDiv.style.fontSize = '0.8rem';
    noticeDiv.style.color = '#94a3b8';
    noticeDiv.style.margin = '8px 0';
    noticeDiv.textContent = text;
    chatContainer.appendChild(noticeDiv);
    scrollToBottom();
  }

  function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
});
