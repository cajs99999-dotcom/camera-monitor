/*
 * Python直接读取MediaMTX的RTSP视频并持续检测，网页只订阅人体框和COCO17姿态。
 * 本文件只在透明Canvas上绘制，不取样、不上传图片，也不接管视频播放。
 */
(() => {
  "use strict";

  const RESULT_TTL = 1200;
  const STREAM_STALL_TIMEOUT = 1500;
  const MESSAGE_TIMEOUT = 5000;
  const COCO17_EDGES = Object.freeze([
    [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6],
    [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16],
  ].map(edge => Object.freeze(edge)));
  function validateKeypoints(points) {
    if (!Array.isArray(points) || points.length !== 17) return null;
    return points.map(point => {
      if (!point || ![point.x, point.y, point.score].every(Number.isFinite) ||
          point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1 ||
          (point.state !== 1 && point.state !== 2)) return null;
      // Rejected / malformed points stay absent. Never clamp them onto an image edge.
      return { x: point.x, y: point.y, score: point.score, state: point.state };
    });
  }

  function normalizeBox(box) {
    if (!box || ![box.x1, box.y1, box.x2, box.y2, box.score].every(Number.isFinite)) return null;
    // Keep the original normalized-box contract and its clipping behavior.
    const normalized = {
      x1: Math.max(0, Math.min(1, box.x1)), y1: Math.max(0, Math.min(1, box.y1)),
      x2: Math.max(0, Math.min(1, box.x2)), y2: Math.max(0, Math.min(1, box.y2)), score: box.score,
    };
    if (normalized.x2 <= normalized.x1 || normalized.y2 <= normalized.y1) return null;
    const points = validateKeypoints(box.keypoints);
    if (points !== null) normalized.keypoints = points;
    return normalized;
  }

  function resultTtlMs(packet) {
    // The server may request a stricter lifetime; it cannot weaken this page's 1.2s barrier.
    return Number.isFinite(packet?.resultTtlMs) && packet.resultTtlMs > 0
      ? Math.min(RESULT_TTL, packet.resultTtlMs) : RESULT_TTL;
  }

  function boundedCount(value, maximum) {
    return Number.isInteger(value) && value >= 0 && value <= maximum ? value : null;
  }

  function poseQuality(packet, boxes) {
    const posed = boxes.filter(box => Array.isArray(box.keypoints));
    const advertisedProcessed = boundedCount(packet.poseProcessedPeople, boxes.length);
    const processed = Math.max(posed.length, advertisedProcessed ?? 0);
    let reliable = 0, estimated = 0;
    for (const box of posed) {
      for (const point of box.keypoints) {
        if (point?.state === 2) reliable++;
        else if (point?.state === 1) estimated++;
      }
    }
    return {
      available: posed.length > 0 || advertisedProcessed !== null ||
        boundedCount(packet.reliableKeypoints, boxes.length * 17) !== null,
      processed,
      valid: boundedCount(packet.poseValidPeople, processed),
      reliable, estimated, rejected: Math.max(0, processed * 17 - reliable - estimated),
    };
  }

  function qualityTitle(packet, quality) {
    const parts = ["后台最近一次人体检测人数；遮挡可能漏检，不同摄像头分别计数"];
    if (quality.available) {
      parts.push(`姿态处理 ${quality.processed} 人；可绘可靠点 ${quality.reliable}，估计点 ${quality.estimated}，拒绝/缺失点 ${quality.rejected}`);
      if (quality.valid !== null) parts.push(`达到服务端姿态质量规则 ${quality.valid} 人（不代表完整骨架准确）`);
    } else parts.push("当前消息只有人体框，尚无姿态数据");
    const timings = [];
    if (Number.isFinite(packet.detectMs) && packet.detectMs >= 0) timings.push(`检测 ${packet.detectMs.toFixed(1)} ms`);
    if (Number.isFinite(packet.poseMs) && packet.poseMs >= 0) timings.push(`姿态 ${packet.poseMs.toFixed(1)} ms`);
    if (Number.isFinite(packet.processingFps) && packet.processingFps >= 0) timings.push(`处理 ${packet.processingFps.toFixed(1)} 次/秒`);
    if (timings.length) parts.push(`最近算法统计：${timings.join("，")}`);
    parts.push("骨架异步刷新，并非每个视频帧都分析；灰色虚线/空心点仅为低把握估计");
    return parts.join("\n");
  }

  function containTransform(width, height, videoWidth, videoHeight) {
    if (![width, height, videoWidth, videoHeight].every(value => Number.isFinite(value) && value > 0)) return null;
    const scale = Math.min(width / videoWidth, height / videoHeight);
    const contentWidth = videoWidth * scale, contentHeight = videoHeight * scale;
    return { left: (width - contentWidth) / 2, top: (height - contentHeight) / 2, contentWidth, contentHeight };
  }

  function drawSkeleton(context, points, transform) {
    const drawable = validateKeypoints(points);
    if (!drawable || !transform) return;
    const { left, top, contentWidth, contentHeight } = transform;
    const pointAt = point => [left + point.x * contentWidth, top + point.y * contentHeight];
    context.save();
    context.shadowBlur = 0;
    context.lineCap = "round";
    for (const estimated of [false, true]) {
      context.beginPath();
      let segments = 0;
      for (const [start, end] of COCO17_EDGES) {
        const a = drawable[start], b = drawable[end];
        if (!a || !b || a.state === 0 || b.state === 0) continue;
        if ((a.state === 1 || b.state === 1) !== estimated) continue;
        context.moveTo(...pointAt(a)); context.lineTo(...pointAt(b)); segments++;
      }
      if (segments) {
        context.strokeStyle = estimated ? "#a7b0ba" : "#73f0b6";
        context.lineWidth = estimated ? 1.3 : 1.8;
        context.setLineDash(estimated ? [4, 4] : []);
        context.stroke();
      }
    }
    context.setLineDash([]);
    for (const point of drawable) {
      if (!point || point.state === 0) continue;
      context.beginPath();
      context.arc(...pointAt(point), 2.5, 0, Math.PI * 2);
      if (point.state === 2) {
        context.fillStyle = "#ffe066"; context.fill();
      } else {
        context.strokeStyle = "#a7b0ba"; context.lineWidth = 1.3; context.stroke();
      }
    }
    context.restore();
  }

  function defaultDetectionUrl() {
    // 检测服务跟随网页所在的电脑，局域网访问时不再连接观看者自己的本机。
    const pageUrl = new URL(window.location.href);
    const endpoint = new URL("/ws", pageUrl.hostname ? pageUrl.origin : "http://127.0.0.1:8767");
    endpoint.protocol = pageUrl.protocol === "https:" ? "wss:" : "ws:";
    return endpoint.href;
  }

  class PersonOverlay {
    constructor({ cameras, toggle, statusElement, url = defaultDetectionUrl() }) {
      this.url = url;
      this.toggle = toggle;
      this.statusElement = statusElement;
      this.enabled = true; // 只控制本页显示，后台始终持续检测。
      this.closed = false;
      this.ready = false;
      this.socket = null;
      this.revision = 0;
      this.sessionId = null;
      this.lastSessionId = null;
      this.serverClockOffset = 0;
      this.connectStartedAt = 0;
      this.lastMessageAt = 0;
      this.clockResyncCount = 0;
      this.retryTimer = null;
      this.cameras = cameras.map(camera => ({
        ...camera, context: camera.canvas.getContext("2d"), result: null, record: null,
        lastSeenGeneration: -1, lastSeenSequence: -1, floorGeneration: -1, floorSequence: -1,
        videoGeneration: null, videoSource: null, videoHealthy: false,
        lastVideoTime: null, lastVideoAdvanceAt: performance.now(),
      }));
      this.onToggle = () => this.setEnabled(!this.enabled);
      this.onVisibility = () => {
        // 隐藏时断开结果订阅，恢复再连，防止旧消息积压；Python检测不受影响。
        this.disconnect(false);
        if (document.visibilityState !== "hidden") this.connect();
      };
      this.toggle.addEventListener("click", this.onToggle);
      document.addEventListener("visibilitychange", this.onVisibility);
      this.resizeObserver = new ResizeObserver(() => {
        for (const camera of this.cameras) {
          if (camera.usesFrameCallback) camera.pendingVideoDraw = true;
          else this.draw(camera);
        }
      });
      for (const camera of this.cameras) this.resizeObserver.observe(camera.media);
      for (const camera of this.cameras) this.bindVideoFrames(camera);
      this.timer = setInterval(() => this.tick(), 100);
      this.clearAll(false);
      this.connect();
    }

    setEnabled(enabled) {
      if (this.closed || this.enabled === enabled) return;
      this.enabled = enabled;
      this.revision++;
      this.toggle.setAttribute("aria-pressed", String(enabled));
      this.toggle.title = enabled
        ? "隐藏本页人体框、全身骨架和人数；后台继续分析" : "显示本页人体框、全身骨架和人数；后台持续分析";
      this.clearAll(true);
      this.updateStatus();
    }

    updateStatus() {
      let text = "检测服务未连接", state = "offline";
      if (!this.enabled) { text = "人体框与骨架已隐藏"; state = "off"; }
      else if (document.visibilityState === "hidden") { text = "后台检测独立运行"; state = "paused"; }
      else if (this.ready) { text = "后台人体与骨架 · 运行中"; state = "ready"; }
      else if (this.socket?.readyState === WebSocket.CONNECTING) { text = "检测服务 · 连接中"; state = "connecting"; }
      if (this.statusElement.textContent !== text) this.statusElement.textContent = text;
      this.statusElement.dataset.aiState = state;
      this.statusElement.title = this.ready
        ? "Python直接从MediaMTX取流分析；人体框和全身骨架异步刷新，并非每个视频帧都分析；低把握仅作估计，隐藏叠加不停止后台分析"
        : "在VS Code中运行项目的main.py，启动后台人体检测与姿态服务";
    }

    connect() {
      if (this.closed || this.socket || document.visibilityState === "hidden") return;
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
      let socket;
      try { socket = new WebSocket(this.url); }
      catch (_) { this.scheduleRetry(); return; }
      this.socket = socket;
      this.connectStartedAt = this.lastMessageAt = performance.now();
      this.revision++;
      this.ready = false;
      this.updateStatus();
      const isCurrent = () => !this.closed && this.socket === socket;
      socket.onmessage = event => {
        if (!isCurrent() || typeof event.data !== "string") return;
        let data;
        try { data = JSON.parse(event.data); } catch (_) { return; }
        if (!data || typeof data !== "object") return;
        if (data.type === "ready") {
          if (data.mode !== "server-rtsp" || typeof data.sessionId !== "string" ||
              !data.sessionId.length || data.sessionId.length > 128 || !Number.isFinite(data.serverTime)) {
            this.disconnect(true);
            return;
          }
          // 服务重启才重置后端序号；同一服务重订阅时保留屏障，等下一帧检测。
          if (this.lastSessionId !== data.sessionId) {
            for (const camera of this.cameras) {
              camera.lastSeenGeneration = camera.lastSeenSequence = -1;
              camera.floorGeneration = camera.floorSequence = -1;
              camera.record = null;
            }
          }
          this.lastSessionId = this.sessionId = data.sessionId;
          this.serverClockOffset = Date.now() - data.serverTime;
          this.lastMessageAt = performance.now();
          this.ready = true;
          this.clearAll(false);
          this.updateStatus();
          return;
        }
        if (!this.ready || data.type !== "detections" || data.sessionId !== this.sessionId ||
            !Array.isArray(data.cameras) || !Number.isFinite(data.serverTime)) return;
        // ageMs补上消息传输/排队时间；不比较独立RTSP与WebRTC的媒体时间轴。
        let transit = Date.now() - data.serverTime - this.serverClockOffset;
        if (transit < -1000) {
          // A delayed initial ready packet can overestimate this offset. Clear
          // current overlays and rebase, rather than closing a healthy socket.
          this.serverClockOffset += transit;
          this.clockResyncCount = (this.clockResyncCount || 0) + 1;
          this.lastMessageAt = performance.now();
          this.clearAll(true);
          return;
        }
        if (transit < 0) { this.serverClockOffset += transit; transit = 0; }
        if (transit > MESSAGE_TIMEOUT) { this.clearAll(true); return; }
        this.lastMessageAt = performance.now();
        const packets = new Map();
        for (const packet of data.cameras) {
          if (packet && typeof packet.cameraId === "string" && !packets.has(packet.cameraId)) {
            packets.set(packet.cameraId, packet);
          }
        }
        for (const camera of this.cameras) {
          this.syncVideo(camera);
          const packet = packets.get(camera.id);
          if (!packet) this.invalidate(camera);
          else this.accept(camera, packet, transit);
        }
      };
      socket.onerror = () => { if (isCurrent()) this.disconnect(true); };
      socket.onclose = () => { if (isCurrent()) this.disconnect(true); };
      socket.onopen = () => { if (isCurrent()) this.updateStatus(); };
    }

    scheduleRetry() {
      this.updateStatus();
      if (!this.closed && document.visibilityState !== "hidden" && !this.retryTimer) {
        this.retryTimer = setTimeout(() => { this.retryTimer = null; this.connect(); }, 2000);
      }
    }

    disconnect(retry) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
      const socket = this.socket;
      this.socket = null;
      this.ready = false;
      this.sessionId = null;
      this.revision++;
      if (socket) {
        socket.onopen = socket.onmessage = socket.onerror = socket.onclose = null;
        try { socket.close(); } catch (_) { /* 页面关闭时无需提示。 */ }
      }
      this.clearAll(true);
      if (retry) this.scheduleRetry();
      else this.updateStatus();
    }

    bindVideoFrames(camera) {
      if (typeof camera.video.requestVideoFrameCallback !== "function") return;
      camera.usesFrameCallback = true;
      camera.lastPresentedAt = null;
      camera.lastPresentedSource = null;
      const onFrame = () => {
        if (this.closed || !camera.usesFrameCallback) return;
        camera.lastPresentedAt = performance.now();
        camera.lastPresentedSource = camera.video.srcObject;
        camera.lastPresentedGeneration = camera.getState().generation;
        this.syncVideo(camera);
        // Repaint only a newly accepted result, on a presented video frame.
        // This is a display guard, not RTSP/WebRTC source-frame matching.
        if (camera.pendingVideoDraw) {
          camera.pendingVideoDraw = false;
          this.draw(camera);
        }
        camera.videoFrameCallback = camera.video.requestVideoFrameCallback(onFrame);
      };
      camera.videoFrameCallback = camera.video.requestVideoFrameCallback(onFrame);
    }

    unbindVideoFrames(camera) {
      camera.usesFrameCallback = false;
      if (camera.videoFrameCallback != null && camera.video.cancelVideoFrameCallback)
        camera.video.cancelVideoFrameCallback(camera.videoFrameCallback);
      camera.videoFrameCallback = null;
    }

    syncVideo(camera) {
      const video = camera.video, state = camera.getState(), now = performance.now(), time = video.currentTime;
      const changed = state.generation !== camera.videoGeneration || video.srcObject !== camera.videoSource;
      const reversed = Number.isFinite(time) && camera.lastVideoTime !== null && time < camera.lastVideoTime - 0.08;
      if (changed || reversed) {
        this.invalidate(camera);
        camera.videoGeneration = state.generation;
        camera.videoSource = video.srcObject;
        camera.lastVideoTime = Number.isFinite(time) ? time : null;
        camera.lastVideoAdvanceAt = now;
      } else if (Number.isFinite(time) && time !== camera.lastVideoTime) {
        camera.lastVideoTime = time;
        camera.lastVideoAdvanceAt = now;
      }
      const progressed = camera.usesFrameCallback
        ? camera.lastPresentedAt !== null && camera.lastPresentedSource === video.srcObject &&
          camera.lastPresentedGeneration === state.generation && now - camera.lastPresentedAt <= STREAM_STALL_TIMEOUT
        : now - camera.lastVideoAdvanceAt <= STREAM_STALL_TIMEOUT;
      const healthy = state.status === "live" && Boolean(video.srcObject) && video.readyState >= 2 &&
        !video.paused && video.videoWidth > 0 && video.videoHeight > 0 && Number.isFinite(time) &&
        progressed;
      if (healthy !== camera.videoHealthy) this.invalidate(camera);
      camera.videoHealthy = healthy;
    }

    accept(camera, packet, transit) {
      const generation = packet.generation, sequence = packet.sequence;
      if (!Number.isInteger(generation) || generation < 0 || !Number.isInteger(sequence) || sequence < 0) {
        this.invalidate(camera);
        return;
      }
      if (generation < camera.lastSeenGeneration ||
          (generation === camera.lastSeenGeneration && sequence < camera.lastSeenSequence)) return;
      camera.lastSeenGeneration = generation;
      camera.lastSeenSequence = sequence;
      if (packet.status !== "live" || !Number.isFinite(packet.ageMs) || packet.ageMs < 0 ||
          !Number.isInteger(packet.count) || packet.count < 0 || packet.count > 200 ||
          !Array.isArray(packet.boxes) || packet.boxes.length !== packet.count ||
          !Number.isFinite(packet.capturedAt) || !Number.isFinite(packet.width) || packet.width <= 0 ||
          !Number.isFinite(packet.height) || packet.height <= 0) {
        this.invalidate(camera);
        return;
      }
      const boxes = packet.boxes.map(normalizeBox).filter(Boolean);
      if (boxes.length !== packet.count) { this.invalidate(camera); return; }
      const now = performance.now(), expiresAt = now + resultTtlMs(packet) - packet.ageMs - transit;
      if (!camera.record || camera.record.generation !== generation || camera.record.sequence !== sequence) {
        camera.record = { generation, sequence, expiresAt };
      } else {
        // 同结果每200ms重复推送时，只缩短有效期，不重新获得1.2秒。
        camera.record.expiresAt = Math.min(camera.record.expiresAt, expiresAt);
      }
      if (camera.record.expiresAt <= now || !this.enabled || document.visibilityState === "hidden" ||
          !camera.videoHealthy) { this.invalidate(camera); return; }
      if (generation < camera.floorGeneration ||
          (generation === camera.floorGeneration && sequence <= camera.floorSequence)) { this.clear(camera); return; }
      const newResult = !camera.result || camera.result.generation !== generation ||
        camera.result.sequence !== sequence;
      camera.result = {
        generation, sequence,
        boxes, expiresAt: camera.record.expiresAt, revision: this.revision,
        videoGeneration: camera.videoGeneration, source: camera.videoSource,
      };
      const quality = poseQuality(packet, boxes);
      const label = quality.available ? `人数 ${boxes.length} · 可靠点 ${quality.reliable}` : `人数 ${boxes.length}`;
      if (camera.badge.textContent !== label) camera.badge.textContent = label;
      camera.badge.dataset.aiState = "ready";
      camera.badge.title = qualityTitle(packet, quality);
      if (camera.usesFrameCallback) camera.pendingVideoDraw ||= newResult;
      else this.draw(camera);
    }

    isCurrent(camera) {
      const result = camera.result;
      return result && !this.closed && this.enabled && this.ready && document.visibilityState !== "hidden" &&
        camera.videoHealthy && result.revision === this.revision &&
        result.videoGeneration === camera.videoGeneration && result.source === camera.video.srcObject &&
        performance.now() < result.expiresAt;
    }

    invalidate(camera) {
      camera.floorGeneration = camera.lastSeenGeneration;
      camera.floorSequence = camera.lastSeenSequence;
      this.clear(camera);
    }

    clear(camera) {
      camera.result = null;
      camera.pendingVideoDraw = false;
      camera.context?.setTransform(1, 0, 0, 1, 0, 0);
      camera.context?.clearRect(0, 0, camera.canvas.width, camera.canvas.height);
      camera.canvas.hidden = camera.badge.hidden = !this.enabled;
      if (camera.badge.textContent !== "人数 —") camera.badge.textContent = "人数 —";
      camera.badge.dataset.aiState = this.enabled ? "waiting" : "off";
      camera.badge.title = this.enabled ? "等待新的后台人体与骨架结果" : "本页人体框、骨架和人数已隐藏，后台继续分析";
    }

    clearAll(block) {
      for (const camera of this.cameras) {
        if (block) this.invalidate(camera);
        else this.clear(camera);
      }
    }

    tick() {
      if (this.closed) return;
      const now = performance.now();
      if (this.socket && ((!this.ready && now - this.connectStartedAt > 6000) ||
          (this.ready && now - this.lastMessageAt > MESSAGE_TIMEOUT))) this.disconnect(true);
      for (const camera of this.cameras) {
        this.syncVideo(camera);
        if (camera.result && !this.isCurrent(camera)) this.invalidate(camera);
      }
    }

    draw(camera) {
      const context = camera.context;
      if (!context) return;
      const width = camera.media.clientWidth, height = camera.media.clientHeight;
      if (!width || !height) return;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const pixelWidth = Math.round(width * ratio), pixelHeight = Math.round(height * ratio);
      if (camera.canvas.width !== pixelWidth || camera.canvas.height !== pixelHeight) {
        camera.canvas.width = pixelWidth;
        camera.canvas.height = pixelHeight;
      }
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, pixelWidth, pixelHeight);
      if (!this.isCurrent(camera)) return;
      context.setTransform(pixelWidth / width, 0, 0, pixelHeight / height, 0, 0);
      // 按object-fit:contain的真实图像区域映射，避开视频黑边。
      const transform = containTransform(width, height, camera.video.videoWidth, camera.video.videoHeight);
      if (!transform) return;
      const { left, top, contentWidth, contentHeight } = transform;
      context.lineWidth = 1.8;
      context.strokeStyle = "#73f0b6";
      context.shadowColor = "rgba(0,0,0,.8)";
      context.shadowBlur = 2;
      context.save();
      context.beginPath(); context.rect(left, top, contentWidth, contentHeight); context.clip();
      for (const box of camera.result.boxes) {
        const x = left + box.x1 * contentWidth, y = top + box.y1 * contentHeight;
        context.strokeRect(x, y,
          (box.x2 - box.x1) * contentWidth, (box.y2 - box.y1) * contentHeight);
        drawSkeleton(context, box.keypoints, transform);
      }
      context.restore();
    }

    destroy() {
      if (this.closed) return;
      this.closed = true;
      clearInterval(this.timer);
      this.disconnect(false);
      this.resizeObserver.disconnect();
      for (const camera of this.cameras) this.unbindVideoFrames(camera);
      this.toggle.removeEventListener("click", this.onToggle);
      document.removeEventListener("visibilitychange", this.onVisibility);
    }
  }

  if (typeof module === "object" && module.exports) {
    module.exports = { PersonOverlay, COCO17_EDGES, validateKeypoints, normalizeBox,
      resultTtlMs, poseQuality, qualityTitle, containTransform, drawSkeleton };
  }
  if (typeof window !== "undefined") window.PersonOverlay = PersonOverlay;
})();
