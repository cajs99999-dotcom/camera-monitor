/* Playback progress and bounded recovery policy. No network/model ownership. */
(() => {
  "use strict";
  const finite = value => Number.isFinite(value) ? value : null;
  const FRAME_WARNING_MS = 2000;
  const FRAME_RECOVERY_MS = 20000;
  const RETRY_OWNER_MS = 8000;
  const RECOVERY_COOLDOWN_MS = 10000;
  class PlaybackHealth {
    constructor({ startupTimeoutMs = FRAME_RECOVERY_MS } = {}) {
      this.startupTimeoutMs = Math.max(FRAME_RECOVERY_MS, finite(startupTimeoutMs) ?? FRAME_RECOVERY_MS);
      this.events = []; this.attempt = 0; this.lastRecoveryAt = -Infinity;
    }
    note(type, reason) {
      this.events.push({ type, reason, at: Date.now() });
      if (this.events.length > 20) this.events.shift();
    }
    begin(now, reason = "manual") {
      this.attempt++; this.startedAt = now; this.lastRecoveryAt = now;
      this.lastRecoveryReason = reason; this.lastFrameAt = null;
      this.mediaTime = null; this.presentedFrames = null; this.frameMetadata = {};
      this.lastReaderErrorAt = null; this.readerRetrying = false; this.terminalAt = null;
      this.note("connect", reason);
    }
    track(now) {
      this.startedAt = now; this.lastFrameAt = null; this.mediaTime = null;
      this.presentedFrames = null; this.frameMetadata = {};
      this.readerRetrying = false; this.terminalAt = null; this.note("track", "video_track");
    }
    frame(now, currentTime, metadata = null) {
      const time = finite(metadata?.mediaTime) ?? finite(currentTime);
      const frames = finite(metadata?.presentedFrames);
      const advanced = (time !== null && time !== this.mediaTime) ||
        (frames !== null && frames !== this.presentedFrames);
      if (!advanced) return false;
      this.mediaTime = time; this.presentedFrames = frames;
      this.frameMetadata = {};
      for (const key of ["mediaTime", "captureTime", "receiveTime", "rtpTimestamp", "expectedDisplayTime", "presentationTime", "processingDuration"]) {
        if (Number.isFinite(metadata?.[key])) this.frameMetadata[key] = metadata[key];
      }
      this.lastFrameAt = now; this.readerRetrying = false; this.terminalAt = null;
      return true;
    }
    fail(now, reason, retrying = false) {
      this.lastReaderErrorAt = now; this.readerRetrying = retrying;
      this.terminalAt = retrying ? null : now; this.note("error", reason);
    }
    resume(now) { this.resumeGraceUntil = now + 5000; }
    decide(now, { visible, status, consumerFrameAt = null }) {
      if (!visible || status === "paused") return { recover: false, buffering: false };
      if (now < (this.resumeGraceUntil ?? -Infinity)) return { recover: false, buffering: false };
      const frameAt = Math.max(this.lastFrameAt ?? -Infinity, finite(consumerFrameAt) ?? -Infinity);
      const hasFrame = Number.isFinite(frameAt);
      const age = hasFrame ? now - frameAt : now - this.startedAt;
      const buffering = hasFrame && age >= FRAME_WARNING_MS;
      // A child view is decoding this same stream. A parent element error must
      // not tear down a source which is still producing presented child frames.
      const freshConsumer = finite(consumerFrameAt) !== null &&
        now >= consumerFrameAt && now - consumerFrameAt < FRAME_WARNING_MS;
      if (freshConsumer) return { recover: false, buffering: false };
      // The official Reader owns its active retry timer. A parallel timer here
      // must not repeatedly tear down a legitimate retry/ICE negotiation.
      if (this.readerRetrying && now - this.lastReaderErrorAt < RETRY_OWNER_MS)
        return { recover: false, buffering };
      if (now - this.lastRecoveryAt < RECOVERY_COOLDOWN_MS) return { recover: false, buffering };
      if (this.terminalAt !== null && now - this.terminalAt >= 2000)
        return { recover: true, buffering, reason: "terminal_playback_error" };
      if (age >= (hasFrame ? FRAME_RECOVERY_MS : this.startupTimeoutMs))
        return { recover: true, buffering, reason: hasFrame ? "video_frame_timeout" : "connection_timeout" };
      return { recover: false, buffering };
    }
    snapshot(now) {
      return { mediaTime: this.mediaTime, presentedFrames: this.presentedFrames,
        lastFrameAgeMs: this.lastFrameAt === null ? null : Math.max(0, now - this.lastFrameAt),
        frameMetadata: { ...this.frameMetadata }, lastRecoveryReason: this.lastRecoveryReason,
        events: this.events.slice(-8).map(event => ({ ...event })) };
    }
  }
  if (typeof module === "object" && module.exports) module.exports = { PlaybackHealth };
  if (typeof window !== "undefined") window.PlaybackHealth = PlaybackHealth;
})();
