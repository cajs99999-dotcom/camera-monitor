/*
 * 浏览器接收端的视频统计。这里只读取 RTCRtpReceiver.getStats()，
 * 不控制播放，也不访问 MediaMTX Reader 的内部 PeerConnection。
 */
(() => {
  "use strict";

  const empty = () => ({ width: null, height: null, fps: null, mbps: null, codec: null,
    jitterBufferMs: null, decodeMs: null, packetsLostDelta: null, framesDroppedDelta: null,
    freezeCountDelta: null, decoderImplementation: null, powerEfficientDecoder: null, iceRoute: null });
  const nonnegative = value => Number.isFinite(value) && value >= 0 ? value : null;
  const positive = value => Number.isFinite(value) && value > 0 ? value : null;

  class StreamMetrics {
    constructor() {
      this._revision = 0;
      this.reset();
    }

    // 断线、手动重连或离开页面时调用；失效中的旧请求不能恢复旧基线。
    reset() {
      this._revision += 1;
      this._receiver = null;
      this._baseline = null;
    }

    async sample(receiver) {
      if (!receiver || typeof receiver.getStats !== "function") {
        this.reset();
        return empty();
      }
      if (receiver !== this._receiver) {
        this.reset();
        this._receiver = receiver;
      }
      const revision = this._revision;

      try {
        const report = await receiver.getStats();
        if (revision !== this._revision) return empty();
        if (!report || typeof report.values !== "function") {
          this._baseline = null;
          return empty();
        }

        const entries = [...report.values()];
        const lookup = new Map(entries.map(entry => [entry.id, entry]));
        const candidates = entries.filter(entry => {
          if (entry.type !== "inbound-rtp" || entry.isRemote) return false;
          if ((entry.kind || entry.mediaType) !== "video") return false;
          const mime = lookup.get(entry.codecId)?.mimeType?.toLowerCase();
          // RTX/FEC 是辅助数据，不能当成一条正常视频来显示码率和帧率。
          return !["video/rtx", "video/red", "video/ulpfec", "video/flexfec-03"].includes(mime);
        });
        if (!candidates.length) {
          this._baseline = null;
          return empty();
        }

        // 同一个 receiver 通常只有一路视频；有多条时优先最近收到数据的那条。
        candidates.sort((a, b) =>
          (nonnegative(b.lastPacketReceivedTimestamp) ?? -1) -
          (nonnegative(a.lastPacketReceivedTimestamp) ?? -1));
        const inbound = candidates[0];
        const transport = lookup.get(inbound.transportId) || entries.find(entry => entry.type === "transport" && entry.selectedCandidatePairId);
        const pair = lookup.get(transport?.selectedCandidatePairId);
        const remote = lookup.get(pair?.remoteCandidateId);
        const local = lookup.get(pair?.localCandidateId);
        const address = remote?.address || remote?.ip;
        const route = address ? `${remote.protocol || local?.protocol || "unknown"}.remote_${address}:${remote.port || 0}.local_${local?.candidateType || "unknown"}_${local?.address || local?.ip || "hidden"}`
          .replace(/[^A-Za-z0-9_.:-]/g, "_").slice(0, 96) : null;
        const mimeType = lookup.get(inbound.codecId)?.mimeType;
        const codec = typeof mimeType === "string" && /^video\/[\w.+-]+$/i.test(mimeType)
          ? mimeType.slice(mimeType.indexOf("/") + 1).toUpperCase() : null;
        const result = {
          ...empty(),
          width: positive(inbound.frameWidth),
          height: positive(inbound.frameHeight),
          fps: null,
          mbps: null,
          codec,
          decoderImplementation: typeof inbound.decoderImplementation === "string" ? inbound.decoderImplementation.slice(0, 96) : null,
          powerEfficientDecoder: typeof inbound.powerEfficientDecoder === "boolean" ? inbound.powerEfficientDecoder : null,
          iceRoute: route,
        };
        const current = {
          id: inbound.id ?? null,
          ssrc: inbound.ssrc ?? null,
          timestamp: nonnegative(inbound.timestamp),
          frames: nonnegative(inbound.framesDecoded),
          bytes: nonnegative(inbound.bytesReceived),
          jitterDelay: nonnegative(inbound.jitterBufferDelay),
          jitterEmitted: nonnegative(inbound.jitterBufferEmittedCount),
          decodeTime: nonnegative(inbound.totalDecodeTime),
          packetsLost: finiteNumber(inbound.packetsLost),
          dropped: nonnegative(inbound.framesDropped),
          freezes: nonnegative(inbound.freezeCount),
        };
        const previous = this._baseline;
        this._baseline = current.timestamp === null ? null : current;

        // 首次、换流或累计计数回退时，只建立基线，不把累计值当成当前速率。
        if (!previous || current.timestamp === null ||
            previous.id !== current.id || previous.ssrc !== current.ssrc ||
            (current.frames !== null && previous.frames !== null && current.frames < previous.frames) ||
            (current.bytes !== null && previous.bytes !== null && current.bytes < previous.bytes)) {
          return result;
        }
        const seconds = (current.timestamp - previous.timestamp) / 1000;
        if (!Number.isFinite(seconds) || seconds <= 0) return result;

        if (current.frames !== null && previous.frames !== null) {
          const fps = (current.frames - previous.frames) / seconds;
          if (Number.isFinite(fps) && fps >= 0) result.fps = fps;
        }
        if (current.bytes !== null && previous.bytes !== null) {
          const mbps = (current.bytes - previous.bytes) * 8 / seconds / 1e6;
          if (Number.isFinite(mbps) && mbps >= 0) result.mbps = mbps;
        }
        const delta = key => current[key] !== null && previous[key] !== null && current[key] >= previous[key]
          ? current[key] - previous[key] : null;
        const emitted = delta("jitterEmitted"), delay = delta("jitterDelay");
        const frames = delta("frames"), decodeTime = delta("decodeTime");
        if (emitted > 0 && delay !== null) result.jitterBufferMs = delay * 1000 / emitted;
        if (frames > 0 && decodeTime !== null) result.decodeMs = decodeTime * 1000 / frames;
        result.packetsLostDelta = delta("packetsLost");
        result.framesDroppedDelta = delta("dropped");
        result.freezeCountDelta = delta("freezes");
        return result;
      } catch (_) {
        // 统计读取失败不改变视频连接；清基线后等待下一次有效采样。
        if (revision === this._revision) this._baseline = null;
        return empty();
      }
    }

    static formatDuration(seconds) {
      if (!Number.isFinite(seconds) || seconds < 0) return "--";
      const total = Math.floor(seconds);
      const pad = value => String(value).padStart(2, "0");
      return `${pad(Math.floor(total / 3600))}:${pad(Math.floor(total / 60) % 60)}:${pad(total % 60)}`;
    }
  }

  function finiteNumber(value) { return Number.isFinite(value) ? value : null; }

  window.StreamMetrics = StreamMetrics;
})();
