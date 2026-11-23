"""Use the tested GPU pipeline with the existing production camera readers."""
from __future__ import annotations

from functools import partial
import time

from ai.streams import MonitoringEngine, RESULT_TTL
from .dual_stage import DualStagePipeline, FairFrameSelector


MIN_CAMERA_INTERVAL = 0.20
POSE_INFLIGHT = 2
CPU_INFLIGHT = 2


class GpuPipelineEngine(MonitoringEngine):
    def __init__(self, detector, urls=None):
        super().__init__(detector, urls=urls)
        self.on_published = None
        self._pipeline_counts = {
            "acceptedPublications": 0,
            "rejectedPublications": 0,
            "publicationErrors": 0,
            "detectionErrors": 0,
            "poseErrors": 0,
            "notificationErrors": 0,
            "workerErrors": 0,
        }

    def _failure(self, counter, message, camera=None):
        with self.lock:
            self._pipeline_counts[counter] += 1
        if camera is None:
            print(f"[AI] {message}", flush=True)
        else:
            camera._warn(message)

    def _infer_loop(self):
        selector = FairFrameSelector(self.cameras, minimum_interval=MIN_CAMERA_INTERVAL, result_ttl=RESULT_TTL)
        try:
            make_context = getattr(self.detector, "make_detection_context", None)
            pipeline = DualStagePipeline(
                self.detector.stage, detection_context=make_context() if make_context else None,
                pose_initializer=self.detector.pose_worker_init, result_ttl=RESULT_TTL,
                max_pose_inflight=POSE_INFLIGHT, max_cpu_inflight=CPU_INFLIGHT,
            )
        except Exception:
            self._failure("workerErrors", "GPU 流水线初始化失败")
            return

        try:
            while not self.stop_event.is_set():
                selected = selector.take(time.perf_counter())
                if not selected:
                    self.stop_event.wait(0.01)
                    continue
                camera, token = selected[0]
                try:
                    stage = pipeline.submit(camera, token)
                except Exception:
                    # Keep the last accepted frame subject to the existing reader TTL.
                    self._failure("detectionErrors", "检测或阶段提交失败，后续新帧会继续尝试", camera)
                    continue
                stage.add_done_callback(partial(self._finish_stage, camera))
        except Exception:
            self._failure("workerErrors", "GPU 流水线调度已退出")
        finally:
            try:
                # Both executors finish before the inherited stop can close the model.
                pipeline.close()
            except Exception:
                self._failure("workerErrors", "GPU 流水线退出时检测到失败的阶段任务")

    def _finish_stage(self, camera, stage_future):
        try:
            stage = stage_future.result()
        except Exception:
            self._failure("poseErrors", "姿态或结果提交失败，后续新帧会继续尝试", camera)
            return
        stage.publication.add_done_callback(partial(self._finish_publication, camera))

    def _finish_publication(self, camera, publication_future):
        try:
            result = publication_future.result()
        except Exception:
            self._failure("publicationErrors", "CPU 结果发布失败，后续新帧会继续尝试", camera)
            return
        with self.lock:
            self.processed_frames += 1
            counter = "acceptedPublications" if result.accepted else "rejectedPublications"
            self._pipeline_counts[counter] += 1
        if not result.accepted:
            return

        notify = self.on_published
        if notify is not None:
            try:
                notify()
            except Exception:
                self._failure("notificationErrors", "结果已发布，网页唤醒失败", camera)

    def snapshot(self):
        snapshot = super().snapshot()
        with self.lock:
            snapshot["throughputOptimization"] = {
                "release": "gpu-pipeline-r3",
                "poseBatchSize": getattr(getattr(self.detector, "pose", None), "pose_batch_size", None),
                "minimumCameraIntervalMs": round(MIN_CAMERA_INTERVAL * 1000),
                "poseInFlightLimit": POSE_INFLIGHT,
                "cpuInFlightLimit": CPU_INFLIGHT,
                "cudaGraph": True,
                "gpuSourceReuse": True,
                "workerAlive": self.inference_thread.is_alive(),
                **self._pipeline_counts,
            }
        return snapshot
