import * as THREE from 'three';

/** Orbit to a camera's side of the room; this is an external 3D inspection view. */
export function createCameraNavigation({ camera, controls, classroom, cameraModels, invalidate, onStart,
  duration = 1100, reducedMotion = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches || false }) {
  const elevation = THREE.MathUtils.degToRad(42);
  let animation = null, activeCamera = null;
  function preset(cameraId) {
    const entry = cameraModels.entries.get(cameraId);
    if (!entry) return null;
    classroom.root.updateMatrixWorld(true);
    const origin = entry.root.getWorldPosition(new THREE.Vector3());
    const aim = classroom.root.localToWorld(entry.target.clone());
    const back = origin.sub(aim); back.y = 0;
    if (back.lengthSq() < 1e-8) back.set(0, 0, 1);
    back.normalize().multiplyScalar(Math.cos(elevation)); back.y = Math.sin(elevation);
    const { minX, maxX, minZ, maxZ } = classroom.bounds;
    const target = classroom.root.localToWorld(new THREE.Vector3((minX + maxX) / 2, .85, (minZ + maxZ) / 2));
    const distance = THREE.MathUtils.clamp(controls.getDistance(), controls.minDistance, controls.maxDistance);
    return { cameraId, target, position: target.clone().addScaledVector(back, distance), distance };
  }
  function cancel() {
    if (animation) controls.enabled = animation.controlsEnabled;
    animation = null; activeCamera = null;
  }
  function drainDamping() {
    // OrbitControls has no public reset for residual angular/pan deltas. Consume
    // those deltas once, then restore the visible pose before starting the arc.
    const position = camera.position.clone(), target = controls.target.clone(), damping = controls.enableDamping;
    controls.enableDamping = false; controls.update(); controls.enableDamping = damping;
    camera.position.copy(position); controls.target.copy(target); controls.update();
  }
  function goTo(cameraId) {
    const destination = preset(cameraId);
    if (!destination) return false;
    cancel(); drainDamping();
    const start = new THREE.Spherical().setFromVector3(camera.position.clone().sub(controls.target));
    const end = new THREE.Spherical().setFromVector3(destination.position.clone().sub(destination.target));
    const deltaTheta = Math.atan2(Math.sin(end.theta - start.theta), Math.cos(end.theta - start.theta));
    animation = { cameraId, startedAt: performance.now(), duration: reducedMotion ? 0 : duration,
      startTarget: controls.target.clone(), endTarget: destination.target, start, end, deltaTheta,
      controlsEnabled: controls.enabled };
    activeCamera = cameraId; controls.enabled = false;
    onStart?.(cameraId); invalidate(); return true;
  }
  function update(now = performance.now()) {
    if (!animation) return false;
    const a = animation;
    const t = a.duration ? THREE.MathUtils.clamp((now - a.startedAt) / a.duration, 0, 1) : 1;
    const eased = t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    controls.target.lerpVectors(a.startTarget, a.endTarget, eased);
    const spherical = new THREE.Spherical(
      THREE.MathUtils.lerp(a.start.radius, a.end.radius, eased),
      THREE.MathUtils.lerp(a.start.phi, a.end.phi, eased),
      a.start.theta + a.deltaTheta * eased,
    );
    camera.position.setFromSpherical(spherical).add(controls.target);
    camera.lookAt(controls.target);
    if (t === 1) { controls.enabled = a.controlsEnabled; animation = null; }
    return Boolean(animation);
  }
  function getState() {
    return { activeCamera, transitioning: Boolean(animation), destinationCamera: animation?.cameraId || null,
      durationMs: reducedMotion ? 0 : duration, elevationDegrees: 42, mode: 'external_3d_view_from_camera_side',
      presets: [...cameraModels.entries.keys()].map(id => {
        const value = preset(id); return { cameraId: id, position: value.position.toArray(), target: value.target.toArray(), distance: value.distance };
      }) };
  }
  return { goTo, update, cancel, getState, isAnimating: () => Boolean(animation) };
}
