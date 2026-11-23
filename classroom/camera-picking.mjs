import * as THREE from 'three';

/** Picking observes pointer events; OrbitControls retains default handling/capture. */
export function createCameraPicking({ canvas, camera, classroom, cameraModels, onChoose = () => {}, onInteract = () => {} }) {
  if (!canvas?.addEventListener || !camera?.isCamera || !classroom?.root || !cameraModels?.entries?.values) {
    throw new TypeError('Camera picking requires a canvas, view camera, classroom and camera model entries.');
  }
  const documentTarget = canvas.ownerDocument || globalThis.document;
  const controller = new AbortController(), raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2(), sphere = new THREE.Sphere(), point = new THREE.Vector3();
  const cameraRoots = new Set([classroom.cameras, cameraModels.root].filter(Boolean));
  const entries = [...cameraModels.entries].map(([key, entry]) => {
    const id = entry.cameraId || (/^cam\d+$/.test(key) ? key : `cam${Number(String(entry.id || key).replace(/\D/g, ''))}`);
    const meshes = []; entry.body?.traverse((object) => { if (object.isMesh) meshes.push(object); });
    cameraRoots.add(entry.root);
    return { ...entry, cameraId: id, meshes };
  });
  const originalCursor = canvas.style.cursor;
  const pointers = new Set();
  const stats = { cameraRaycasts: 0, roomRaycasts: 0 };
  let disposed = false, gesture = null, lastGesture = null, hoveredCameraId = null, lastHit = null, lastRejected = null;
  function visible(object) {
    for (let node = object; node; node = node.parent) if (node.visible === false) return false;
    return true;
  }
  function inCameraTree(object) {
    for (let node = object; node; node = node.parent) if (cameraRoots.has(node)) return true;
    return false;
  }
  function blocking(material) {
    return material && material.visible !== false && !material.transparent &&
      (material.opacity ?? 1) > 0.05 && !(material.transmission > 0);
  }
  function hitMaterial(hit) {
    const material = hit.object.material;
    return Array.isArray(material) ? material[hit.face?.materialIndex || 0] : material;
  }
  function reject(reason, extra = {}) { lastHit = null; lastRejected = { reason, ...extra }; return null; }

  function pickAt(clientX, clientY, { occlusion = true } = {}) {
    if (disposed) return null;
    const rectangle = canvas.getBoundingClientRect();
    if (!Number.isFinite(clientX + clientY) || rectangle.width <= 0 || rectangle.height <= 0 ||
        clientX < rectangle.left || clientX > rectangle.right || clientY < rectangle.top || clientY > rectangle.bottom) return reject('outside_canvas');
    camera.updateWorldMatrix(true, false);
    ndc.set((clientX - rectangle.left) / rectangle.width * 2 - 1, 1 - (clientY - rectangle.top) / rectangle.height * 2);
    raycaster.setFromCamera(ndc, camera); raycaster.near = Math.max(0, camera.near || 0); raycaster.far = camera.far || Infinity;
    const candidates = [];
    for (const entry of entries) {
      if (!entry.body || !visible(entry.root) || !visible(entry.body)) continue;
      entry.body.updateWorldMatrix(true, true);
      stats.cameraRaycasts++;
      const bodyMeshes = entry.meshes.filter((mesh) => visible(mesh) &&
        (Array.isArray(mesh.material) ? mesh.material : [mesh.material]).some((material) => material?.visible !== false));
      if (!bodyMeshes.length) continue;
      const hits = raycaster.intersectObjects(bodyMeshes, false);
      const hit = hits.find((value) => hitMaterial(value)?.visible !== false);
      const centre = entry.body.getWorldPosition(new THREE.Vector3());
      if (hit) {
        candidates.push({ cameraId: entry.cameraId, id: entry.id, point: hit.point.toArray(), centre: centre.toArray(),
          distance: hit.distance, referenceDistance: hit.distance, kind: 'body' });
        continue;
      }
      // A mathematical tolerance only: no helper mesh can hide the actual scene.
      const scale = entry.body.getWorldScale(new THREE.Vector3());
      sphere.set(centre, 0.16 * Math.max(Math.abs(scale.x), Math.abs(scale.y), Math.abs(scale.z)));
      const projectionDistance = centre.clone().sub(raycaster.ray.origin).dot(raycaster.ray.direction);
      if (projectionDistance < raycaster.near || projectionDistance > raycaster.far || !raycaster.ray.intersectSphere(sphere, point)) continue;
      candidates.push({ cameraId: entry.cameraId, id: entry.id, point: point.toArray(), centre: centre.toArray(),
        distance: point.distanceTo(raycaster.ray.origin), referenceDistance: projectionDistance, kind: 'tolerance' });
    }
    if (!candidates.length) return reject('no_camera');
    candidates.sort((a, b) => Number(a.kind !== 'body') - Number(b.kind !== 'body') || a.distance - b.distance);
    const candidate = candidates[0];
    if (occlusion) {
      const roomMeshes = [];
      classroom.root.updateWorldMatrix(true, true);
      classroom.root.traverseVisible((object) => {
        if (!object.isMesh || !visible(object) || inCameraTree(object)) return;
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        if (materials.some(blocking)) roomMeshes.push(object);
      });
      stats.roomRaycasts++;
      const blocker = raycaster.intersectObjects(roomMeshes, false)
        .find((hit) => hit.distance < candidate.referenceDistance - 0.008 && blocking(hitMaterial(hit)));
      if (blocker) return reject('occluded', { cameraId: candidate.cameraId, blocker: blocker.object.userData?.name || blocker.object.name,
        blockerDistance: blocker.distance, cameraDistance: candidate.referenceDistance });
    }
    lastRejected = null; lastHit = { ...candidate, occlusionChecked: occlusion };
    return { ...lastHit, point: [...lastHit.point], centre: [...lastHit.centre] };
  }

  function setHover(id) { hoveredCameraId = id; canvas.style.cursor = id ? 'pointer' : originalCursor; }
  function cancelGesture(reason) { if (gesture) { gesture.cancelled = true; gesture.reason ||= reason; } }
  function trackMovement(event) {
    if (!gesture || event.pointerId !== gesture.pointerId) return;
    if (event.pointerType !== 'touch' && (event.buttons & ~1)) cancelGesture('non_primary_button');
    const samples = typeof event.getCoalescedEvents === 'function' ? event.getCoalescedEvents() : [];
    for (const sample of [...samples, event]) {
      const distance = Math.hypot(sample.clientX - gesture.startX, sample.clientY - gesture.startY);
      gesture.maxDistance = Math.max(gesture.maxDistance, distance);
    }
    if (gesture.maxDistance > 5) cancelGesture('drag');
  }
  function finish(event, reason) {
    pointers.delete(event.pointerId);
    if (gesture?.pointerId === event.pointerId) {
      lastGesture = { ...gesture, endReason: reason }; gesture = null;
    }
  }
  function pointerDown(event) {
    onInteract(event);
    pointers.add(event.pointerId);
    if (pointers.size > 1) cancelGesture('multiple_pointers');
    if (gesture || event.button !== 0 || event.isPrimary === false || pointers.size !== 1) return;
    gesture = { pointerId: event.pointerId, pointerType: event.pointerType, startX: event.clientX, startY: event.clientY,
      maxDistance: 0, cancelled: false, reason: null };
  }
  function documentDown(event) {
    if (gesture && event.pointerId !== gesture.pointerId) {
      pointers.add(event.pointerId); cancelGesture('multiple_pointers');
    }
  }
  function pointerMove(event) {
    trackMovement(event);
    if (gesture || (event.buttons && event.buttons !== 0) || event.pointerType === 'touch') { setHover(null); return; }
    setHover(pickAt(event.clientX, event.clientY, { occlusion: false })?.cameraId || null);
  }
  function pointerUp(event) {
    trackMovement(event);
    const current = gesture;
    try {
      if (current?.pointerId === event.pointerId && event.button === 0 && event.isPrimary !== false &&
          !current.cancelled && current.maxDistance <= 5 && pointers.size === 1) {
        const hit = pickAt(event.clientX, event.clientY);
        if (hit) onChoose(hit.cameraId, hit);
      }
    } finally { finish(event, 'pointerup'); }
  }
  function pointerCancel(event) { cancelGesture('pointercancel'); finish(event, 'pointercancel'); setHover(null); }
  const listen = (target, type, callback, capture = false) => target?.addEventListener(type, callback,
    { signal: controller.signal, capture, passive: true });
  listen(canvas, 'pointerdown', pointerDown, true);
  listen(canvas, 'wheel', (event) => { onInteract(event); cancelGesture('wheel'); setHover(null); }, true);
  listen(canvas, 'pointermove', pointerMove);
  listen(canvas, 'pointerleave', () => setHover(null));
  // Selection happens in the bubble phase, after OrbitControls can end its gesture.
  listen(canvas, 'pointerup', pointerUp);
  listen(documentTarget, 'pointerdown', documentDown, true);
  listen(documentTarget, 'pointermove', trackMovement, true);
  listen(documentTarget, 'pointerup', (event) => { trackMovement(event); finish(event, 'outside_or_completed'); });
  listen(documentTarget, 'pointercancel', pointerCancel, true);
  listen(canvas, 'pointercancel', pointerCancel);

  return {
    pickAt,
    getState: () => ({ disposed, hoveredCameraId, lastHit: lastHit ? { ...lastHit, point: [...lastHit.point], centre: [...lastHit.centre] } : null,
      lastRejected: lastRejected ? { ...lastRejected } : null, gesture: gesture ? { ...gesture } : null,
      lastGesture: lastGesture ? { ...lastGesture } : null, activePointerCount: pointers.size, stats: { ...stats } }),
    dispose() { if (disposed) return; disposed = true; controller.abort(); gesture = null; pointers.clear(); setHover(null); },
  };
}
