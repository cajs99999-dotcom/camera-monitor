import * as THREE from 'three';

const VERSION = 'classroom-v3-furniture-clearance-2';
const nameOf = (object) => object.userData?.name || object.name;

/** Small, named V3 corrections. Never edits buffer geometry or the source GLB. */
export function applySceneRepairs(classroom) {
  const root = classroom?.root;
  if (!root || typeof classroom.getNodeByOriginalName !== 'function') throw new Error('Scene repairs require the V3 classroom adapter.');
  if (root.userData.sceneRepairs?.version === VERSION) return root.userData.sceneRepairs;
  root.updateMatrixWorld(true);
  const audit = { version: VERSION, method: 'named_cross_furniture_triangle_checks',
    sourceGeometryChanged: false, sourceMaterialsChanged: false, changes: [], checks: [], skipped: [],
    scope: 'Named furniture intersections, including the front work chair against sibling meshes in its shared scene group. Intentional within-object joints and supported tabletop props are excluded.' };
  const geometryCache = new Map(), boxCache = new Map();
  const ray = new THREE.Ray(), direction = new THREE.Vector3(), intersection = new THREE.Vector3();
  const point = new THREE.Vector3();
  function resetCaches() { geometryCache.clear(); boxCache.clear(); }
  function box(object) {
    if (!boxCache.has(object)) boxCache.set(object, new THREE.Box3().setFromObject(object));
    return boxCache.get(object);
  }
  function meshes(object) { const result = []; object.traverse((child) => { if (child.isMesh) result.push(child); }); return result; }
  function triangles(mesh) {
    if (geometryCache.has(mesh)) return geometryCache.get(mesh);
    const position = mesh.geometry.attributes.position, index = mesh.geometry.index, result = [];
    if (position) for (let offset = 0; offset < (index ? index.count : position.count); offset += 3) {
      const vertices = [0, 1, 2].map((corner) => new THREE.Vector3()
        .fromBufferAttribute(position, index ? index.getX(offset + corner) : offset + corner).applyMatrix4(mesh.matrixWorld));
      result.push({ vertices, bounds: new THREE.Box3().setFromPoints(vertices) });
    }
    geometryCache.set(mesh, result); return result;
  }
  function edgesCross(a, b) {
    for (let edge = 0; edge < 3; edge++) {
      direction.copy(a[(edge + 1) % 3]).sub(a[edge]);
      const length = direction.length(); if (length < 1e-8) continue;
      ray.set(a[edge], direction.divideScalar(length));
      const hit = ray.intersectTriangle(b[0], b[1], b[2], false, intersection);
      if (hit) {
        const distance = hit.distanceTo(a[edge]);
        // End-point/coplanar contact is not treated as penetration.
        if (distance > 1e-5 && distance < length - 1e-5) return true;
      }
    }
    return false;
  }
  function crosses(a, b) {
    const aBox = box(a), bBox = box(b); if (!aBox.intersectsBox(bBox)) return false;
    const aTriangles = triangles(a).filter((triangle) => triangle.bounds.intersectsBox(bBox));
    const bTriangles = triangles(b).filter((triangle) => triangle.bounds.intersectsBox(aBox));
    for (const first of aTriangles) for (const second of bTriangles) {
      if (first.bounds.intersectsBox(second.bounds) &&
          (edgesCross(first.vertices, second.vertices) || edgesCross(second.vertices, first.vertices))) return true;
    }
    return false;
  }
  function evidence(a, b, { includeCabinetContainment = false } = {}) {
    const pairs = [], containmentPairs = [], ownMeshes = meshes(a), own = new Set(ownMeshes);
    // The front-window scene group also contains the target chair. Exclude
    // only that chair's own pieces, not every sibling sharing its parent.
    const otherMeshes = meshes(b).filter((mesh) => !own.has(mesh));
    if (box(a).intersectsBox(box(b))) for (const first of ownMeshes) for (const second of otherMeshes) {
      if (crosses(first, second)) pairs.push([nameOf(first), nameOf(second)]);
      else if (includeCabinetContainment && nameOf(second) === 'Front window short drawer cabinet' && box(second).containsBox(box(first))) {
        // This exact source mesh is a closed, axis-aligned cuboid. Entire
        // chair pieces inside it have no triangle-edge crossings to detect.
        containmentPairs.push([nameOf(first), nameOf(second)]);
      }
    }
    const aBox = box(a), bBox = box(b);
    return { crossingMeshPairs: pairs.length, containedMeshPairs: containmentPairs.length,
      samplePairs: pairs.slice(0, 18), containmentPairs,
      boundsA: { min: aBox.min.toArray(), max: aBox.max.toArray() },
      boundsB: { min: bBox.min.toArray(), max: bBox.max.toArray() } };
  }
  function positionInRoot(object) { return root.worldToLocal(object.getWorldPosition(new THREE.Vector3())); }
  function translateInRoot(object, delta) {
    const target = root.localToWorld(positionInRoot(object).add(new THREE.Vector3().fromArray(delta)));
    object.position.copy(object.parent.worldToLocal(target)); root.updateMatrixWorld(true); resetCaches();
  }
  function syncSeatMetadata(object, delta) {
    const updated = [];
    for (const seat of classroom.seats || []) if (seat.chair === object) {
      for (const field of ['position', 'chairPosition', 'seatCentre']) {
        if (Array.isArray(seat[field])) seat[field] = seat[field].map((value, axis) => value + delta[axis]);
      }
      updated.push(seat.label);
    }
    return updated;
  }
  function repair({ id, target, other, delta, scaleX, reason, includeCabinetContainment = false }) {
    const object = classroom.getNodeByOriginalName(target), obstacle = classroom.getNodeByOriginalName(other);
    if (!object || !obstacle) { audit.skipped.push({ id, reason: 'Expected V3 source node missing.', target, other }); return; }
    const before = evidence(object, obstacle, { includeCabinetContainment });
    if (!before.crossingMeshPairs && !before.containedMeshPairs) { audit.checks.push({ id, target, other, before, after: before, changed: false }); return; }
    const originalPosition = object.position.clone(), originalScale = object.scale.clone();
    const beforePosition = positionInRoot(object).toArray();
    if (scaleX) object.scale.x *= scaleX;
    translateInRoot(object, delta);
    const after = evidence(object, obstacle, { includeCabinetContainment });
    if (after.crossingMeshPairs || after.containedMeshPairs) {
      object.position.copy(originalPosition); object.scale.copy(originalScale); root.updateMatrixWorld(true); resetCaches();
      audit.skipped.push({ id, target, other, reason: 'Correction did not fully clear the named pair; original transform restored.', before, attemptedAfter: after });
      return;
    }
    const change = { id, target, reason, deltaRoot: [...delta], scaleXMultiplier: scaleX || 1,
      beforePosition, afterPosition: positionInRoot(object).toArray(),
      beforeScale: originalScale.toArray(), afterScale: object.scale.toArray(),
      updatedSeatLabels: syncSeatMetadata(object, delta) };
    audit.changes.push(change); audit.checks.push({ id, target, other, before, after, changed: true });
  }

  repair({ id: 'tools-chair-table-leg', target: 'White tools chair front',
    other: 'White electronics tools table / across room', delta: [0, 0, -0.25],
    reason: 'Pull the loose chair 25 cm toward the blackboard so its seat/skid no longer cuts through the table leg.' });
  repair({ id: 'rear-spare-chairs', target: 'Rear black corner office chair',
    other: 'Rear utility white chair 1.06', delta: [0, 0, 0.34],
    reason: 'Move the spare black chair 34 cm into the open rear corner, clearing the white chair back and seat.' });
  repair({ id: 'rear-cabinet-printer-table', target: 'Rear white monitor unit',
    other: 'Rear yellow printer board 1 / 1m', delta: [0.045, 0, 0], scaleX: 0.8875,
    reason: 'Reduce the estimated cabinet width from 0.80 m to 0.71 m while keeping its right edge fixed; clear the adjacent printer table by 1 cm without moving the tabletop props.' });
  repair({ id: 'front-work-chair-short-cabinet', target: 'Front black office chair / faces toward rear',
    other: 'Front window / computer desk and separate printer cabinet', delta: [0.36, 0, 0], includeCabinetContainment: true,
    reason: 'Move W01 36 cm toward the room centre, clearing the short drawer cabinet, its door/drawer and bottles; leave the computer desk and cabinet in place.' });

  // Recheck the immediate neighbours of the corrected cabinet/chair. These
  // also document that the width correction is not a move into the next unit.
  const neighbours = [
    ['Rear white monitor unit', 'Rear white kettle unit'],
    ['Rear black corner office chair', 'Rear yellow printer board 2 / 1m'],
    ['Rear black corner office chair', 'Rear wall stacked boxes'],
    ['White tools chair front', 'White glass cabinet next to front door'],
    ['Front black office chair / faces toward rear', 'Front white computer desk'],
    ['Front black office chair / faces toward rear', 'Backpack white chair on platform'],
  ];
  audit.neighbourChecks = neighbours.flatMap(([first, second]) => {
    const a = classroom.getNodeByOriginalName(first), b = classroom.getNodeByOriginalName(second);
    return a && b ? [{ first, second, ...evidence(a, b) }] : [];
  });
  const rearChair = classroom.getNodeByOriginalName('Rear black corner office chair');
  if (rearChair && classroom.bounds) {
    const rearBox = box(rearChair); point.copy(rearBox.max); root.worldToLocal(point);
    audit.rearWallClearance = classroom.bounds.maxZ - point.z;
  }
  audit.beforeCrossingMeshPairs = audit.checks.reduce((sum, check) => sum + check.before.crossingMeshPairs, 0);
  audit.afterCrossingMeshPairs = audit.checks.reduce((sum, check) => sum + check.after.crossingMeshPairs, 0);
  audit.beforeContainedMeshPairs = audit.checks.reduce((sum, check) => sum + check.before.containedMeshPairs, 0);
  audit.afterContainedMeshPairs = audit.checks.reduce((sum, check) => sum + check.after.containedMeshPairs, 0);
  audit.status = audit.skipped.length ? 'partial' : audit.changes.length ? 'repaired' : 'already_clear';
  resetCaches(); root.userData.sceneRepairs = audit;
  return audit;
}
