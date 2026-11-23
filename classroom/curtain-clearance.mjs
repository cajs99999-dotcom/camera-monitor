import * as THREE from 'three';

const VERSION = 'window-curtain-radiator-clearance-3';
const TARGET_MIN_X = -3.513, TARGET_MAX_X = -3.493;
const CURTAINS = [
  'Front window left gathered curtain', 'Front window right curtain',
  'Rear window left curtain', 'Rear window right gathered curtain',
];

// Reduce only the excessive front/back pleat depth. The four cloth panels stay
// below their original fixed pelmets; radiator hardware and window sills stay put.
export function applyCurtainClearance(classroom) {
  const root = classroom.root;
  if (root.userData.curtainClearance?.version === VERSION) return root.userData.curtainClearance;
  const previousVersion = root.userData.curtainClearance?.version || null;
  root.updateMatrixWorld(true);
  const inverseRoot = root.matrixWorld.clone().invert();
  function localBox(object) {
    const result = new THREE.Box3();
    // Transform each geometry box directly to the room frame. Transforming a
    // world AABB back would inflate it if the entire room were rotated later.
    object.traverse((child) => {
      if (!child.isMesh || !child.geometry?.attributes.position) return;
      if (!child.geometry.boundingBox) child.geometry.computeBoundingBox();
      const transform = new THREE.Matrix4().multiplyMatrices(inverseRoot, child.matrixWorld);
      result.union(child.geometry.boundingBox.clone().applyMatrix4(transform));
    });
    if (result.isEmpty()) throw new Error('A curtain-clearance object has no finite mesh bounds.');
    return result;
  }
  function required(name) {
    const object = classroom.getNodeByOriginalName(name);
    if (!object) throw new Error(`Missing curtain-clearance source node: ${name}`);
    return object;
  }
  function translateRootX(object, deltaX) {
    const target = root.worldToLocal(object.getWorldPosition(new THREE.Vector3()));
    target.x += deltaX;
    object.position.copy(object.parent.worldToLocal(root.localToWorld(target)));
    root.updateMatrixWorld(true);
  }
  const serializeBox = (box) => ({ min: box.min.toArray(), max: box.max.toArray() });
  const unchangedYZ = (a, b) => ['y', 'z'].every((axis) =>
    Math.abs(a.min[axis] - b.min[axis]) < 2e-6 && Math.abs(a.max[axis] - b.max[axis]) < 2e-6);
  const originalName = (object) => object.userData?.name || object.name;
  const triangleCache = new Map();
  const ray = new THREE.Ray(), direction = new THREE.Vector3(), hit = new THREE.Vector3();
  function triangles(mesh) {
    if (triangleCache.has(mesh)) return triangleCache.get(mesh);
    const position = mesh.geometry.attributes.position, index = mesh.geometry.index;
    const transform = new THREE.Matrix4().multiplyMatrices(inverseRoot, mesh.matrixWorld), result = [];
    for (let i = 0; i < (index ? index.count : position.count); i += 3) {
      const vertices = [0, 1, 2].map((j) => new THREE.Vector3()
        .fromBufferAttribute(position, index ? index.getX(i + j) : i + j).applyMatrix4(transform));
      result.push({ vertices, bounds: new THREE.Box3().setFromPoints(vertices) });
    }
    triangleCache.set(mesh, result);
    return result;
  }
  function edgesCross(a, b) {
    for (let i = 0; i < 3; i++) {
      direction.copy(a[(i + 1) % 3]).sub(a[i]);
      const length = direction.length();
      if (length < 1e-8) continue;
      ray.set(a[i], direction.divideScalar(length));
      if (ray.intersectTriangle(b[0], b[1], b[2], false, hit)) {
        const distance = hit.distanceTo(a[i]);
        if (distance > 1e-5 && distance < length - 1e-5) return true;
      }
    }
    return false;
  }
  function crosses(a, b) {
    const aBox = localBox(a), bBox = localBox(b);
    if (!aBox.intersectsBox(bBox)) return false;
    const aTriangles = triangles(a).filter((t) => t.bounds.intersectsBox(bBox));
    const bTriangles = triangles(b).filter((t) => t.bounds.intersectsBox(aBox));
    for (const first of aTriangles) for (const second of bTriangles) {
      if (first.bounds.intersectsBox(second.bounds) &&
          (edgesCross(first.vertices, second.vertices) || edgesCross(second.vertices, first.vertices))) return true;
    }
    return false;
  }
  function solidFurniture(mesh) {
    if (!mesh.isMesh || !mesh.geometry?.attributes.position) return false;
    const surfaces = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    if (surfaces.every((surface) => surface.transparent || surface.transmission > 0.05)) return false;
    for (let node = mesh; node && node !== root; node = node.parent) {
      if (!node.visible || node === classroom.cameras ||
          /^(ARCH_FRONT|ARCH_REAR|ARCH_WINDOWS|ARCH_DOORS|Floor and tile grid|Fixed_Seat_Occupancy_Avatars|Detailed_CCTV_Camera_Models)$/.test(originalName(node))) return false;
    }
    return true;
  }
  function interpolateVertex(a, b, t, names) {
    const attributes = {};
    for (const name of names) attributes[name] = a.attributes[name].map((v, i) => v + (b.attributes[name][i] - v) * t);
    return { point: a.point.clone().lerp(b.point, t), attributes };
  }
  function splitPolygon(polygon, axis, value, sign, names) {
    const inside = [], outside = [];
    for (let i = 0; i < polygon.length; i++) {
      const a = polygon[i], b = polygon[(i + 1) % polygon.length];
      const da = sign * (a.point[axis] - value), db = sign * (b.point[axis] - value);
      const aInside = da >= 0, bInside = db >= 0;
      (aInside ? inside : outside).push(a);
      if (aInside !== bInside) {
        const vertex = interpolateVertex(a, b, da / (da - db), names);
        inside.push(vertex); outside.push(vertex);
      }
    }
    return { inside, outside };
  }
  function outsideBox(polygon, box, names) {
    let remaining = polygon;
    const outside = [];
    // Emit the part outside each half-space; only the remaining inside part
    // proceeds to the next plane. What survives all six planes is discarded.
    for (const [axis, value, sign] of [
      ['x', box.min.x, 1], ['x', box.max.x, -1], ['y', box.min.y, 1],
      ['y', box.max.y, -1], ['z', box.min.z, 1], ['z', box.max.z, -1],
    ]) {
      if (remaining.length < 3) break;
      const parts = splitPolygon(remaining, axis, value, sign, names);
      if (parts.outside.length >= 3) outside.push(parts.outside);
      remaining = parts.inside;
    }
    return outside;
  }
  function cutCloth(mesh, boxes) {
    const geometry = mesh.geometry, attributeNames = Object.keys(geometry.attributes);
    const output = Object.fromEntries(attributeNames.map((name) => [name, []]));
    const index = geometry.index, position = geometry.attributes.position;
    const transform = new THREE.Matrix4().multiplyMatrices(inverseRoot, mesh.matrixWorld);
    const groups = [], edgeA = new THREE.Vector3(), edgeB = new THREE.Vector3();
    let beforeArea = 0, afterArea = 0, vertexCount = 0;
    function area(a, b, c) { return edgeA.subVectors(b.point, a.point).cross(edgeB.subVectors(c.point, a.point)).length() * 0.5; }
    function vertex(i) {
      return { point: new THREE.Vector3().fromBufferAttribute(position, i).applyMatrix4(transform),
        attributes: Object.fromEntries(attributeNames.map((name) => {
          const attribute = geometry.attributes[name];
          return [name, Array.from({ length: attribute.itemSize }, (_, component) => attribute.getComponent(i, component))];
        })) };
    }
    function emit(a, b, c, materialIndex) {
      const triangleArea = area(a, b, c);
      if (triangleArea < 1e-12) return;
      afterArea += triangleArea;
      const previous = groups[groups.length - 1];
      if (previous?.materialIndex === materialIndex) previous.count += 3;
      else groups.push({ start: vertexCount, count: 3, materialIndex });
      for (const v of [a, b, c]) for (const name of attributeNames) {
        const values = [...v.attributes[name]];
        if ((name === 'normal' || name === 'tangent') && values.length >= 3) {
          const length = Math.hypot(values[0], values[1], values[2]);
          if (length > 1e-12) for (let i = 0; i < 3; i++) values[i] /= length;
          if (name === 'tangent' && values.length === 4) values[3] = values[3] < 0 ? -1 : 1;
        }
        output[name].push(...values);
      }
      vertexCount += 3;
    }
    for (let i = 0; i < (index ? index.count : position.count); i += 3) {
      const triangle = [0, 1, 2].map((j) => vertex(index ? index.getX(i + j) : i + j));
      beforeArea += area(...triangle);
      let polygons = [triangle];
      const materialIndex = geometry.groups.find((group) => i >= group.start && i < group.start + group.count)?.materialIndex || 0;
      for (const box of boxes) {
        polygons = polygons.flatMap((polygon) => {
          const bounds = new THREE.Box3().setFromPoints(polygon.map((v) => v.point));
          return bounds.intersectsBox(box) ? outsideBox(polygon, box, attributeNames) : [polygon];
        });
        if (!polygons.length) break;
      }
      for (const polygon of polygons) for (let j = 1; j + 1 < polygon.length; j++) emit(polygon[0], polygon[j], polygon[j + 1], materialIndex);
    }
    // Only this runtime cloth receives the clipped copy. All source attributes,
    // original shared geometry, materials and the GLB binary remain intact.
    const clipped = geometry.clone();
    clipped.setIndex(null); clipped.clearGroups();
    for (const name of attributeNames) clipped.setAttribute(name,
      new THREE.Float32BufferAttribute(output[name], geometry.attributes[name].itemSize));
    for (const group of groups) clipped.addGroup(group.start, group.count, group.materialIndex);
    clipped.setDrawRange(0, vertexCount); clipped.computeBoundingBox(); clipped.computeBoundingSphere();
    mesh.geometry = clipped; triangleCache.delete(mesh);
    return { inputTriangles: (index ? index.count : position.count) / 3, outputTriangles: vertexCount / 3,
      beforeAreaMetres2: beforeArea, afterAreaMetres2: afterArea, removedAreaMetres2: beforeArea - afterArea,
      geometryCloned: true, preservedAttributes: attributeNames };
  }
  const radiators = Array.from({ length: 44 }, (_, index) => classroom.getNodeByOriginalName(
    index ? `Window radiator rib.${String(index).padStart(3, '0')}` : 'Window radiator rib',
  )).filter(Boolean);
  const curtains = CURTAINS.map((name) => {
    const object = required(name);
    const rear = name.startsWith('Rear');
    return { name, object, before: localBox(object),
      originalPosition: object.position.clone(), originalScale: object.scale.clone(), originalGeometry: object.geometry,
      sill: localBox(required(`Broad white window sill${rear ? '.001' : ''}`)),
      pelmet: localBox(required(`Window pelmet${rear ? '.001' : ''}`)) };
  });
  if (!radiators.length) throw new Error('No source radiator ribs found for curtain clearance.');
  const radiatorBox = new THREE.Box3();
  for (const object of radiators) radiatorBox.union(localBox(object));
  const changes = [];
  const furnitureCuts = [];
  try {
    for (const { name, object, before, originalScale, sill, pelmet } of curtains) {
      const span = before.max.x - before.min.x;
      if (!Number.isFinite(span) || span < 1e-6) throw new Error(`Invalid cloth depth: ${name}`);
      const scaleXMultiplier = (TARGET_MAX_X - TARGET_MIN_X) / span;
      object.scale.x *= scaleXMultiplier;
      root.updateMatrixWorld(true);
      const scaled = localBox(object);
      const deltaX = TARGET_MIN_X - scaled.min.x;
      translateRootX(object, deltaX);
      const after = localBox(object);
      const yzUnchanged = unchangedYZ(before, after);
      const xOnTarget = Math.abs(after.min.x - TARGET_MIN_X) < 2e-6 && Math.abs(after.max.x - TARGET_MAX_X) < 2e-6;
      if (!yzUnchanged || !xOnTarget) throw new Error(`Cloth axes do not preserve the required room Y/Z bounds: ${name}`);
      changes.push({ name, deltaRoot: [deltaX, 0, 0], scaleXMultiplier,
        beforeScale: originalScale.toArray(), afterScale: object.scale.toArray(),
        beforeBounds: serializeBox(before), afterBounds: serializeBox(after),
        beforeMinX: before.min.x, afterMinX: after.min.x,
        frontFaceShiftMetres: after.max.x - before.max.x,
        yzUnchanged, xOnTarget, hangingTopBefore: before.max.y, hangingTopAfter: after.max.y,
        gapFromRadiatorMetres: after.min.x - radiatorBox.max.x,
        gapFromSillMetres: after.min.x - sill.max.x,
        behindPelmetFrontGapMetres: pelmet.max.x - after.max.x });
    }
    const furniture = [];
    root.traverse((object) => { if (solidFurniture(object)) furniture.push(object); });
    for (const { name, object } of curtains) {
      const beforeCut = localBox(object);
      const colliders = furniture.filter((other) => crosses(object, other));
      const beforePairs = colliders.map(originalName);
      if (!colliders.length) { furnitureCuts.push({ name, beforePairs, afterPairs: [], changed: false }); continue; }
      // One millimetre hides the seam within the opaque furniture silhouette and
      // prevents a new coplanar edge. Use per-mesh boxes, never a whole desk group.
      const boxes = colliders.map((other) => localBox(other).expandByScalar(0.001));
      const cut = cutCloth(object, boxes);
      root.updateMatrixWorld(true);
      const afterCut = localBox(object);
      // Float32 interpolation can retract a rounded hem extremum by a few
      // hundredths of a millimetre. Hidden lower fabric may also disappear
      // entirely inside a cabinet; neither is a change to the visible outline.
      const silhouetteTolerance = 0.0001;
      const topPreserved = Math.abs(afterCut.max.y - beforeCut.max.y) <= silhouetteTolerance;
      const zPreserved = Math.abs(afterCut.min.z - beforeCut.min.z) <= silhouetteTolerance &&
        Math.abs(afterCut.max.z - beforeCut.max.z) <= silhouetteTolerance;
      const bottomTrimMetres = Math.max(0, afterCut.min.y - beforeCut.min.y);
      const bottomTrimmedBehindFurniture = bottomTrimMetres > silhouetteTolerance;
      if (!topPreserved || !zPreserved || afterCut.min.y < beforeCut.min.y - silhouetteTolerance ||
          Math.abs(afterCut.min.x - TARGET_MIN_X) > 2e-6 || Math.abs(afterCut.max.x - TARGET_MAX_X) > 2e-6) {
        throw new Error(`A hidden cloth cut changed its visible outline: ${name}; before=${JSON.stringify(serializeBox(beforeCut))}; after=${JSON.stringify(serializeBox(afterCut))}`);
      }
      const afterPairs = furniture.filter((other) => crosses(object, other)).map(originalName);
      if (afterPairs.length) throw new Error(`Furniture clearance is incomplete: ${name}`);
      furnitureCuts.push({ name, changed: true, beforePairs, afterPairs, paddingMetres: 0.001,
        outerBoundsPreserved: !bottomTrimmedBehindFurniture, visibleOuterBoundsPreserved: true,
        silhouetteToleranceMetres: silhouetteTolerance, topPreserved, zPreserved,
        bottomTrimmedBehindFurniture, bottomTrimMetres,
        beforeCutBounds: serializeBox(beforeCut), afterCutBounds: serializeBox(afterCut), ...cut });
    }
  } catch (error) {
    for (const item of curtains) {
      item.object.position.copy(item.originalPosition); item.object.scale.copy(item.originalScale);
      if (item.object.geometry !== item.originalGeometry) item.object.geometry.dispose();
      item.object.geometry = item.originalGeometry;
    }
    root.updateMatrixWorld(true);
    throw error;
  }
  const audit = { version: VERSION, previousVersion,
    method: 'cloth_x_depth_compression_and_furniture_hidden_box_cuts',
    sourceGeometryChanged: false, runtimeClothGeometryCloned: true, sourceMaterialsChanged: false,
    radiatorRibs: radiators.length, minimumGapMetres: 0.02,
    targetRootX: [TARGET_MIN_X, TARGET_MAX_X], targetPleatDepthMetres: TARGET_MAX_X - TARGET_MIN_X,
    radiatorIndoorFaceX: radiatorBox.max.x, changes, furnitureCuts,
    beforeFurnitureCrossingPairs: furnitureCuts.reduce((sum, cut) => sum + cut.beforePairs.length, 0),
    afterFurnitureCrossingPairs: furnitureCuts.reduce((sum, cut) => sum + cut.afterPairs.length, 0),
    status: changes.every((item) => item.yzUnchanged && item.xOnTarget &&
      item.gapFromRadiatorMetres >= 0.02 - 1e-6 && item.gapFromSillMetres >= 0.01 - 1e-6 &&
      item.behindPelmetFrontGapMetres >= 0.001 - 1e-6) ? 'cleared' : 'needs_review' };
  root.userData.curtainClearance = audit;
  return audit;
}
