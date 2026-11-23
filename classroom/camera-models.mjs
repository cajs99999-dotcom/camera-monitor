import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

const UP = new THREE.Vector3(0, 1, 0);
const FORWARD = new THREE.Vector3(0, 0, 1);
// Conservative bounds of the casing, hood and lens in its +Z optical frame.
const BODY_MIN = new THREE.Vector3(-0.074, -0.055, -0.136);
const BODY_MAX = new THREE.Vector3(0.074, 0.066, 0.140);

/** Add physical camera markers without changing the supplied GLB or feed IDs. */
export function createCameraModels(classroom) {
  if (!classroom?.root?.isObject3D || !classroom.cameras?.isObject3D) {
    throw new TypeError('Camera models require the classroom root and camera proxy.');
  }
  const definitions = classroom.cameraDefs || [];
  if (definitions.length !== 6 || new Set(definitions.map((item) => item.id)).size !== definitions.length ||
      definitions.some((item) => !/^CAM0[1-6]$/.test(item.id) ||
        ![item.position, item.target].every((v) => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite)))) {
    throw new TypeError('Expected six finite CAM01–CAM06 camera definitions.');
  }
  const bounds = classroom.bounds;
  if (!bounds || !['minX', 'maxX', 'minZ', 'maxZ'].every((key) => Number.isFinite(bounds[key]))) {
    throw new TypeError('Camera mounting needs finite room bounds.');
  }
  const root = new THREE.Group();
  root.name = 'Detailed_CCTV_Camera_Models';
  root.userData = { kind: 'physical_camera_markers', calibrated: false, cameras: definitions.length,
    nominalBodyLengthMetres: 0.276, hasProjectionFrusta: false };
  const entries = new Map();
  const hiddenSourceNodeNames = [];
  const material = (name, color, roughness, metalness = 0) => {
    const result = new THREE.MeshStandardMaterial({ color, roughness, metalness });
    result.name = name;
    return result;
  };
  const white = material('CCTV_OffWhiteHousing', '#e4e9e5', 0.68, 0.025);
  const grey = material('CCTV_MountMetal', '#b1bab6', 0.66, 0.18);
  const black = material('CCTV_DarkBezel', '#202a29', 0.66, 0.035);
  const glass = material('CCTV_DarkOpticalGlass', '#14282d', 0.30, 0.08);
  const pupil = material('CCTV_LensPupil', '#070e11', 0.44, 0.02);
  const trim = material('CCTV_LensRing', '#65716d', 0.57, 0.16);

  function mesh(name, geometry, surface, parent, position = [0, 0, 0]) {
    const object = new THREE.Mesh(geometry, surface);
    object.name = name; object.position.fromArray(position);
    object.castShadow = true; object.receiveShadow = true;
    parent.add(object);
    return object;
  }
  function rounded(name, size, position, surface, parent, radius = 0.009) {
    return mesh(name, new RoundedBoxGeometry(...size, 2, radius), surface, parent, position);
  }
  function cylinder(name, radius, depth, position, direction, surface, parent, segments = 24) {
    const result = mesh(name, new THREE.CylinderGeometry(radius, radius, depth, segments), surface, parent, position.toArray());
    result.quaternion.setFromUnitVectors(UP, direction);
    return result;
  }
  function rod(name, start, end, radius, surface, parent) {
    const direction = end.clone().sub(start);
    const length = direction.length();
    if (length < 1e-5) return null;
    return cylinder(name, radius, length, start.clone().add(end).multiplyScalar(0.5), direction.divideScalar(length), surface, parent, 14);
  }
  const aim = new THREE.Object3D();
  function rotationAt(position, target) {
    aim.position.copy(position);
    aim.up.copy(UP);
    aim.lookAt(target); // Object3D's +Z axis, matching the lens geometry below.
    return aim.quaternion.clone();
  }
  function bodyBounds(position, quaternion) {
    const box = new THREE.Box3();
    for (const x of [BODY_MIN.x, BODY_MAX.x]) for (const y of [BODY_MIN.y, BODY_MAX.y]) for (const z of [BODY_MIN.z, BODY_MAX.z]) {
      box.expandByPoint(new THREE.Vector3(x, y, z).applyQuaternion(quaternion).add(position));
    }
    return box;
  }
  function safeBodyPosition(position, target) {
    // A tilted casing can reach further back than its optical centre. Keep its
    // complete box inside the room instead of moving the GLB wall or feed label.
    const result = position.clone(), clearance = 0.008;
    for (let i = 0; i < 4; i++) {
      const box = bodyBounds(result, rotationAt(result, target));
      const dx = Math.max(0, bounds.minX + clearance - box.min.x) - Math.max(0, box.max.x - bounds.maxX + clearance);
      const dz = Math.max(0, bounds.minZ + clearance - box.min.z) - Math.max(0, box.max.z - bounds.maxZ + clearance);
      if (Math.abs(dx) + Math.abs(dz) < 1e-7) break;
      result.x += dx; result.z += dz;
    }
    return result;
  }
  function wallFor(definition) {
    if (definition.mountingWall) return definition.mountingWall;
    // C3 is the unmodelled rear/window corner; use its solid rear-wall pier.
    if (definition.id === 'CAM03') return 'back';
    if (definition.id === 'CAM01' || definition.id === 'CAM04') return 'front';
    if (definition.id === 'CAM02' || definition.id === 'CAM05') return 'right';
  }
  function wallAnchor(wall, position, mountingHeight) {
    const point = position.clone(), normal = new THREE.Vector3();
    point.y = THREE.MathUtils.clamp(mountingHeight ?? (point.y - 0.125), 0.15, (classroom.config?.height || 3.35) - 0.15);
    point.x = THREE.MathUtils.clamp(point.x, bounds.minX + 0.075, bounds.maxX - 0.075);
    point.z = THREE.MathUtils.clamp(point.z, bounds.minZ + 0.075, bounds.maxZ - 0.075);
    if (wall === 'front') { point.z = bounds.minZ; normal.set(0, 0, 1); }
    else if (wall === 'back') { point.z = bounds.maxZ; normal.set(0, 0, -1); }
    else if (wall === 'left') { point.x = bounds.minX; normal.set(1, 0, 0); }
    else { point.x = bounds.maxX; normal.set(-1, 0, 0); }
    return { point, normal };
  }
  function buildBody(parent) {
    rounded('CCTV_WhiteCasing', [0.130, 0.100, 0.210], [0, 0, -0.014], white, parent, 0.014);
    rounded('CCTV_UpperWeatherHood', [0.144, 0.018, 0.245], [0, 0.055, -0.009], white, parent, 0.006);
    rounded('CCTV_BlackFrontInsert', [0.112, 0.083, 0.012], [0, 0, 0.098], black, parent, 0.009);
    cylinder('CCTV_LensBezel', 0.034, 0.022, new THREE.Vector3(0, 0, 0.119), FORWARD, black, parent, 32);
    mesh('CCTV_GlassRetainingRing', new THREE.TorusGeometry(0.0285, 0.0026, 8, 32), trim, parent, [0, 0, 0.131]);
    cylinder('CCTV_OpticalGlass', 0.0252, 0.003, new THREE.Vector3(0, 0, 0.133), FORWARD, glass, parent, 32);
    cylinder('CCTV_CentralLens', 0.012, 0.0018, new THREE.Vector3(0, 0, 0.1358), FORWARD, pupil, parent, 24);
    cylinder('CCTV_RearCableGrommet', 0.009, 0.012, new THREE.Vector3(0, -0.014, -0.126), FORWARD, black, parent, 16);
    // Small side ribs remain geometry; no screen, text, UI or view-cone helper.
    for (const side of [-1, 1]) for (let i = 0; i < 3; i++) {
      rounded('CCTV_SideVent', [0.0015, 0.020, 0.003], [side * 0.064, -0.006, -0.065 + i * 0.012], grey, parent, 0.0007);
    }
  }
  function sourceHardwareNames(definition) {
    const name = definition.sourceNodeName;
    if (!name) return [];
    const front = name.match(/^Audit3 \/ Front black monitor camera ([12])$/);
    if (front) return [name, `Audit3 / Front camera foot ${front[1]}`, `Audit3 / Front camera lens ${front[1]}`];
    const side = name.match(/^Small wall camera white housing(\.\d+)?$/);
    if (side) {
      const suffix = side[1] || '';
      return [name, `Small wall camera dark front${suffix}`, `Camera support arm${suffix}`, `Yellow camera wall bracket${suffix}`];
    }
    return [name];
  }

  for (const definition of definitions) {
    const cameraId = `cam${Number(definition.id.slice(3))}`;
    const desired = new THREE.Vector3().fromArray(definition.position);
    const target = new THREE.Vector3().fromArray(definition.target);
    const position = safeBodyPosition(desired, target), quaternion = rotationAt(position, target);
    const group = new THREE.Group(); group.name = `CCTV_${definition.id}`;
    group.position.copy(position);
    // Object3D.id remains Three.js's numeric identity; monitoring IDs live here.
    group.userData = { cameraId, id: definition.id, calibrated: false,
      estimated: definition.estimated !== false, mountingWall: wallFor(definition),
      positionStatus: definition.positionStatus || 'estimated', sourceNodeName: definition.sourceNodeName || null,
      definitionPosition: [...definition.position], casingAdjustmentMetres: position.distanceTo(desired) };
    const body = new THREE.Group(); body.name = 'CCTV_AimedBody'; body.quaternion.copy(quaternion);
    group.add(body); buildBody(body);
    const mount = new THREE.Group(); mount.name = 'CCTV_WallMount'; group.add(mount);
    const { point, normal } = wallAnchor(group.userData.mountingWall, position, definition.mountingHeight);
    const local = (p) => p.clone().sub(position);
    cylinder('CCTV_BaseGasket', 0.054, 0.003, local(point.clone().addScaledVector(normal, 0.0025)), normal, grey, mount);
    cylinder('CCTV_WallPlate', 0.052, 0.018, local(point.clone().addScaledVector(normal, 0.013)), normal, white, mount);
    const tangent = new THREE.Vector3().crossVectors(UP, normal).normalize();
    for (const a of [-1, 1]) for (const b of [-1, 1]) {
      const screw = point.clone().addScaledVector(normal, 0.0235).addScaledVector(tangent, a * 0.028).addScaledVector(UP, b * 0.027);
      cylinder('CCTV_MountScrew', 0.0042, 0.003, local(screw), normal, black, mount, 12);
    }
    const start = point.clone().addScaledVector(normal, 0.032);
    const elbow = point.clone().addScaledVector(normal, 0.092);
    const attachment = new THREE.Vector3(0, -0.068, -0.035).applyQuaternion(quaternion).add(position);
    rod('CCTV_WallArm', local(start), local(elbow), 0.012, white, mount);
    rod('CCTV_RisingArm', local(elbow), local(attachment), 0.011, grey, mount);
    mesh('CCTV_ElbowJoint', new THREE.SphereGeometry(0.015, 14, 10), white, mount, local(elbow).toArray());
    mesh('CCTV_TiltJoint', new THREE.SphereGeometry(0.019, 16, 12), grey, mount, local(attachment).toArray());
    const neck = new THREE.Vector3(0, -0.051, -0.035).applyQuaternion(quaternion).add(position);
    rod('CCTV_CasingFoot', local(attachment), local(neck), 0.014, white, mount);
    root.add(group);
    entries.set(cameraId, { cameraId, id: definition.id, root: group, body, mount, definition,
      position, target, quaternion, mountingPoint: point, mountingNormal: normal, hiddenSourceNodes: [] });
  }
  // Keep these independent of ARCH_* groups: cutaway walls must not remove the
  // clickable camera locations or their hardware models.
  classroom.cameras.add(root);
  for (const entry of entries.values()) for (const name of sourceHardwareNames(entry.definition)) {
    const original = classroom.getNodeByOriginalName?.(name);
    if (!original || original === classroom.root || original === classroom.cameras ||
        Object.values(classroom.walls || {}).includes(original)) continue;
    // Hide the named hardware itself, never its architecture parent/siblings.
    original.visible = false;
    entry.hiddenSourceNodes.push(name); hiddenSourceNodeNames.push(name);
  }
  root.updateMatrixWorld(true);
  function getState() {
    return { count: entries.size, nominalBodyLengthMetres: 0.276, calibrated: false,
      hiddenSourceNodeNames: [...hiddenSourceNodeNames],
      cameras: [...entries.values()].map((entry) => {
        const box = bodyBounds(entry.position, entry.quaternion);
        return { cameraId: entry.cameraId, id: entry.id, visible: entry.root.visible,
          position: entry.position.toArray(), definitionPosition: [...entry.definition.position], target: entry.target.toArray(),
          forward: FORWARD.clone().applyQuaternion(entry.quaternion).toArray(),
          mountingWall: entry.root.userData.mountingWall, mountingPoint: entry.mountingPoint.toArray(),
          casingAdjustmentMetres: entry.root.userData.casingAdjustmentMetres,
          calibrated: false, estimated: entry.root.userData.estimated,
          bodyRoomBounds: { min: box.min.toArray(), max: box.max.toArray() },
          hiddenSourceNodeNames: [...entry.hiddenSourceNodes] };
      }) };
  }
  return { root, entries, getState };
}
