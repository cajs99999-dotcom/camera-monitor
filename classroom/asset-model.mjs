import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export const DEFAULT_CONFIG = Object.freeze({
  width: 7.45, depth: 8.9, height: 3.35, deskWidth: 1.0, tileSize: 0.6,
  status: 'user_supplied_video_review_v3_room_dimensions_estimated',
});

// These are spatial bindings to the existing monitor's fixed numbers. The
// supplied GLB names furniture identities, but does not contain seat 01–15 IDs.
const STUDENTS = [
  ['01', 'F1', 1], ['02', 'F2', 1], ['03', 'F3', 1], ['04', 'C_Fw', 1], ['05', 'C_Fd', 1],
  ['06', 'O_window', 2], ['07', 'O_room', 2], ['08', 'C_Mw', 2], ['09', 'C_Md', 2], ['10', 'B2', 2],
  ['11', 'P_window', 3], ['12', 'P_room', 3], ['13', 'C_Rw', 3], ['14', 'C_Rd', 3], ['15', 'BLUE', 3],
];
const SOURCE_ZIP = 'classroom-video-review-v3.zip';
const SOURCE_MEMBER = 'classroom-video-review-v3/classroom-v3.glb';

/** Load the supplied scene and apply explicitly named display replacements. */
export async function loadClassroomModel({ assetUrl = 'assets/classroom-v3.glb', onProgress } = {}) {
  const gltf = await new GLTFLoader().loadAsync(assetUrl, onProgress);
  const config = { ...DEFAULT_CONFIG };
  const root = new THREE.Group();
  root.name = 'Classroom_311_User_Video_Review_V3';
  root.userData = {
    units: 'metres', up: '+Y', front: '-Z_blackboard', windows: '-X',
    sourceZip: SOURCE_ZIP, sourceMember: SOURCE_MEMBER,
    metricCalibration: false, roomDimensionsEstimated: true,
    seatBinding: 'manual_spatial_order_not_confirmed_numbering_in_source_archive',
    cameraBinding: 'retained_monitor_ids_by_spatial_correspondence_not_embedded_camera_ids',
    notes: 'The source GLB stays intact; named runtime edits are audited separately.',
  };

  // The GLB already has +Y up. Its exported root includes an X reflection;
  // keep that node intact. This outer rotation/translation only centres the
  // room and makes its blackboard -Z and exterior windows -X for the viewer.
  const orientation = new THREE.Group();
  orientation.name = 'V3_Centred_Viewer_Frame';
  orientation.rotation.y = Math.PI;
  orientation.position.set(config.width / 2, 0, -config.depth / 2);
  orientation.add(gltf.scene); root.add(orientation);
  root.updateMatrixWorld(true);

  const lookup = new Map();
  function originalName(object) {
    if (typeof object.userData?.name === 'string') return object.userData.name;
    const index = gltf.parser?.associations?.get(object)?.nodes;
    const preserved = Number.isInteger(index) ? gltf.parser?.json?.nodes?.[index]?.name : null;
    return typeof preserved === 'string' ? preserved : object.name;
  }
  gltf.scene.traverse((object) => {
    const name = originalName(object);
    if (name && !lookup.has(name)) lookup.set(name, object);
  });
  function required(name) {
    const object = lookup.get(name);
    if (!object) throw new Error(`The supplied classroom asset is missing required node: ${name}`);
    return object;
  }
  // F3 uses the same complete black office-chair model as the neighbouring
  // central seat. Replace its shape as well as its color, keeping its placement.
  const replacedChairName = 'Chair / F3', referenceChairName = 'Chair / C_Fw';
  const previousChair = required(replacedChairName), referenceChair = required(referenceChairName);
  const replacementChair = referenceChair.clone(true);
  const sourceParts = [], replacementParts = [];
  referenceChair.traverse((object) => sourceParts.push(object));
  replacementChair.traverse((object) => replacementParts.push(object));
  for (let index = 0; index < replacementParts.length; index++) {
    const object = replacementParts[index];
    const name = originalName(sourceParts[index]).replace(referenceChairName, replacedChairName);
    object.name = name;
    object.userData = { ...object.userData, name };
  }
  replacementChair.position.copy(previousChair.position);
  replacementChair.quaternion.copy(previousChair.quaternion);
  replacementChair.scale.copy(previousChair.scale);
  replacementChair.userData.reference_kind = 'black_office';
  replacementChair.userData.styleReference = referenceChairName;
  previousChair.traverse((object) => {
    const name = originalName(object);
    if (lookup.get(name) === object) lookup.delete(name);
  });
  previousChair.parent.add(replacementChair);
  previousChair.removeFromParent();
  replacementChair.traverse((object) => lookup.set(object.userData.name, object));
  root.userData.chairReplacements = [{ target: replacedChairName, reference: referenceChairName,
    method: 'full_chair_subtree_clone', positionPreserved: true, orientationPreserved: true,
    scalePreserved: true, originalChairRemoved: true }];
  // Align the rear central white office chair with its desk and the adjacent chair.
  const rearWhiteChair = required('Chair / C_Rd');
  const previousOrientation = rearWhiteChair.quaternion.toArray();
  rearWhiteChair.quaternion.set(0, 1, 0, 0);
  root.userData.chairOrientations = [{ target: 'Chair / C_Rd',
    beforeQuaternion: previousOrientation, afterQuaternion: rearWhiteChair.quaternion.toArray(),
    positionPreserved: true, scalePreserved: true, facing: '-Z_blackboard' }];
  // Pull the adjacent student chair out from under the desk to leave sitting space.
  const rearStudentChair = required('Chair / C_Rw');
  root.updateMatrixWorld(true);
  const beforeChairPosition = root.worldToLocal(rearStudentChair.getWorldPosition(new THREE.Vector3()));
  const afterChairPosition = beforeChairPosition.clone().add(new THREE.Vector3(0, 0, .25));
  rearStudentChair.position.copy(rearStudentChair.parent.worldToLocal(root.localToWorld(afterChairPosition.clone())));
  root.userData.chairOffsets = [{ target: 'Chair / C_Rw', deltaRoot: [0, 0, .25],
    beforePosition: beforeChairPosition.toArray(), afterPosition: afterChairPosition.toArray(),
    orientationPreserved: true, scalePreserved: true }];
  root.updateMatrixWorld(true);
  function localPosition(object) {
    return root.worldToLocal(object.getWorldPosition(new THREE.Vector3()));
  }
  function localTop(object, anchor) {
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty() || !Number.isFinite(box.max.y)) return null;
    const worldAnchor = root.localToWorld(anchor.clone());
    return root.worldToLocal(new THREE.Vector3(worldAnchor.x, box.max.y, worldAnchor.z)).y;
  }

  const walls = {
    front: required('ARCH_FRONT'), back: required('ARCH_REAR'),
    left: required('ARCH_WINDOWS'), right: required('ARCH_DOORS'),
  };
  const bounds = { minX: -config.width / 2, maxX: config.width / 2, minZ: -config.depth / 2, maxZ: config.depth / 2 };
  const floor = lookup.get('Floor and tile grid');
  const floorHeight = floor ? localTop(floor, new THREE.Vector3()) ?? 0 : 0;
  function describeSeat({ label, identity, chairName, equipmentName, row, column, kind }) {
    const chair = required(chairName);
    const anchor = localPosition(chair);
    const equipment = required(equipmentName);
    const target = localPosition(equipment);
    const dx = target.x - anchor.x, dz = target.z - anchor.z;
    const rotationY = Math.hypot(dx, dz) > 1e-5 ? Math.atan2(-dx, -dz) : 0;
    const tops = [];
    chair.traverse((object) => {
      if (!object.isMesh || !/\.(SeatCushion|IvorySeat|OpenWovenSeatGrid)$/.test(originalName(object))) return;
      const top = localTop(object, anchor);
      if (top !== null) tops.push(top);
    });
    return {
      id: kind === 'student' ? `S${label}` : 'W01', label, number: kind === 'student' ? Number(label) : null,
      kind, row, column, position: anchor.toArray(), rotationY,
      // Both heights are absolute Y values in the returned root's local frame.
      seatHeight: tops.length ? Math.max(...tops) : anchor.y + 0.48,
      floorHeight, chair, equipment, chairName, equipmentName, sourceIdentity: identity,
      seatHeightSource: tops.length ? 'named_seat_surface_world_bounds' : 'estimated_0.48_above_chair_base',
      bindingStatus: 'manual_spatial_order_not_source_number', estimated: true,
    };
  }
  const seats = STUDENTS.map(([label, identity, row], index) => describeSeat({
    label, identity, row, column: index % 5 + 1, kind: 'student',
    chairName: `Chair / ${identity}`, equipmentName: `Equipment / ${identity}`,
  }));
  seats.push(describeSeat({ label: 'W01', identity: 'front_workplace', row: 0, column: 1, kind: 'worktable',
    chairName: 'Front black office chair / faces toward rear', equipmentName: 'Front white computer desk' }));

  // Four inputs match hardware in the GLB. C3 and C6 use estimated positions.
  const cameraSpecs = [
    { id: 'CAM01', label: '1号 · 黑板墙中部', color: '#0b9d9a', node: 'Audit3 / Front black monitor camera 2', target: [0, 1.2, 1.2] },
    { id: 'CAM02', label: '2号 · 走廊墙后段', color: '#b276d2', node: 'Small wall camera white housing.001', target: [-0.8, 1.2, -0.7] },
    { id: 'CAM03', label: '3号 · 后方窗角（估算）', color: '#458ad2', position: [-config.width / 2 + 0.18, 2.65, config.depth / 2 - 0.18], target: [0.4, 1.2, -1.5] },
    { id: 'CAM04', label: '4号 · 黑板墙偏窗侧', color: '#df9b40', node: 'Audit3 / Front black monitor camera 1', target: [0.1, 1.2, 1.6] },
    { id: 'CAM05', label: '5号 · 走廊墙前中段', color: '#d56567', node: 'Small wall camera white housing', target: [-1.0, 1.2, 1.4] },
    // The screenshot marks the rear side of the air conditioner on the centre window pier.
    { id: 'CAM06', label: '6号 · 窗间空调上方', color: '#75bb73', position: [bounds.minX + 0.18, 2.81, 0],
      target: [0, 1.2, 0], mountingWall: 'left', mountingHeight: 2.76, positionStatus: 'estimated_from_screenshot' },
  ];
  const cameraDefs = cameraSpecs.map((spec) => ({
    id: spec.id, label: spec.label, color: spec.color,
    position: spec.node ? localPosition(required(spec.node)).toArray() : [...spec.position], target: [...spec.target],
    sourceNodeName: spec.node || null, estimated: true, calibrated: false,
    mountingWall: spec.mountingWall, mountingHeight: spec.mountingHeight,
    positionStatus: spec.positionStatus ?? (spec.node ? 'model_hardware_spatial_correspondence' : 'estimated_missing_source_hardware'),
    bindingStatus: 'existing_feed_number_not_confirmed_in_source_archive',
  }));
  const cameras = new THREE.Group(); cameras.name = 'CCTV_Camera_Entry_Proxy'; root.add(cameras);
  // The existing viewer hides its `overhead` field. Keep that legacy control
  // away from the supplied scene: source fixtures must retain their visibility.
  const sourceOverhead = lookup.get('OVERHEAD');
  let sourceOverheadMeshCount = 0;
  sourceOverhead?.traverse((object) => { if (object.isMesh) sourceOverheadMeshCount++; });
  const overhead = new THREE.Group(); overhead.name = 'Legacy_Overhead_Visibility_Proxy'; root.add(overhead);
  root.userData.seatCounts = { student: 15, independentWorkplace: 1 };
  root.userData.cameraCounts = {
    monitoringEntries: cameraDefs.length,
    sourceHardwareLocations: cameraDefs.filter((definition) => definition.sourceNodeName).length,
    estimatedLocations: cameraDefs.filter((definition) => !definition.sourceNodeName).map((definition) => definition.id),
  };
  root.userData.overheadPolicy = 'source_group_preserved_legacy_visibility_field_is_empty_proxy';
  root.userData.sourceOverhead = { present: Boolean(sourceOverhead), meshCount: sourceOverheadMeshCount,
    note: sourceOverheadMeshCount ? 'Source fixtures preserved.' : 'This supplied V3 has an empty OVERHEAD node; no overhead fixtures were exported.' };
  root.updateMatrixWorld(true);

  return { root, walls, overhead, cameras, cameraDefs, seats, config, bounds, floorHeight,
    sourceAsset: assetUrl, sourceZip: SOURCE_ZIP, sourceMember: SOURCE_MEMBER,
    sourceScene: gltf.scene, getNodeByOriginalName: (name) => lookup.get(name) || null };
}
