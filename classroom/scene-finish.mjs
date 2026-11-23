import * as THREE from 'three';

const VERSION = 'plain-green-board-and-cleaning-plants-swap-v1';
const NAMES = Object.freeze({
  board: 'Audit3 / actual video blackboard surface',
  boardBacking: 'Audit3 / Blackboard solid green backing',
  platform: 'Audit3 / platform rebuilt from four observed corners',
  bin: 'Blue bin on blackboard end of platform',
  tools: 'Front blue bucket and cleaning rods',
  rubber: 'Rear rubber plant',
  shrub: 'Rear shrub',
});

/** Small, auditable runtime edits; the supplied GLB stays unchanged. */
export function applySceneFinish(classroom) {
  const { root, bounds } = classroom;
  if (root.userData.sceneFinish?.version === VERSION) return root.userData.sceneFinish;
  const find = (name) => {
    const object = classroom.getNodeByOriginalName(name);
    if (!object) throw new Error(`Scene finish requires source node: ${name}`);
    return object;
  };
  root.updateMatrixWorld(true);
  const inverseRoot = root.matrixWorld.clone().invert();
  const box = (object) => new THREE.Box3().setFromObject(object, true).applyMatrix4(inverseRoot);
  const compactBox = (value) => ({ min: value.min.toArray(), max: value.max.toArray() });
  const position = (object) => root.worldToLocal(object.getWorldPosition(new THREE.Vector3())).toArray();
  const board = find(NAMES.board);
  const backing = find(NAMES.boardBacking);
  const oldBoardMaterial = Array.isArray(board.material) ? board.material[0] : board.material;
  const greenMaterial = (Array.isArray(backing.material) ? backing.material[0] : backing.material).clone();
  greenMaterial.name = 'SceneFinish_Plain_Original_Green_Blackboard';
  for (const key of ['map', 'alphaMap', 'normalMap', 'bumpMap', 'emissiveMap', 'roughnessMap', 'metalnessMap']) {
    if (key in greenMaterial) greenMaterial[key] = null;
  }
  greenMaterial.emissive?.set(0x000000);
  greenMaterial.metalness = 0;
  greenMaterial.roughness = 0.97;
  greenMaterial.needsUpdate = true;
  board.material = greenMaterial;

  const objects = {
    bin: find(NAMES.bin), tools: find(NAMES.tools),
    rubber: find(NAMES.rubber), shrub: find(NAMES.shrub),
  };
  const before = Object.fromEntries(Object.entries(objects).map(([id, object]) => [id, {
    sourceName: NAMES[id], position: position(object), bounds: compactBox(box(object)),
  }]));
  const cleaning = new THREE.Group(); cleaning.name = 'SceneFinish_Rear_Cleaning_Corner';
  const plants = new THREE.Group(); plants.name = 'SceneFinish_Front_Door_Plants';
  root.add(cleaning, plants);
  // attach() preserves each complete object's world placement and all its parts.
  cleaning.attach(objects.bin); cleaning.attach(objects.tools);
  plants.attach(objects.rubber); plants.attach(objects.shrub);
  root.updateMatrixWorld(true);
  const cleaningBox = box(cleaning), plantsBox = box(plants);
  const cleaningCentre = cleaningBox.getCenter(new THREE.Vector3());
  const plantsCentre = plantsBox.getCenter(new THREE.Vector3());
  const nominalToFront = new THREE.Vector3(cleaningCentre.x - plantsCentre.x, 0, cleaningCentre.z - plantsCentre.z);
  const nominalToRear = nominalToFront.clone().negate();
  const platform = find(NAMES.platform);
  const floorHeight = Number.isFinite(classroom.floorHeight) ? classroom.floorHeight : 0.009;
  const ray = new THREE.Raycaster();
  const rootUp = new THREE.Vector3(0, 1, 0).transformDirection(root.matrixWorld);
  function supportHeight(x, z) {
    const origin = root.localToWorld(new THREE.Vector3(x, 10, z));
    ray.set(origin, rootUp.clone().negate());
    const hit = ray.intersectObject(platform, true)[0];
    return hit ? root.worldToLocal(hit.point.clone()).y : floorHeight;
  }
  function footprintSupport(footprint, offset) {
    const minX = footprint.min.x + offset.x, maxX = footprint.max.x + offset.x;
    const minZ = footprint.min.z + offset.z, maxZ = footprint.max.z + offset.z;
    const points = [[minX, minZ], [maxX, minZ], [minX, maxZ], [maxX, maxZ], [(minX + maxX) / 2, (minZ + maxZ) / 2]];
    const heights = points.map(([x, z]) => supportHeight(x, z));
    return { level: Math.max(...heights) - Math.min(...heights) < 0.01, height: Math.max(...heights), samples: heights };
  }
  function insideRoom(value) {
    return value.min.x >= bounds.minX + 0.08 && value.max.x <= bounds.maxX - 0.08 &&
      value.min.z >= bounds.minZ + 0.08 && value.max.z <= bounds.maxZ - 0.08;
  }
  const plantFeet = [find('Rear rubber plant_pot'), find('Rear shrub_pot')].map(box);
  const offsets = [];
  for (let ix = -12; ix <= 12; ix++) for (let iz = -8; iz <= 8; iz++) {
    offsets.push(new THREE.Vector3(ix * 0.05, 0, iz * 0.05));
  }
  offsets.sort((a, b) => a.lengthSq() - b.lengthSq());
  const plantOffset = offsets.map((extra) => nominalToFront.clone().add(extra)).find((candidate) =>
    insideRoom(plantsBox.clone().translate(candidate)) && plantFeet.every((footprint) => footprintSupport(footprint, candidate).level));
  if (!plantOffset) throw new Error('No level placement for the two front-door plant pots.');

  // The source leaves extend past the nominal rear wall. Keep the swapped
  // cleaning group inside that wall instead of copying this source overhang.
  const clampToRoom = (value, candidate) => {
    const moved = value.clone().translate(candidate);
    candidate.x += Math.max(0, bounds.minX + 0.10 - moved.min.x) - Math.max(0, moved.max.x - (bounds.maxX - 0.10));
    candidate.z += Math.max(0, bounds.minZ + 0.10 - moved.min.z) - Math.max(0, moved.max.z - (bounds.maxZ - 0.10));
    return candidate;
  };
  const rearNominal = clampToRoom(cleaningBox, nominalToRear.clone());
  const umbrella = classroom.getNodeByOriginalName('Closed white umbrella near rear plants');
  const umbrellaBox = umbrella ? box(umbrella).expandByScalar(0.025) : null;
  const cleaningPartBoxes = [box(objects.bin), box(objects.tools)];
  const rearOffset = offsets.map((extra) => rearNominal.clone().add(extra)).find((candidate) =>
    insideRoom(cleaningBox.clone().translate(candidate)) &&
    (!umbrellaBox || cleaningPartBoxes.every((part) => !part.clone().translate(candidate).intersectsBox(umbrellaBox))));
  if (!rearOffset) throw new Error('No clear rear placement for the bin and cleaning tools.');

  plants.position.add(plantOffset); cleaning.position.add(rearOffset);
  root.updateMatrixWorld(true);
  const landing = {};
  for (const [id, object] of Object.entries(objects)) {
    const currentBox = box(object);
    const footprint = id === 'rubber' ? box(find('Rear rubber plant_pot'))
      : id === 'shrub' ? box(find('Rear shrub_pot')) : currentBox;
    const centre = footprint.getCenter(new THREE.Vector3());
    const support = supportHeight(centre.x, centre.z);
    const dy = support + 0.003 - currentBox.min.y;
    // These new groups share the returned room frame, so this Y is metre-based.
    object.position.y += dy;
    root.updateMatrixWorld(true);
    landing[id] = { supportHeight: support, bottomHeight: box(object).min.y, clearance: 0.003 };
  }
  const after = Object.fromEntries(Object.entries(objects).map(([id, object]) => [id, {
    sourceName: NAMES[id], position: position(object), bounds: compactBox(box(object)), ...landing[id],
  }]));
  const audit = {
    version: VERSION,
    board: { sourceNode: NAMES.board, previousMaterial: oldBoardMaterial?.name || null,
      previousTexture: oldBoardMaterial?.map?.name || null, material: greenMaterial.name,
      colorLinear: greenMaterial.color.toArray(), colorSRGBHex: `#${greenMaterial.color.getHexString()}`,
      hasTexture: Boolean(greenMaterial.map), framePreserved: true },
    swap: { before, after, frontGroupTranslation: plantOffset.toArray(), rearGroupTranslation: rearOffset.toArray(),
      relativeXZPreserved: true, floorClearance: 0.003, movedSourceGroups: Object.values(NAMES).filter((name) =>
        [NAMES.bin, NAMES.tools, NAMES.rubber, NAMES.shrub].includes(name)),
      untouched: ['rear umbrella', 'loose network cables', 'walls', 'doors', 'cabinets', 'platform'] },
  };
  root.userData.sceneFinish = audit;
  return audit;
}
