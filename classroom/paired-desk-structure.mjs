import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

const VERSION = 'paired-middle-desk-structure-1';

/** Static detail only; the original desks, equipment, chairs and buffers stay intact. */
export function applyPairedDeskStructure(classroom) {
  const root = classroom?.root;
  if (!root || typeof classroom.getNodeByOriginalName !== 'function') throw new Error('Paired desk details require the V3 classroom adapter.');
  if (root.userData.pairedDeskStructure?.version === VERSION) return root.userData.pairedDeskStructure;
  root.updateMatrixWorld(true);
  const inverseRoot = root.matrixWorld.clone().invert();
  const holder = new THREE.Group(); holder.name = 'Paired_Middle_Desk_Structure_Detail';
  const material = (name, color, extra = {}) => {
    const value = new THREE.MeshStandardMaterial({ color, roughness: 0.80, ...extra });
    value.name = `PairedDesk_${name}`; return value;
  };
  const materials = {
    steel: material('FrameSteel', '#a9b3b0', { metalness: 0.32, roughness: 0.64 }),
    screws: material('Fasteners', '#687570', { metalness: 0.48, roughness: 0.56 }),
    layer: material('LaminateEdge', '#8a713c', { side: THREE.DoubleSide, roughness: 0.86 }),
    storage: material('DarkStorage', '#3d474d', { roughness: 0.93 }),
    plastic: material('FanIvory', '#d6ddda', { roughness: 0.78 }),
    grille: material('FanGrille', '#b6c0bd', { metalness: 0.18, roughness: 0.70 }),
    cable: material('Cable', '#343e41', { roughness: 0.94 }),
  };
  function rootBox(object) { return new THREE.Box3().setFromObject(object).applyMatrix4(inverseRoot); }
  function rounded(size, radius = 0.006) {
    return new RoundedBoxGeometry(...size, 2, Math.min(radius, ...size.map((value) => value / 3)));
  }
  function boxRecord(box) { return { min: box.min.toArray(), max: box.max.toArray(), size: box.getSize(new THREE.Vector3()).toArray() }; }
  const audit = { version: VERSION, source: 'IMG_1993.JPG_reference_static_underdesk_detail',
    originalFurnitureTransformsChanged: false, originalGeometryChanged: false, tabletopSizePreserved: [1, 0.52],
    desktops: [], meshCount: 0, materialCount: Object.keys(materials).length,
    notes: 'Underdesk fans only. Desk-top equipment is handled separately. Dimensions of new small props are visual estimates.' };

  for (const identity of ['C_Mw', 'C_Md']) {
    const desk = classroom.getNodeByOriginalName(`Yellow 1m desk / ${identity}`);
    const tabletop = classroom.getNodeByOriginalName(`Yellow 1m desk / ${identity}.WhiteTop`);
    const shelf = classroom.getNodeByOriginalName(`Audit3.Center.${identity}.CubbyFloor`);
    if (!desk || !tabletop || !shelf) throw new Error(`Required source desk surface missing: ${identity}`);
    const top = rootBox(tabletop), shelfBox = rootBox(shelf), centre = top.getCenter(new THREE.Vector3());
    const width = top.max.x - top.min.x, depth = top.max.z - top.min.z;
    const group = new THREE.Group(); group.name = `PairedDeskStructure_${identity}`;
    group.position.set(centre.x, 0, centre.z); holder.add(group);
    const batches = new Map(), taggedBounds = new Map();
    function add(geometry, mat, position, rotation = [0, 0, 0], scale = [1, 1, 1], tag = 'hardware') {
      const prepared = geometry.index ? geometry.toNonIndexed() : geometry;
      if (prepared !== geometry) geometry.dispose();
      const matrix = new THREE.Matrix4().compose(new THREE.Vector3(...position),
        new THREE.Quaternion().setFromEuler(new THREE.Euler(...rotation)), new THREE.Vector3(...scale));
      prepared.applyMatrix4(matrix); prepared.computeBoundingBox();
      if (!taggedBounds.has(tag)) taggedBounds.set(tag, new THREE.Box3());
      taggedBounds.get(tag).union(prepared.boundingBox);
      if (!batches.has(mat)) batches.set(mat, []);
      batches.get(mat).push(prepared);
    }
    const addBox = (size, position, mat, tag = 'hardware', radius = 0.004) => add(rounded(size, radius), mat, position, [0, 0, 0], [1, 1, 1], tag);
    function rod(first, second, radius, mat, tag) {
      const a = new THREE.Vector3(...first), b = new THREE.Vector3(...second), direction = b.clone().sub(a);
      const geometry = new THREE.CylinderGeometry(radius, radius, direction.length(), 6);
      geometry.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize()));
      const midpoint = a.add(b).multiplyScalar(0.5);
      add(geometry, mat, midpoint.toArray(), [0, 0, 0], [1, 1, 1], tag);
    }

    // Sub-millimetre face offsets avoid coplanar flicker; source tops remain 1 x .52 m.
    const edgeY = top.max.y - 0.014;
    for (const sign of [-1, 1]) {
      add(new THREE.PlaneGeometry(width - 0.03, 0.0025), materials.layer,
        [0, edgeY, sign * (depth / 2 + 0.0007)], [0, sign > 0 ? 0 : Math.PI, 0], [1, 1, 1], 'edge');
      add(new THREE.PlaneGeometry(depth - 0.03, 0.0025), materials.layer,
        [sign * (width / 2 + 0.0007), edgeY, 0], [0, sign * Math.PI / 2, 0], [1, 1, 1], 'edge');
    }
    // Small L plates sit inside the four steel uprights, below the tabletop.
    for (const side of [-1, 1]) for (const end of [-1, 1]) {
      const x = side * (width / 2 - 0.0835), z = end * (depth / 2 - 0.065);
      addBox([0.002, 0.050, 0.030], [x, top.min.y - 0.027, z], materials.steel);
      addBox([0.030, 0.002, 0.030], [x - side * 0.013, top.min.y - 0.0015, z], materials.steel);
      for (const dy of [-0.011, 0.011]) add(new THREE.CylinderGeometry(0.003, 0.003, 0.003, 8), materials.screws,
        [x - side * 0.0017, top.min.y - 0.027 + dy, z], [0, 0, Math.PI / 2]);
    }
    const shelfFront = shelfBox.max.z - centre.z;
    addBox([0.87, 0.016, 0.003], [0, shelfBox.max.y + 0.009, shelfFront + 0.0025], materials.steel, 'shelfLip', 0.001);

    // A closed dark fabric/plastic box on the existing shelf, with a small handle.
    const storageBottom = shelfBox.max.y + 0.003;
    const storageZ = -0.04;
    addBox([0.285, 0.120, 0.250], [-0.20, storageBottom + 0.060, storageZ], materials.storage, 'storage', 0.012);
    addBox([0.080, 0.012, 0.004], [-0.20, storageBottom + 0.072, storageZ + 0.127], materials.screws, 'storage', 0.003);

    // The photo's larger white fans hang BELOW the book shelf, away from chairs.
    const fanX = 0.18, fanY = shelfBox.min.y - 0.210, fanZ = depth / 2 - 0.076;
    add(new THREE.CylinderGeometry(0.109, 0.109, 0.046, 24), materials.plastic,
      [fanX, fanY, fanZ], [Math.PI / 2, 0, 0], [1, 1, 1], 'fanBody');
    add(new THREE.TorusGeometry(0.112, 0.004, 6, 32), materials.plastic,
      [fanX, fanY, fanZ + 0.029], [0, 0, 0], [1, 1, 1], 'fanBody');
    for (let blade = 0; blade < 3; blade++) {
      const angle = blade * Math.PI * 2 / 3 + 0.25;
      add(new THREE.SphereGeometry(1, 10, 6), materials.plastic,
        [fanX + Math.sin(angle) * 0.045, fanY + Math.cos(angle) * 0.045, fanZ + 0.025],
        [0, 0, -angle - 0.35], [0.027, 0.060, 0.003], 'fanBody');
    }
    const faceZ = fanZ + 0.036;
    for (const radius of [0.037, 0.071, 0.105]) add(new THREE.TorusGeometry(radius, 0.0013, 5, 32), materials.grille,
      [fanX, fanY, faceZ], [0, 0, 0], [1, 1, 1], 'fanBody');
    for (let spoke = 0; spoke < 16; spoke++) {
      const angle = spoke * Math.PI * 2 / 16;
      rod([fanX + Math.cos(angle) * 0.019, fanY + Math.sin(angle) * 0.019, faceZ],
        [fanX + Math.cos(angle) * 0.108, fanY + Math.sin(angle) * 0.108, faceZ], 0.0011, materials.grille, 'fanBody');
    }
    add(new THREE.SphereGeometry(1, 12, 8), materials.plastic,
      [fanX, fanY, faceZ + 0.002], [0, 0, 0], [0.018, 0.018, 0.007], 'fanBody');
    addBox([0.075, 0.010, 0.045], [fanX, shelfBox.min.y - 0.0055, fanZ - 0.005], materials.plastic, 'fanMount', 0.003);
    rod([fanX, shelfBox.min.y - 0.0105, fanZ - 0.018], [fanX, fanY + 0.097, fanZ - 0.018],
      0.006, materials.plastic, 'fanMount');

    // Route around the shelf's front edge, rather than through its solid floor.
    const cableZ = depth / 2 - 0.025;
    const cable = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.30, top.min.y - 0.006, cableZ),
      new THREE.Vector3(0.30, shelfBox.max.y + 0.040, cableZ),
      new THREE.Vector3(0.30, shelfBox.min.y - 0.035, cableZ),
      new THREE.Vector3(fanX, fanY + 0.120, fanZ + 0.010),
      new THREE.Vector3(fanX, fanY + 0.070, fanZ - 0.018),
    ], false, 'centripetal');
    add(new THREE.TubeGeometry(cable, 22, 0.0015, 6, false), materials.cable,
      [0, 0, 0], [0, 0, 0], [1, 1, 1], 'cable');

    for (const [mat, geometries] of batches) {
      const merged = mergeGeometries(geometries, false);
      if (!merged) throw new Error(`Cannot merge paired desk details: ${identity}/${mat.name}`);
      const mesh = new THREE.Mesh(merged, mat); mesh.name = `${group.name}_${mat.name}`;
      mesh.castShadow = mat !== materials.layer; mesh.receiveShadow = true; group.add(mesh);
      geometries.forEach((geometry) => geometry.dispose()); audit.meshCount++;
    }
    const componentBounds = Object.fromEntries([...taggedBounds].map(([tag, bounds]) => [tag, boxRecord(bounds.clone().translate(group.position))]));
    const detailBounds = new THREE.Box3(); for (const bounds of taggedBounds.values()) detailBounds.union(bounds);
    const chair = classroom.getNodeByOriginalName(`Chair / ${identity}`), previous = identity === 'C_Mw' ? 'C_Fw' : 'C_Fd';
    const chairBox = chair ? rootBox(chair) : null, previousChair = classroom.getNodeByOriginalName(`Chair / ${previous}`);
    const total = detailBounds.clone().translate(group.position), storage = componentBounds.storage, fan = componentBounds.fanBody;
    audit.desktops.push({ identity, groupName: group.name, tabletop: boxRecord(top), shelf: boxRecord(shelfBox),
      bounds: boxRecord(total), components: componentBounds,
      clearance: { fanBodyBelowShelf: shelfBox.min.y - fan.max[1], mountBelowShelf: 0.0005,
        storageAboveShelf: storage.min[1] - shelfBox.max.y, storageBelowDesktop: top.min.y - storage.max[1],
        toOwnChairZ: chairBox ? chairBox.min.z - total.max.z : null,
        toPreviousChairZ: previousChair ? total.min.z - rootBox(previousChair).max.z : null } });
  }
  root.add(holder); root.updateMatrixWorld(true);
  audit.bounds = boxRecord(new THREE.Box3().setFromObject(holder).applyMatrix4(root.matrixWorld.clone().invert()));
  root.userData.pairedDeskStructure = audit;
  return audit;
}
