const VERSION = 'classroom-v3-layered-warm-materials-2';
const PALETTE_STRENGTH = 0.80;

// Colors are sRGB inputs; Three.js converts them to its linear working space.
// Floor colors are deliberately gentle multipliers: the four original tile
// photographs already average roughly RGB(186,177,163), and remain attached.
const PALETTE = {
  wall: { color: '#c9c3b6', roughness: 0.92, metalness: 0 },
  wallWear: { color: '#b3ad9f', roughness: 0.96, metalness: 0 },
  floor0: { color: '#dedbd2', roughness: 0.81, metalness: 0 },
  floor1: { color: '#ddd9d0', roughness: 0.82, metalness: 0 },
  floor2: { color: '#dfdcd4', roughness: 0.81, metalness: 0 },
  floor3: { color: '#dcd8ce', roughness: 0.83, metalness: 0 },
  platform: { color: '#dbd3c4', roughness: 0.82, metalness: 0 },
  grout: { color: '#70685a', roughness: 0.96, metalness: 0 },
  goldenDesktop: { color: '#c49a47', roughness: 0.56, metalness: 0 },
  cabinetBody: { color: '#b1c0c6', roughness: 0.66, metalness: 0.02 },
  cabinetFace: { color: '#bdc9ce', roughness: 0.62, metalness: 0.02 },
  coolDesktop: { color: '#c3ced1', roughness: 0.59, metalness: 0.015 },
  toolsDesktop: { color: '#b9c7cb', roughness: 0.65, metalness: 0.015 },
  tablePanels: { color: '#869196', roughness: 0.68, metalness: 0.12 },
  cubbyInterior: { color: '#505b5d', roughness: 0.85, metalness: 0.04 },
  steelFrame: { color: '#5b686f', roughness: 0.48, metalness: 0.42 },
  chrome: { color: '#8e9ba0', roughness: 0.34, metalness: 0.72 },
  starMetal: { color: '#76858c', roughness: 0.36, metalness: 0.65 },
  handleMetal: { color: '#8b989e', roughness: 0.39, metalness: 0.62 },
  windowMetal: { color: '#879399', roughness: 0.43, metalness: 0.52 },
  windowPaint: { color: '#bcc5c4', roughness: 0.73, metalness: 0.025 },
  radiatorPaint: { color: '#b7b9b0', roughness: 0.78, metalness: 0.025 },
  pelmet: { color: '#c5c0b2', roughness: 0.84, metalness: 0 },
  curtain: { color: '#657b86', roughness: 0.94, metalness: 0 },
};

const EXACT = new Map([
  ['Audit3 / neutral used white wall paint', 'wall'],
  ['Room / faint wall wear', 'wallWear'],
  ['Floor / grout', 'grout'],
  ['Video reference / lemon desktop', 'goldenDesktop'],
  ['Video reference / pale grey tools laminate', 'toolsDesktop'],
  ['Audit3 / actual neutral grey curtain cloth', 'curtain'],
  ['Audit3 / warm off-white pelmet', 'pelmet'],
  ['RoughFurniture.steel', 'steelFrame'],
  ['RoughFurniture.chrome', 'chrome'],
  ['RoughFurniture.IdentifiedChair.StarMetal', 'starMetal'],
  ['RoughFurniture.FixtureSilver', 'handleMetal'],
  ['Audit3 / dull aluminium', 'windowMetal'],
  ['Audit3.Center.CubbyInterior', 'cubbyInterior'],
]);

function categoryFor(materialName, meshName) {
  // Preserve existing identification colors, including the freshly plain green
  // board, all camera markers, upholstery, computers, plants and colored props.
  if (/^(SceneFinish_|CCTV_|SeatSymbol_)/.test(materialName) ||
      /blackboard|chalkboard|camera|^Rear rubber plant|^Rear shrub/i.test(meshName)) return null;
  const floor = materialName.match(/^Floor \/ beige ([0-3])$/);
  if (floor) return /platform rebuilt/i.test(meshName) ? 'platform' : `floor${floor[1]}`;
  if (materialName === 'RoughFurniture.FixtureWhite') {
    if (/\.Desktop$/.test(meshName)) return 'coolDesktop';
    return /\.Drawer\d/.test(meshName) ? 'cabinetFace' : 'cabinetBody';
  }
  // These originals are shared with fans, bottle caps, paper bags and camera
  // housings. Only the explicitly reviewed cabinet parts receive the new color.
  if (materialName === 'Room / painted white') {
    if (/^(Front window short drawer cabinet|Printer white support cabinet|Rear mobile short drawer cabinet)$/.test(meshName)) return 'cabinetBody';
    if (/^(Printer cabinet front door|Mobile drawer front(?:\.\d+)?)$/.test(meshName)) return 'cabinetFace';
    return null;
  }
  if (materialName === 'Audit3Window.WhitePlastic') {
    return /^Audit3Window\.front\.small\.cabinet\.(lower\.door|upper\.drawer)$/.test(meshName) ? 'cabinetFace' : null;
  }
  if (materialName === 'Audit3.Center.WarmGraySteel') {
    return /SteelLeg|Brace|Crossbar|Apron/.test(meshName) ? 'steelFrame' : 'tablePanels';
  }
  if (materialName === 'Audit3Window.DeskBookboxGrey') return /bookbox/i.test(meshName) ? 'tablePanels' : null;
  if (materialName === 'Audit3 / aged white architectural paint') return /radiator|heating pipe/i.test(meshName) ? 'radiatorPaint' : 'windowPaint';
  return EXACT.get(materialName) || null;
}

/** Reassign reviewed surface colors at runtime, preserving the source materials. */
export function applySceneAppearance(classroom) {
  const root = classroom?.root;
  if (!root?.isObject3D) throw new TypeError('Scene appearance requires the classroom root.');
  if (root.userData.sceneAppearance?.version === VERSION) return root.userData.sceneAppearance;
  const cache = new Map(), mappings = new Map(), categories = {};
  const originals = new Set(), preserved = new Set();
  let meshesChanged = 0, materialSlotsChanged = 0;
  const nameOf = (object) => object.userData?.name || object.name || '';
  function replacement(original, category, meshName) {
    let variants = cache.get(original);
    if (!variants) { variants = new Map(); cache.set(original, variants); }
    if (!variants.has(category)) {
      const copy = original.clone(), settings = PALETTE[category];
      copy.name = `SceneAppearance_${category}_${original.name}`;
      copy.color.set(settings.color).lerp(original.color, 1 - PALETTE_STRENGTH);
      if ('roughness' in copy) copy.roughness = settings.roughness * PALETTE_STRENGTH + original.roughness * (1 - PALETTE_STRENGTH);
      if ('metalness' in copy) copy.metalness = settings.metalness * PALETTE_STRENGTH + original.metalness * (1 - PALETTE_STRENGTH);
      copy.userData = { ...copy.userData, appearanceCategory: category, appearanceOriginalName: original.name };
      copy.needsUpdate = true;
      // Texture pointers, UV transforms, opacity, side and all geometry are
      // inherited unchanged. In particular, screens and board textures are not
      // reintroduced or cleared by this module.
      variants.set(category, copy);
      mappings.set(copy, { originalMaterial: original.name, category, colorSRGB: `#${copy.color.getHexString()}`,
        targetColorSRGB: settings.color, roughness: copy.roughness, metalness: copy.metalness,
        previousColorSRGB: `#${original.color.getHexString()}`,
        textureRetained: copy.map === original.map, hasTexture: Boolean(copy.map), meshCount: 0, examples: [] });
    }
    const result = variants.get(category), mapping = mappings.get(result);
    mapping.meshCount++;
    if (mapping.examples.length < 4) mapping.examples.push(meshName);
    return result;
  }
  root.traverse((object) => {
    if (!object.isMesh || !object.material) return;
    const meshName = nameOf(object), surfaces = Array.isArray(object.material) ? object.material : [object.material];
    let changed = false;
    const assigned = surfaces.map((surface) => {
      originals.add(surface);
      if (!surface?.color || typeof surface.clone !== 'function') return surface;
      const category = categoryFor(surface.name || '', meshName);
      if (!category) { preserved.add(surface.name || '(unnamed)'); return surface; }
      changed = true; materialSlotsChanged++;
      categories[category] = (categories[category] || 0) + 1;
      return replacement(surface, category, meshName);
    });
    if (changed) { object.material = Array.isArray(object.material) ? assigned : assigned[0]; meshesChanged++; }
  });
  const audit = {
    version: VERSION, method: 'reviewed_original_material_names_and_shared_material_mesh_roles',
    reviewedGLBMaterialCatalogCount: 167, uniqueMaterialInstancesBefore: originals.size,
    sourceMaterialsMutated: false, geometryChanged: false, layoutChanged: false, paletteStrength: PALETTE_STRENGTH,
    meshesChanged, materialSlotsChanged, clonedMaterialVariants: mappings.size,
    categories, mappings: [...mappings.values()], palette: Object.fromEntries(Object.entries(PALETTE).map(([name, value]) => [name, { ...value }])),
    preservedMaterialNames: [...preserved].sort(),
    preservedIdentities: ['black/ivory/colored chair upholstery', 'computer screens and equipment colors', 'plants and planters', 'plain green blackboard', 'camera bodies/lenses', 'doors and colored props'],
    floorTexturePolicy: 'Four source tile photographs retained; warm-neutral multipliers deepen the stone colors without removing texture variation.',
  };
  root.userData.sceneAppearance = audit;
  return audit;
}
