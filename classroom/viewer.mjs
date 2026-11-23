import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { loadClassroomModel } from './asset-model.mjs';
import { createCameraModels } from './camera-models.mjs';
import { createCameraNavigation } from './camera-navigation.mjs';
import { createCameraPicking } from './camera-picking.mjs';
import { applySceneRepairs } from './scene-repairs.mjs';
import { applyCurtainClearance } from './curtain-clearance.mjs';
import { applySceneFinish } from './scene-finish.mjs';
import { applySceneAppearance } from './scene-appearance.mjs';
import { applyPairedWorkstations } from './paired-workstations.mjs';
import { applyPairedDeskStructure } from './paired-desk-structure.mjs';

const viewport = document.querySelector('#viewport');
const canvas = document.querySelector('#scene');
const status = document.querySelector('#model-status');

function showLoadError(error) {
  console.error('Classroom V3 viewer failed:', error);
  status.hidden = false;
  status.textContent = '三维模型加载失败，请刷新页面重试。';
  document.body.dataset.ready = 'error';
}

async function initialize() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#727e84');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(2);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.92;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.shadowMap.autoUpdate = false;
  renderer.shadowMap.needsUpdate = true;

  const environment = new RoomEnvironment();
  const pmrem = new THREE.PMREMGenerator(renderer);
  const environmentTarget = pmrem.fromScene(environment, 0.04);
  scene.environment = environmentTarget.texture;
  scene.environmentIntensity = 0.36;
  environment.dispose();
  pmrem.dispose();
  scene.add(new THREE.HemisphereLight('#e2edff', '#74664e', 1.00));
  scene.add(new THREE.AmbientLight('#ffffff', 0.10));
  const key = new THREE.DirectionalLight('#fff2dd', 2.4);
  key.position.set(-6, 11, -1);
  key.castShadow = true;
  key.shadow.mapSize.set(4096, 4096);
  Object.assign(key.shadow.camera, { left: -7, right: 7, top: 8, bottom: -8, near: 0.5, far: 30 });
  key.shadow.normalBias = 0.012;
  key.shadow.bias = -0.00015;
  key.shadow.radius = 3.5;
  scene.add(key);
  const fill = new THREE.DirectionalLight('#dce9ff', 0.68);
  fill.position.set(7, 6, -5);
  scene.add(fill);

  const classroom = await loadClassroomModel();
  applySceneRepairs(classroom);
  applyCurtainClearance(classroom);
  applySceneFinish(classroom);
  applyPairedWorkstations(classroom);
  applyPairedDeskStructure(classroom);
  applySceneAppearance(classroom);
  const cameraModels = createCameraModels(classroom);
  scene.add(classroom.root);
  if (classroom.overhead) classroom.overhead.visible = false;
  classroom.cameras.visible = true;
  for (const item of classroom.cameras.children) item.visible = true;
  const anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  classroom.root.traverse((object) => {
    if (!object.isMesh) return;
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    object.castShadow = !materials.some((material) => material.transparent) &&
      !/Screen_generic_UI|Webcam|FloorTile|FloorGrout/.test(object.name);
    object.receiveShadow = true;
    for (const material of materials) {
      if (material.map) material.map.anisotropy = anisotropy;
    }
  });
  classroom.getNodeByOriginalName('Floor and tile grid')?.traverse((object) => {
    if (object.isMesh) object.castShadow = false;
  });
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(150, 150),
    new THREE.ShadowMaterial({ color: '#69817c', opacity: 0.12 }));
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.13;
  ground.receiveShadow = true;
  scene.add(ground);

  const camera = new THREE.PerspectiveCamera(42, 1, 0.15, 100);
  camera.position.set(10, 14, 14);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 0.7, (classroom.bounds.minZ + classroom.bounds.maxZ) / 2);
  controls.rotateSpeed = 0.45;
  controls.enableDamping = true;
  controls.dampingFactor = 0.09;
  controls.minPolarAngle = 0.01;
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.screenSpacePanning = true;
  const zoomLimits = { min: 160, max: 350, default: 175 };

  const target = new THREE.WebGLRenderTarget(1, 1, {
    type: THREE.HalfFloatType, samples: Math.min(4, renderer.capabilities.maxSamples),
  });
  const composer = new EffectComposer(renderer, target);
  composer.addPass(new RenderPass(scene, camera));
  const contactAO = new GTAOPass(scene, camera, 512, 512, undefined,
    { radius: 0.18, distanceExponent: 1.4, thickness: 0.25, scale: 1 },
    { radius: 2.5, rings: 2, samples: 8 });
  contactAO.blendIntensity = 0.44;
  composer.addPass(contactAO);
  composer.addPass(new OutputPass());

  const selectionHalo = new THREE.Mesh(new THREE.SphereGeometry(0.19, 12, 8),
    new THREE.MeshBasicMaterial({ color: '#64d5a4', wireframe: true, transparent: true, opacity: 0.8, depthTest: false }));
  selectionHalo.visible = false;
  selectionHalo.renderOrder = 10;
  scene.add(selectionHalo);

  let selectedCamera = null;
  let activeView = 'overview';
  let viewportWidth = 0;
  let viewportHeight = 0;
  let needsFraming = true;
  let needsRender = true;
  let frameId = 0;
  let disposed = false;
  let suspended = false;
  const framingPointCache = new Map();
  const projected = new THREE.Vector3();
  const forward = new THREE.Vector3();

  const labels = classroom.cameraDefs.map((definition, index) => {
    const id = `cam${index + 1}`;
    const caption = `C${index + 1} · ${definition.label.split('·').at(-1).trim()}`;
    const label = document.createElement('button');
    label.type = 'button';
    label.className = 'camera-label';
    label.textContent = `C${index + 1}`;
    label.title = caption;
    label.style.setProperty('--camera-color', definition.color);
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label.textContent;
    button.title = caption;
    for (const control of [label, button]) {
      control.setAttribute('aria-label', caption);
      control.setAttribute('aria-pressed', 'false');
      control.addEventListener('click', () => selectCamera(id));
    }
    document.querySelector('#camera-labels').append(label);
    document.querySelector('#camera-presets').append(button);
    return { id, label, button, definition };
  });
  const cameraNavigation = createCameraNavigation({
    camera, controls, classroom, cameraModels, invalidate,
    onStart() {
      activeView = 'camera';
      camera.clearViewOffset();
      needsFraming = false;
    },
  });
  const cameraPicking = createCameraPicking({
    canvas, camera, classroom, cameraModels, onChoose: selectCamera,
    onInteract: startInteraction,
  });

  function selectCamera(id) {
    selectedCamera = selectedCamera === id ? null : id;
    const selection = labels.find((entry) => entry.id === selectedCamera);
    selectionHalo.visible = Boolean(selection);
    if (selection) {
      selectionHalo.position.fromArray(selection.definition.position);
      cameraNavigation.goTo(selectedCamera);
    } else {
      cameraNavigation.cancel();
    }
    for (const { label, button, id } of labels) {
      const pressed = String(id === selectedCamera);
      label.setAttribute('aria-pressed', pressed);
      button.setAttribute('aria-pressed', pressed);
    }
    invalidate();
  }

  function updateLabels() {
    camera.getWorldDirection(forward);
    for (const { label, definition } of labels) {
      projected.fromArray(definition.position);
      projected.y += 0.22;
      const inFront = projected.clone().sub(camera.position).dot(forward) > 0;
      projected.project(camera);
      label.hidden = !inFront || projected.z <= -1 || projected.z >= 1 ||
        Math.abs(projected.x) > 1.02 || Math.abs(projected.y) > 1.02;
      if (label.hidden) continue;
      const x = (projected.x * 0.5 + 0.5) * viewportWidth;
      const y = (-projected.y * 0.5 + 0.5) * viewportHeight;
      label.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`;
    }
  }

  function updateCutaway() {
    const { minX, maxX, minZ, maxZ } = classroom.bounds;
    const visible = {
      front: camera.position.z >= minZ - 0.08, back: camera.position.z <= maxZ + 0.08,
      left: camera.position.x >= minX - 0.08, right: camera.position.x <= maxX + 0.08,
    };
    for (const [side, wall] of Object.entries(classroom.walls)) {
      if (wall.visible === visible[side]) continue;
      wall.visible = visible[side];
      needsRender = true;
      renderer.shadowMap.needsUpdate = true;
    }
  }

  function visibleModelPoints() {
    const key = Object.values(classroom.walls).map((wall) => Number(wall.visible)).join('');
    if (framingPointCache.has(key)) return framingPointCache.get(key);
    const points = [];
    classroom.root.updateMatrixWorld(true);
    classroom.root.traverseVisible((object) => {
      if (!object.isMesh) return;
      if (object.isInstancedMesh) object.computeBoundingBox();
      else if (!object.geometry.boundingBox) object.geometry.computeBoundingBox();
      const box = object.isInstancedMesh ? object.boundingBox : object.geometry.boundingBox;
      for (const x of [box.min.x, box.max.x]) {
        for (const y of [box.min.y, box.max.y]) {
          for (const z of [box.min.z, box.max.z]) {
            points.push(new THREE.Vector3(x, y, z).applyMatrix4(object.matrixWorld));
          }
        }
      }
    });
    framingPointCache.set(key, points);
    return points;
  }

  function updateVerticalFraming() {
    if (!needsFraming) return;
    camera.clearViewOffset();
    camera.updateMatrixWorld(true);
    let top = Infinity;
    let bottom = -Infinity;
    const point = new THREE.Vector3();
    for (const worldPoint of visibleModelPoints()) {
      point.copy(worldPoint).project(camera);
      if (point.z <= -1 || point.z >= 1) continue;
      const y = (1 - point.y) * viewportHeight / 2;
      top = Math.min(top, y);
      bottom = Math.max(bottom, y);
    }
    if (Number.isFinite(top + bottom)) {
      const upwardPixels = Math.min(48, viewportHeight * 0.05,
        Math.max(0, top - 8), Math.max(0, bottom - viewportHeight + 8));
      if (upwardPixels > 0) {
        camera.setViewOffset(viewportWidth, viewportHeight, 0, upwardPixels, viewportWidth, viewportHeight);
      }
    }
    needsFraming = false;
  }

  function fitOverview() {
    needsFraming = true;
    const back = new THREE.Vector3(10, 14, 14).normalize();
    const right = new THREE.Vector3(0, 1, 0).cross(back).normalize();
    const up = back.clone().cross(right).normalize();
    updateCutaway();
    const points = visibleModelPoints();
    if (!points.length) return;
    const low = new THREE.Vector3(Infinity, Infinity, Infinity);
    const high = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    const coordinates = new THREE.Vector3();
    for (const point of points) {
      coordinates.set(point.dot(right), point.dot(up), point.dot(back));
      low.min(coordinates);
      high.max(coordinates);
    }
    const centre = low.add(high).multiplyScalar(0.5);
    const focus = right.clone().multiplyScalar(centre.x).addScaledVector(up, centre.y).addScaledVector(back, centre.z);
    const halfV = THREE.MathUtils.degToRad(camera.fov) / 2;
    const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
    const offset = new THREE.Vector3();
    let distance = 0;
    for (const point of points) {
      offset.subVectors(point, focus);
      distance = Math.max(distance, offset.dot(back) + Math.abs(offset.dot(right)) / Math.tan(halfH),
        offset.dot(back) + Math.abs(offset.dot(up)) / Math.tan(halfV));
    }
    controls.target.copy(focus);
    const overviewDistance = distance * 1.12;
    controls.minDistance = overviewDistance * 100 / zoomLimits.max;
    controls.maxDistance = overviewDistance * 100 / zoomLimits.min;
    camera.position.copy(focus).addScaledVector(back, overviewDistance * 100 / zoomLimits.default);
  }

  function startInteraction() {
    cameraNavigation.cancel();
    activeView = 'free';
  }

  function constrainPan() {
    const { minX, maxX, minZ, maxZ } = classroom.bounds;
    const focus = controls.target.clone().clamp(
      new THREE.Vector3(minX * 0.6, 0.25, minZ * 0.6),
      new THREE.Vector3(maxX * 0.6, 2.1, maxZ * 0.6));
    const offset = focus.clone().sub(controls.target);
    if (offset.lengthSq() < 1e-10) return false;
    controls.target.copy(focus);
    camera.position.add(offset);
    return true;
  }

  function requestFrame() {
    if (!frameId && !disposed && !suspended && !document.hidden && viewportWidth && viewportHeight) {
      frameId = requestAnimationFrame(renderFrame);
    }
  }

  function invalidate() {
    needsRender = true;
    requestFrame();
  }

  function renderFrame(now) {
    frameId = 0;
    const transitioning = cameraNavigation.isAnimating();
    cameraNavigation.update(now);
    const moved = controls.update();
    const constrained = constrainPan();
    updateCutaway();
    if (needsRender || moved || constrained || transitioning) {
      updateVerticalFraming();
      composer.render();
      updateLabels();
      needsRender = false;
      if (document.body.dataset.ready !== 'true') {
        document.body.dataset.ready = 'true';
        status.hidden = true;
      }
    }
    if (moved || cameraNavigation.isAnimating()) requestFrame();
  }

  function resize() {
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    if (!width || !height || (width === viewportWidth && height === viewportHeight)) return;
    viewportWidth = width;
    viewportHeight = height;
    needsFraming = activeView !== 'camera';
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    composer.setSize(width, height);
    if (activeView === 'overview') fitOverview();
    controls.update();
    invalidate();
  }

  function cancelFrame() {
    if (frameId) cancelAnimationFrame(frameId);
    frameId = 0;
  }

  controls.addEventListener('change', invalidate);
  controls.addEventListener('start', startInteraction);
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(viewport);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) cancelFrame();
    else { resize(); invalidate(); }
  });
  window.addEventListener('pagehide', (event) => {
    suspended = true;
    cancelFrame();
    if (event.persisted) return;
    disposed = true;
    resizeObserver.disconnect();
    cameraNavigation.cancel();
    cameraPicking.dispose();
    controls.dispose();
    contactAO.dispose();
    composer.dispose();
    environmentTarget.dispose();
    renderer.dispose();
  });
  window.addEventListener('pageshow', () => {
    suspended = false;
    resize();
    invalidate();
  });
  resize();
}

initialize().catch(showLoadError);
