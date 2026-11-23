import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

const VERSION = 'paired-workstations-photo-3-shorter-uprights';
const TOP = 0.764;

/** Visible forms from IMG_1993; dimensions are fitted to the unchanged 1 m desks. */
export function applyPairedWorkstations(classroom) {
  if (classroom.root.userData.pairedWorkstations?.version === VERSION) return classroom.root.userData.pairedWorkstations;
  const root = new THREE.Group(); root.name = 'PhotoRefined_Central_Paired_Workstations';
  classroom.root.add(root);
  const materials = {}, geometryCache = new Map();
  const mat = (name, color, roughness = .6, metalness = 0) => {
    const m = new THREE.MeshStandardMaterial({ color, roughness, metalness });
    m.name = `PairPhoto_${name}`; materials[name] = m; return m;
  };
  const black = mat('BlackPlastic', '#171b1d', .48), bezel = mat('DisplayBezel', '#101315', .4);
  const silver = mat('BrushedAluminium', '#9aa5a8', .36, .65), white = mat('IvoryPlastic', '#e2e4df', .66);
  const pale = mat('Keycaps', '#e0dfd8', .68), pink = mat('PinkKeycaps', '#df97ad', .7);
  const matPink = mat('PinkDeskMat', '#d995ad', .96), mousePink = mat('RoseMouse', '#d63d83', .46);
  const charcoal = mat('LaptopShell', '#2e3338', .4, .2), darkKey = mat('LaptopKeys', '#15191d', .6);
  const green = mat('PhoneCase', '#a5bc4b', .7), yellow = mat('AstronautTrim', '#d1ab43', .5);
  const visor = mat('AstronautVisor', '#25323a', .19, .68), rubber = mat('SoftRubber', '#303737', .95);
  const cableBlack = mat('BlackCable', '#171a1d', .82), cableWhite = mat('WhiteCable', '#bfc6c5', .82);
  function box(parent, name, size, position, material, radius = .002) {
    const r = Math.min(radius, ...size.map(n => n * .35));
    const key = `${size.join(',')}:${r}`;
    if (!geometryCache.has(key)) geometryCache.set(key, new RoundedBoxGeometry(...size, 2, r));
    const mesh = new THREE.Mesh(geometryCache.get(key), material); mesh.name = `PairPhoto_${name}`;
    mesh.position.set(...position); parent.add(mesh); return mesh;
  }
  function sphere(parent, name, radius, position, material, scale = [1, 1, 1]) {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 16, 10), material);
    mesh.name = `PairPhoto_${name}`; mesh.position.set(...position); mesh.scale.set(...scale); parent.add(mesh); return mesh;
  }
  function cylinder(parent, name, radius, height, position, material) {
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, height, 16), material);
    mesh.name = `PairPhoto_${name}`; mesh.position.set(...position); parent.add(mesh); return mesh;
  }
  function cable(parent, name, points, material = cableBlack, radius = .0023) {
    const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(...p)));
    const mesh = new THREE.Mesh(new THREE.TubeGeometry(curve, 30, radius, 5, false), material);
    mesh.name = `PairPhoto_${name}`; parent.add(mesh); return mesh;
  }
  function texture(draw, width = 1024, height = 640) {
    const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
    draw(canvas.getContext('2d'), width, height);
    const map = new THREE.CanvasTexture(canvas); map.colorSpace = THREE.SRGBColorSpace; return map;
  }
  function uiTexture(portrait = false) {
    return texture((c, w, h) => {
      c.fillStyle = '#111922'; c.fillRect(0, 0, w, h);
      c.fillStyle = '#252c3d'; c.fillRect(0, 0, w, 28);
      for (let i = 0; i < 3; i++) { c.fillStyle = ['#ad6676','#bfa470','#6eaa9a'][i]; c.beginPath(); c.arc(17+i*16,14,4,0,Math.PI*2); c.fill(); }
      c.fillStyle = '#1b2330'; c.fillRect(0, 30, portrait ? 170 : 198, h-30);
      for (let j=0;j<15;j++) { c.fillStyle = j===3 ? '#477294' : '#5b6b7e'; c.fillRect(20+(j%3)*8, 55+j*29, 90+(j%4)*12, 5); }
      if (portrait) {
        for (let row=0;row<4;row++) for(let col=0;col<2;col++) {
          const x=195+col*225,y=55+row*225;
          c.fillStyle='#263447';c.fillRect(x,y,205,195);
          c.fillStyle=['#467b9d','#6f648d','#568575','#a68157'][(row+col)%4];c.fillRect(x+15,y+20,40,30);
          for(let k=0;k<5;k++){c.fillStyle=k===0?'#b2bac7':'#60758b';c.fillRect(x+15,y+75+k*19,80+(k%3)*28,6);}
          c.fillStyle='#306981';c.fillRect(x+15,y+169,170,10);
        }
      } else {
        for(let i=0;i<30;i++){
          const y=47+i*17,x=220+(i%4)*13;
          c.fillStyle=['#7598ac','#b18cae','#a3b78c','#8a9bac'][i%4];
          c.fillRect(x,y,48+(i*37)%180,5);
          c.fillStyle='#536171';c.fillRect(x+220,y,40+(i*29)%210,5);
        }
      }
      c.fillStyle='#29374b';c.fillRect(0,h-18,w,18);
    }, portrait ? 680 : 1024, portrait ? 1100 : 640);
  }
  const displayMaterial = (name, portrait=false) => {
    const map = uiTexture(portrait);
    const material = new THREE.MeshStandardMaterial({color:'#d9e1e8',map,emissive:'#c3d1df',emissiveMap:map,emissiveIntensity:.14,roughness:.48,metalness:0});
    material.name=`PairPhoto_${name}_Screen_generic_UI`; return material;
  };
  const landscapeUI = displayMaterial('Landscape'), portraitUI = displayMaterial('Portrait',true), laptopUI = displayMaterial('Laptop');
  function monitor(name, x, y, z, width, height, screenMaterial) {
    const group = new THREE.Group(); group.name=`PairPhoto_${name}`; group.position.set(x,y,z);root.add(group);
    box(group,`${name}_shell`,[width,height,.025],[0,0,0],bezel,.004);
    const screen = new THREE.Mesh(new THREE.PlaneGeometry(width-.014,height-.020),screenMaterial);
    screen.name=`PairPhoto_${name}_Screen_generic_UI`; screen.position.set(0,.003,.0131);group.add(screen);
    box(group,`${name}_chin`,[width-.008,.012,.003],[0,-height/2+.008,.014],black,.001);
    box(group,`${name}_back_mount`,[.078,.083,.018],[0,-.012,-.020],black);
    box(root,`${name}_foot`,[.185,.012,.145],[x,TOP+.006,z+.010],black,.006);
    const bottom=y-height/2;
    box(root,`${name}_stem`,[.032,bottom-TOP+.060,.024],[x,TOP+(bottom-TOP+.060)/2,z-.018],silver);
    cylinder(group,`${name}_power_led`,.0015,.001,[width/2-.017,-height/2+.008,.0145],white).rotation.x=Math.PI/2;
    return group;
  }
  const main = monitor('LandscapeMonitor',-.035,1.055,-.358,.640,.370,landscapeUI);
  const portrait = monitor('PortraitMonitor',.465,1.018,-.391,.240,.432,portraitUI);
  // The portrait foot stays on the left table; only the elevated screen overhangs the 2 cm join.
  box(main,'Webcam_clip',[.041,.015,.034],[0,.186,-.003],black);
  box(main,'Webcam_body',[.095,.033,.031],[0,.210,.007],black,.006);
  const lens=cylinder(main,'Webcam_lens',.010,.003,[0,.210,.024],silver); lens.rotation.x=Math.PI/2;
  const lensGlass=cylinder(main,'Webcam_lens_glass',.007,.003,[0,.210,.026],visor);lensGlass.rotation.x=Math.PI/2;

  function keyboard(parent,name,width,depth,position,{pinkKeys=false,grey=false,rotation=0}={}) {
    const group=new THREE.Group();group.name=`PairPhoto_${name}`;group.position.set(...position);group.rotation.y=rotation;parent.add(group);
    box(group,`${name}_case`,[width,.014,depth],[0,.007,0],grey?silver:white,.004);
    const rows=5,cols=16,keyW=width/17.2,keyD=depth/6.0;
    const keyGeo=new RoundedBoxGeometry(keyW*.84,.006,keyD*.80,1,.0012), transforms=[[],[]];
    const dummy=new THREE.Object3D();
    for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){
      if(row===4&&col>=4&&col<=9)continue;
      dummy.position.set((col-7.5)*keyW,.018,(row-2)*keyD);
      dummy.updateMatrix();const accent=pinkKeys&&(col===0||col>=14||row===4);
      transforms[accent?1:0].push(dummy.matrix.clone());
    }
    transforms.forEach((items,index)=>{
      if(!items.length)return;
      const mesh=new THREE.InstancedMesh(keyGeo,index?pink:pale,items.length);mesh.name=`PairPhoto_${name}_${index?'accent':'ivory'}_keys`;
      items.forEach((matrix,i)=>mesh.setMatrixAt(i,matrix));mesh.instanceMatrix.needsUpdate=true;group.add(mesh);
    });
    box(group,`${name}_spacebar`,[keyW*5.5,.006,keyD*.8],[-keyW*.5,.018,keyD*2],pinkKeys?pink:pale,.0015);
    for(let i=0;i<3;i++)sphere(group,`${name}_status${i}`,.0013,[width*.43-i*.009,.017,-depth*.42],i?rubber:green);
    return group;
  }
  box(root,'Pink_mouse_mat',[.500,.003,.280],[.307,TOP+.0015,-.170],matPink,.004);
  keyboard(root,'PinkWhiteMechanicalKeyboard',.425,.142,[-.158,TOP,-.126],{pinkKeys:true});
  const mouse = new THREE.Group();mouse.name='PairPhoto_RoseMouse';mouse.position.set(.332,TOP+.004,-.133);root.add(mouse);
  sphere(mouse,'Rose_mouse_shell',.028,[0,.017,0],mousePink,[.77,.63,1.25]);
  box(mouse,'Rose_mouse_split',[.001,.0015,.029],[0,.033,-.010],rubber,.0003);
  box(mouse,'Rose_mouse_wheel',[.005,.004,.014],[0,.034,-.012],rubber,.001);

  // Thin white laptop base placed vertically, as visible behind the two screens.
  const upright=new THREE.Group();upright.name='PairPhoto_WhiteLaptopUpright';upright.position.set(.270,1.060,-.477);upright.scale.setScalar(.8);root.add(upright);
  box(upright,'WhiteLaptop_thin_base',[.274,.422,.017],[0,0,0],white,.005);
  box(upright,'WhiteLaptop_edge',[.270,.418,.004],[0,0,-.0105],silver,.003);
  const uprightMap=texture((c,w,h)=>{
    c.fillStyle='#e8e8e1';c.fillRect(0,0,w,h);
    c.strokeStyle='#a6aaa7';c.lineWidth=2;
    for(let row=0;row<10;row++)for(let col=0;col<4;col++){c.fillStyle='#d9dcd7';c.fillRect(330+col*66,40+row*58,54,47);c.strokeRect(330+col*66,40+row*58,54,47);}
    c.strokeStyle='#bcc0b9';c.strokeRect(75,510,155,185);
    // A simple bee sticker; no personal text or screen capture is reproduced.
    c.fillStyle='#f0c645';c.beginPath();c.ellipse(175,260,48,65,-.35,0,Math.PI*2);c.fill();
    c.strokeStyle='#423b35';c.lineWidth=10;
    for(let j=0;j<3;j++){c.beginPath();c.moveTo(132,230+j*24);c.lineTo(217,210+j*24);c.stroke();}
    c.fillStyle='#ecefeb';c.strokeStyle='#423b35';c.lineWidth=5;
    for(const x of [135,205]){c.beginPath();c.ellipse(x,186,27,43,x<170?-.7:.7,0,Math.PI*2);c.fill();c.stroke();}
  },620,950);
  const uprightSurface=new THREE.Mesh(new THREE.PlaneGeometry(.263,.409),new THREE.MeshStandardMaterial({map:uprightMap,roughness:.74}));
  uprightSurface.name='PairPhoto_WhiteLaptop_keyboard_face';uprightSurface.position.z=.0091;upright.add(uprightSurface);
  // A small cradle supports the vertical body without inventing a tower case.
  box(root,'WhiteLaptop_support_left',[.016,.1272,.0496],[.194,TOP+.0636,-.475],white);
  box(root,'WhiteLaptop_support_right',[.016,.1272,.0496],[.346,TOP+.0636,-.475],white);

  // Raised black laptop and its separate pale mechanical keyboard.
  const laptop=new THREE.Group();laptop.name='PairPhoto_RaisedBlackLaptop';laptop.position.set(1.155,.883,-.362);laptop.rotation.x=.10;root.add(laptop);
  box(laptop,'Laptop_deck',[.388,.016,.254],[0,0,0],charcoal,.004);
  box(laptop,'Laptop_lower_lip',[.365,.003,.007],[0,-.005,.126],black,.001);
  const keyGeo=new RoundedBoxGeometry(.018,.002,.011,1,.0007),keys=new THREE.InstancedMesh(keyGeo,darkKey,70),dummy=new THREE.Object3D();
  keys.name='PairPhoto_Laptop_keyboard';
  for(let i=0;i<70;i++){dummy.position.set((i%14-6.5)*.023,.009,-.074+Math.floor(i/14)*.016);dummy.updateMatrix();keys.setMatrixAt(i,dummy.matrix);}laptop.add(keys);
  box(laptop,'Laptop_trackpad',[.100,.0008,.059],[0,.0088,.071],silver,.001);
  const lid=new THREE.Group();lid.name='PairPhoto_Laptop_open_lid';lid.position.set(0,.008,-.118);lid.rotation.x=-.24;laptop.add(lid);
  box(lid,'Laptop_lid',[.385,.238,.012],[0,.119,0],charcoal,.003);
  const lcd=new THREE.Mesh(new THREE.PlaneGeometry(.365,.214),laptopUI);lcd.name='PairPhoto_Laptop_Screen_generic_UI';lcd.position.set(0,.121,.0066);lid.add(lcd);
  box(root,'Laptop_stand_base',[.300,.009,.180],[1.155,TOP+.0045,-.363],black,.003);
  for(const x of [1.025,1.285]){
    const strut=box(root,`Laptop_stand_strut_${x}`,[.013,.103,.032],[x,.820,-.408],silver);strut.rotation.x=-.31;
    box(root,`Laptop_stand_rail_${x}`,[.023,.010,.220],[x,.872,-.362],rubber,.002).rotation.x=.10;
  }
  keyboard(root,'GreyMechanicalKeyboard',.428,.144,[1.104,TOP,-.108],{grey:true});
  // A green phone lies on the right palm rest rather than floating beside it.
  const phone=box(laptop,'Green_phone_case',[.067,.009,.112],[.132,.0125,.054],green,.005);
  box(phone,'Green_phone_back',[.059,.0008,.103],[0,.0049,0],green,.004);
  for(let i=0;i<2;i++)cylinder(phone,`Phone_camera${i}`,.006,.001,[-.018,.005,-.033+i*.016],bezel);
  box(root,'Black_mouse_mat',[.145,.002,.175],[1.462,TOP+.001,-.137],rubber,.004);
  sphere(root,'Grey_mouse',.025,[1.466,TOP+.017,-.125],black,[.8,.65,1.3]);

  function astronaut(name,x,z,scale=1,surfaceOffset=0){
    const g=new THREE.Group();g.name=`PairPhoto_${name}`;g.position.set(x,TOP+surfaceOffset,z);g.scale.setScalar(scale);root.add(g);
    box(g,`${name}_feet`,[.045,.012,.026],[0,.006,.003],white,.005);
    box(g,`${name}_torso`,[.033,.035,.024],[0,.031,0],white,.006);
    sphere(g,`${name}_helmet`,.027,[0,.065,0],silver,[1,1,1]);
    sphere(g,`${name}_visor`,.023,[0,.065,.013],visor,[.85,.75,.50]);
    box(g,`${name}_gold_stripe`,[.005,.034,.005],[.004,.077,.023],yellow,.001);
    for(const side of [-1,1])sphere(g,`${name}_glove${side}`,.009,[side*.023,.032,.004],white);
  }
  astronaut('SilverAstronaut',.205,-.278,.85,.003);
  astronaut('SmallAstronaut',.754,-.183,.75);
  function bottle(name,x,z,height=.190){
    const bottleMaterial=mat(`${name}_body`,'#c4d5d0',.28,.02),label=mat(`${name}_label`,'#4c8b65',.7);
    cylinder(root,`${name}_body`,.018,height*.72,[x,TOP+height*.36,z],bottleMaterial);
    cylinder(root,`${name}_shoulder`,.014,height*.16,[x,TOP+height*.80,z],white);
    cylinder(root,`${name}_cap`,.012,height*.10,[x,TOP+height*.93,z],white);
    cylinder(root,`${name}_label`,.0185,height*.23,[x,TOP+height*.46,z],label);
  }
  bottle('Water_left',-.382,-.424,.190);
  bottle('Water_right',1.513,-.419,.205);
  // Sparse cable routes follow the desktop rear/side and fall beside the frame.
  cable(root,'MainDisplay_cable',[[-.035,1.02,-.382],[-.035,.85,-.427],[-.11,.779,-.471],[.22,.777,-.493]]);
  cable(root,'Webcam_cable',[[.012,1.25,-.370],[.11,1.22,-.389],[.15,1.04,-.40],[.205,.778,-.485]],cableBlack,.0015);
  cable(root,'Keyboard_cable',[[-.155,.781,-.201],[-.17,.785,-.247],[-.21,.778,-.28],[-.21,.776,-.47]],cableWhite,.0015);
  cable(root,'Laptop_cable',[[1.354,.885,-.37],[1.45,.812,-.38],[1.594,.777,-.39],[1.628,.72,-.23],[1.628,.54,-.22],[1.628,.20,-.22]],cableBlack,.002);
  cable(root,'Portrait_cable',[[.465,.902,-.415],[.495,.82,-.441],[.51,.775,-.49],[.58,.774,-.49]],cableWhite,.0018);

  const hidden = [];
  for(const identity of ['C_Mw','C_Md']){
    const old=classroom.getNodeByOriginalName(`Equipment / ${identity}`);
    if(!old)throw new Error(`Missing equipment group ${identity}`);
    old.visible=false;hidden.push(old.userData?.name||old.name);
  }
  classroom.root.updateMatrixWorld(true);
  const audit={version:VERSION,reference:'IMG_1993.JPG',desks:['C_Mw','C_Md'],
    originalEquipmentGroupsHidden:hidden,deskDimensionsPreserved:[1,.52,.032],deskPositionsPreserved:true,chairsPreserved:true,
    equipment:['landscape monitor with clip webcam','portrait monitor','thin upright white laptop base with keyboard face','raised black laptop and green phone','white-pink and grey mechanical keyboards','pink mouse mat and mouse','small astronaut figures','sparse routed leads'],
    circledUprightRefinement:{portraitScale:.8,whiteUprightScale:.8,portraitBottom:.802,whiteUprightBottom:.8912,landscapeAndLaptopRestored:true},
    estimated:true,privateScreenContentCopied:false,meshCount:0,instancedKeys:0};
  root.traverse(o=>{if(o.isMesh)audit.meshCount++;if(o.isInstancedMesh)audit.instancedKeys+=o.count;});
  root.userData.audit=audit;classroom.root.userData.pairedWorkstations=audit;return {...audit,root};
}
