import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { CLASSROOM_ASSET_SHA256 } from './scene-frame.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const arguments_ = process.argv.slice(2);
function option(name) {
  const index = arguments_.indexOf(name);
  if (index === -1) return null;
  const value = arguments_[index + 1];
  if (!value || value.startsWith('--')) throw new Error(`Missing value for ${name}`);
  return value;
}

const dependencyArgument = option('--dependencies');
const dependencyRoot = dependencyArgument
  ? path.resolve(dependencyArgument, path.basename(path.resolve(dependencyArgument)) === 'node_modules' ? '' : 'node_modules')
  : path.join(here, 'node_modules');
const dependencyRequire = createRequire(path.join(path.dirname(dependencyRoot), 'package.json'));
let build;
try {
  ({ build } = await import(pathToFileURL(dependencyRequire.resolve('esbuild')).href));
} catch {
  throw new Error('Build dependencies missing. Use the locked npm packages or pass --dependencies <existing node_modules>.');
}

// The published model is also the maintained model. Verify it before writing the page.
const published = path.resolve(here, '../web/classroom');
const output = path.resolve(here, option('--out') ?? published);
const relativeAsset = path.join('assets', 'classroom-v3.glb');
const sourceAsset = path.join(published, relativeAsset);
const asset = await fs.readFile(sourceAsset);
const sha256 = createHash('sha256').update(asset).digest('hex');
if (sha256 !== CLASSROOM_ASSET_SHA256 || asset.readUInt32LE(0) !== 0x46546c67 ||
    asset.readUInt32LE(4) !== 2 || asset.readUInt32LE(8) !== asset.length) {
  throw new Error('The active classroom-v3.glb differs from the original model.');
}

const bundle = await build({
  entryPoints: [path.join(here, 'viewer.mjs')], bundle: true, minify: true,
  write: false, format: 'iife', target: 'es2022', nodePaths: [dependencyRoot], legalComments: 'inline',
});
const template = await fs.readFile(path.join(here, 'template.html'), 'utf8');
if (!template.includes('__BUNDLE__')) throw new Error('Viewer template is missing its bundle slot.');
const html = template.replace('__BUNDLE__', () => bundle.outputFiles[0].text.replace(/<\/script/gi, '<\\/script'));
await fs.mkdir(output, { recursive: true });
if (output !== published) {
  await fs.mkdir(path.join(output, 'assets'), { recursive: true });
  await fs.copyFile(sourceAsset, path.join(output, relativeAsset));
}
const threeRoot = path.resolve(path.dirname(dependencyRequire.resolve('three')), '..');
await fs.copyFile(path.join(threeRoot, 'LICENSE'), path.join(output, 'THREE_LICENSE.txt'));
await fs.writeFile(path.join(output, 'index.html'), html);
console.log(JSON.stringify({ output, htmlBytes: Buffer.byteLength(html), assetBytes: asset.length, sha256 }, null, 2));
