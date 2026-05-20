"""
Visualize predicted vs GT point clouds in an interactive 3D HTML viewer.
Uses Three.js WebGL — handles millions of points efficiently.

Modes:
  --mode overlay    Both clouds in the SAME scene with toggle buttons (default)
  --mode sidebyside Side-by-side in two panels

Usage:
    python scripts/visualize_pointclouds.py \
        --predicted outputs/dust3r/teddybear_101_11758_21048_10frames/predicted.ply \
        --gt        data/co3d/teddybear/101_11758_21048/pointcloud.ply \
        --output    outputs/dust3r/teddybear_101_11758_21048_10frames/viewer.html \
        --title     "teddybear · 10 frames"
"""

import argparse
import base64
import struct
import os
import numpy as np


def load_ply(path: str, max_points: int = None):
    from plyfile import PlyData
    plydata = PlyData.read(path)
    v = plydata["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    try:
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    except Exception:
        rgb = np.full((len(xyz), 3), 180, dtype=np.uint8)
    if max_points and len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb


def encode_cloud_fast(xyz: np.ndarray, rgb: np.ndarray) -> str:
    dt = np.dtype([('xyz', '<f4', (3,)), ('rgb', 'u1', (3,))])
    arr = np.empty(len(xyz), dtype=dt)
    arr['xyz'] = xyz
    arr['rgb'] = rgb
    return base64.b64encode(arr.tobytes()).decode('ascii')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted", required=True)
    parser.add_argument("--gt",        required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--title",     default="DUSt3R Predicted vs Ground Truth")
    parser.add_argument("--mode",      default="overlay", choices=["overlay", "sidebyside"])
    parser.add_argument("--max_points", type=int, default=None,
                        help="Max points per cloud (default: all points)")
    args = parser.parse_args()

    print(f"Loading predicted: {args.predicted}")
    pred_xyz, pred_rgb = load_ply(args.predicted, args.max_points)
    print(f"  {len(pred_xyz):,} points")

    print(f"Loading GT:        {args.gt}")
    gt_xyz, gt_rgb = load_ply(args.gt, args.max_points)
    print(f"  {len(gt_xyz):,} points")

    print("Encoding binary buffers...")
    pred_b64 = encode_cloud_fast(pred_xyz, pred_rgb)
    gt_b64   = encode_cloud_fast(gt_xyz,   gt_rgb)

    n_pred, n_gt = len(pred_xyz), len(gt_xyz)

    if args.mode == "overlay":
        html = build_overlay_html(pred_b64, gt_b64, n_pred, n_gt, args.title)
    else:
        html = build_sidebyside_html(pred_b64, gt_b64, n_pred, n_gt, args.title)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved to: {args.output}  ({size_mb:.1f} MB)")
    print("Download and open in your browser.")


# ── overlay mode (both clouds in one scene) ───────────────────────────────────
def build_overlay_html(pred_b64, gt_b64, n_pred, n_gt, title):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:#12121f;color:#eee;font-family:'Segoe UI',sans-serif;overflow:hidden;}}
  #header{{
    display:flex;align-items:center;justify-content:space-between;
    padding:8px 18px;background:#1a1a30;border-bottom:1px solid #2a2a50;
    user-select:none;
  }}
  #header h1{{font-size:13px;font-weight:500;color:#ccc;}}
  #controls{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;}}
  .btn{{
    padding:5px 14px;border-radius:20px;border:1px solid #444;
    background:#2a2a40;color:#ddd;cursor:pointer;font-size:12px;
    transition:all .15s;
  }}
  .btn:hover{{background:#3a3a55;}}
  .btn.active-pred{{background:#e05060;border-color:#e05060;color:#fff;}}
  .btn.active-gt  {{background:#50b0e0;border-color:#50b0e0;color:#fff;}}
  .btn.active-both{{background:#60c060;border-color:#60c060;color:#fff;}}
  #stats{{font-size:11px;color:#888;margin-left:6px;}}
  #canvas-container{{width:100vw;height:calc(100vh - 42px);position:relative;}}
  canvas{{display:block;}}
  #info{{
    position:absolute;bottom:10px;left:50%;transform:translateX(-50%);
    background:rgba(0,0,0,.55);padding:4px 14px;border-radius:12px;
    font-size:11px;color:#aaa;pointer-events:none;
  }}
  #metrics{{
    position:absolute;top:10px;right:14px;
    background:rgba(10,10,30,.75);padding:10px 14px;border-radius:10px;
    font-size:11px;color:#bbb;line-height:1.8;min-width:200px;
  }}
  #metrics b{{color:#e0e0ff;}}
</style>
</head>
<body>
<div id="header">
  <h1>{title}</h1>
  <div id="controls">
    <button class="btn active-both" id="btn-both"  onclick="showBoth()">&#10003; Both</button>
    <button class="btn"             id="btn-pred"  onclick="showPred()">Predicted only</button>
    <button class="btn"             id="btn-gt"    onclick="showGT()">GT only</button>
    <span id="stats">Predicted: <b style="color:#e05060">{n_pred:,}</b> pts &nbsp;|&nbsp;
                     GT: <b style="color:#50b0e0">{n_gt:,}</b> pts</span>
  </div>
</div>
<div id="canvas-container">
  <canvas id="c"></canvas>
  <div id="info">Left-drag: rotate &nbsp;|&nbsp; Scroll / right-drag: zoom &nbsp;|&nbsp; Middle-drag: pan</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.134.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.134.0/examples/js/controls/OrbitControls.js"></script>
<script>
const PRED_B64 = "{pred_b64}";
const GT_B64   = "{gt_b64}";

function b64ToBytes(b64){{
  const bin=atob(b64); const b=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) b[i]=bin.charCodeAt(i); return b;
}}
function decodeCloud(b64){{
  const bytes=b64ToBytes(b64); const n=bytes.length/15;
  const pos=new Float32Array(n*3), col=new Float32Array(n*3);
  const dv=new DataView(bytes.buffer);
  for(let i=0;i<n;i++){{
    const o=i*15;
    pos[i*3]=dv.getFloat32(o,true); pos[i*3+1]=dv.getFloat32(o+4,true); pos[i*3+2]=dv.getFloat32(o+8,true);
    col[i*3]=bytes[o+12]/255; col[i*3+1]=bytes[o+13]/255; col[i*3+2]=bytes[o+14]/255;
  }}
  return {{pos,col,n}};
}}
function strideMinMax(a,s,o){{
  let mn=Infinity,mx=-Infinity;
  for(let i=o;i<a.length;i+=s){{if(a[i]<mn)mn=a[i];if(a[i]>mx)mx=a[i];}} return[mn,mx];
}}
function centreAndShift(pos){{
  const[xn,xx]=strideMinMax(pos,3,0),[yn,yx]=strideMinMax(pos,3,1),[zn,zx]=strideMinMax(pos,3,2);
  const cx=(xn+xx)/2,cy=(yn+yx)/2,cz=(zn+zx)/2;
  for(let i=0;i<pos.length;i+=3){{pos[i]-=cx;pos[i+1]-=cy;pos[i+2]-=cz;}}
  return Math.sqrt((xx-xn)**2+(yx-yn)**2+(zx-zn)**2);
}}

let predMesh, gtMesh, renderer, camera, controls, scene;

function buildScene(pred, gt){{
  const container=document.getElementById('canvas-container');
  const canvas=document.getElementById('c');
  const W=container.clientWidth, H=container.clientHeight;

  renderer=new THREE.WebGLRenderer({{canvas,antialias:false}});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  renderer.setSize(W,H); renderer.setClearColor(0x12121f);

  scene=new THREE.Scene();
  camera=new THREE.PerspectiveCamera(60,W/H,0.0001,1000);

  // compute global centroid over both clouds combined
  const allPos=new Float32Array(pred.pos.length+gt.pos.length);
  allPos.set(pred.pos,0); allPos.set(gt.pos,pred.pos.length);
  const diag=centreAndShift(allPos);
  // apply same shift to both
  pred.pos.set(allPos.subarray(0,pred.pos.length));
  gt.pos.set(allPos.subarray(pred.pos.length));

  function makePoints(cloud, tintR, tintG, tintB){{
    // slightly tint colours so clouds are distinguishable when overlaid
    for(let i=0;i<cloud.col.length;i+=3){{
      cloud.col[i]  =Math.min(1, cloud.col[i]  *tintR);
      cloud.col[i+1]=Math.min(1, cloud.col[i+1]*tintG);
      cloud.col[i+2]=Math.min(1, cloud.col[i+2]*tintB);
    }}
    const geo=new THREE.BufferGeometry();
    geo.setAttribute('position',new THREE.BufferAttribute(cloud.pos,3));
    geo.setAttribute('color',   new THREE.BufferAttribute(cloud.col,3));
    const mat=new THREE.PointsMaterial({{size:0.004,vertexColors:true,sizeAttenuation:true}});
    return new THREE.Points(geo,mat);
  }}

  predMesh=makePoints(pred, 1.15, 0.85, 0.85); // warm tint → red-ish
  gtMesh  =makePoints(gt,   0.85, 0.95, 1.20); // cool tint → blue-ish
  scene.add(predMesh); scene.add(gtMesh);

  camera.position.set(0,0,diag*0.8);
  controls=new THREE.OrbitControls(camera,canvas);
  controls.enableDamping=true; controls.dampingFactor=0.08;

  window.addEventListener('resize',()=>{{
    const W2=container.clientWidth,H2=container.clientHeight;
    camera.aspect=W2/H2; camera.updateProjectionMatrix();
    renderer.setSize(W2,H2);
  }});
  (function animate(){{requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);}})();
}}

function showBoth(){{predMesh.visible=true; gtMesh.visible=true; setActive('btn-both','active-both');}}
function showPred(){{predMesh.visible=true; gtMesh.visible=false; setActive('btn-pred','active-pred');}}
function showGT()  {{predMesh.visible=false;gtMesh.visible=true;  setActive('btn-gt','active-gt');}}
function setActive(id,cls){{
  ['btn-both','btn-pred','btn-gt'].forEach(b=>{{
    const el=document.getElementById(b);
    el.className='btn'+(b===id?' '+cls:'');
  }});
}}

setTimeout(()=>{{
  console.log('Decoding clouds...');
  const pred=decodeCloud(PRED_B64), gt=decodeCloud(GT_B64);
  console.log('Building scene...');
  buildScene(pred,gt);
  console.log('Done.');
}},50);
</script>
</body>
</html>
"""


# ── side-by-side mode ─────────────────────────────────────────────────────────
def build_sidebyside_html(pred_b64, gt_b64, n_pred, n_gt, title):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:#12121f;color:#eee;font-family:'Segoe UI',sans-serif;overflow:hidden;}}
  #header{{text-align:center;padding:7px;background:#1a1a30;
           border-bottom:1px solid #2a2a50;font-size:13px;user-select:none;}}
  #container{{display:flex;width:100vw;height:calc(100vh - 34px);}}
  .panel{{flex:1;position:relative;border-right:1px solid #2a2a50;}}
  .panel:last-child{{border-right:none;}}
  .label{{position:absolute;top:8px;left:50%;transform:translateX(-50%);
          background:rgba(0,0,0,.55);padding:4px 14px;border-radius:20px;
          font-size:12px;pointer-events:none;z-index:5;white-space:nowrap;}}
  canvas{{display:block;}}
</style>
</head>
<body>
<div id="header">
  {title} &nbsp;—&nbsp;
  Predicted: <b style="color:#e05060">{n_pred:,} pts</b> &nbsp;|&nbsp;
  GT: <b style="color:#50b0e0">{n_gt:,} pts</b>
</div>
<div id="container">
  <div class="panel" id="p-pred">
    <div class="label" style="color:#f08090">DUSt3R Predicted &nbsp;·&nbsp; {n_pred:,} pts</div>
    <canvas id="c-pred"></canvas>
  </div>
  <div class="panel" id="p-gt">
    <div class="label" style="color:#70c8f0">Ground Truth &nbsp;·&nbsp; {n_gt:,} pts</div>
    <canvas id="c-gt"></canvas>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.134.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.134.0/examples/js/controls/OrbitControls.js"></script>
<script>
const PRED_B64="{pred_b64}", GT_B64="{gt_b64}";
function b64ToBytes(b64){{const bin=atob(b64);const b=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)b[i]=bin.charCodeAt(i);return b;}}
function decodeCloud(b64){{const bytes=b64ToBytes(b64);const n=bytes.length/15;const pos=new Float32Array(n*3),col=new Float32Array(n*3);const dv=new DataView(bytes.buffer);for(let i=0;i<n;i++){{const o=i*15;pos[i*3]=dv.getFloat32(o,true);pos[i*3+1]=dv.getFloat32(o+4,true);pos[i*3+2]=dv.getFloat32(o+8,true);col[i*3]=bytes[o+12]/255;col[i*3+1]=bytes[o+13]/255;col[i*3+2]=bytes[o+14]/255;}}return{{pos,col,n}};}}
function smm(a,s,o){{let mn=Infinity,mx=-Infinity;for(let i=o;i<a.length;i+=s){{if(a[i]<mn)mn=a[i];if(a[i]>mx)mx=a[i];}}return[mn,mx];}}
function buildViewer(canvasId,panelId,cloud){{
  const panel=document.getElementById(panelId),canvas=document.getElementById(canvasId);
  const W=panel.clientWidth,H=panel.clientHeight;
  canvas.width=W;canvas.height=H;
  const renderer=new THREE.WebGLRenderer({{canvas,antialias:false}});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(W,H);renderer.setClearColor(0x12121f);
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(60,W/H,0.0001,1000);
  const[xn,xx]=smm(cloud.pos,3,0),[yn,yx]=smm(cloud.pos,3,1),[zn,zx]=smm(cloud.pos,3,2);
  const cx=(xn+xx)/2,cy=(yn+yx)/2,cz=(zn+zx)/2;
  for(let i=0;i<cloud.pos.length;i+=3){{cloud.pos[i]-=cx;cloud.pos[i+1]-=cy;cloud.pos[i+2]-=cz;}}
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.BufferAttribute(cloud.pos,3));
  geo.setAttribute('color',new THREE.BufferAttribute(cloud.col,3));
  scene.add(new THREE.Points(geo,new THREE.PointsMaterial({{size:0.004,vertexColors:true,sizeAttenuation:true}})));
  const diag=Math.sqrt((xx-xn)**2+(yx-yn)**2+(zx-zn)**2);
  camera.position.set(0,0,diag*0.8);
  const ctrl=new THREE.OrbitControls(camera,canvas);ctrl.enableDamping=true;ctrl.dampingFactor=0.08;
  window.addEventListener('resize',()=>{{const W2=panel.clientWidth,H2=panel.clientHeight;camera.aspect=W2/H2;camera.updateProjectionMatrix();renderer.setSize(W2,H2);}});
  (function animate(){{requestAnimationFrame(animate);ctrl.update();renderer.render(scene,camera);}})();
}}
setTimeout(()=>{{const pred=decodeCloud(PRED_B64),gt=decodeCloud(GT_B64);buildViewer('c-pred','p-pred',pred);buildViewer('c-gt','p-gt',gt);}},50);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
