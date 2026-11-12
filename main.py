#dev/creator=tubakhxn---
import os, sys, glob, subprocess, time
import numpy as np, cv2
ROOT = os.path.dirname(os.path.abspath(__file__))
FPS_PROC, FPS_OUT = (10, 30)
W, H = (960, 532)
IW, IH = (518, 294)
Z_NEAR, Z_FAR = (4.0, 60.0)
HFOV_DEG = 90.0
HOOD_ROW = int(0.7 * H)
STRIDE = 1
WATERMARK = 'tubakhxn'

def run_depth():
    import onnxruntime as ort
    fr = f'{ROOT}/cache/frames'
    os.makedirs(fr, exist_ok=True)
    if not glob.glob(fr + '/*.jpg'):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', f'{ROOT}/input/original_dashcam.mp4', '-vf', f'fps={FPS_PROC},scale={W}:{H}', '-q:v', '2', fr + '/%04d.jpg'], check=True)
    files = sorted(glob.glob(fr + '/*.jpg'))
    os.makedirs(f'{ROOT}/cache/depth', exist_ok=True)
    todo = [i for i in range(len(files)) if not os.path.exists(f'{ROOT}/cache/depth/{i:04d}.npy')]
    if todo:
        sess = ort.InferenceSession(f'{ROOT}/models/depth_anything_v2_small.onnx', providers=['CPUExecutionProvider'])
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        for i in todo:
            rgb = cv2.cvtColor(cv2.imread(files[i]), cv2.COLOR_BGR2RGB)
            x = cv2.resize(rgb, (IW, IH), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
            x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
            d = sess.run(None, {'pixel_values': x})[0][0]
            np.save(f'{ROOT}/cache/depth/{i:04d}.npy', cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float16))
    return files

def load_normalised(n):
    out, lo_e, hi_e, prev = ([], None, None, None)
    for i in range(n):
        d = np.load(f'{ROOT}/cache/depth/{i:04d}.npy').astype(np.float32)
        reg = d[:HOOD_ROW:4, ::4]
        lo, hi = (np.percentile(reg, 2), np.percentile(reg, 99.5))
        lo_e = lo if lo_e is None else 0.9 * lo_e + 0.1 * lo
        hi = hi + 0.15 * (hi - lo)
        hi_e = hi if hi_e is None else 0.9 * hi_e + 0.1 * hi
        dn = np.clip((d - lo_e) / max(hi_e - lo_e, 1e-06), 0, 1)
        dn = dn if prev is None else 0.65 * dn + 0.35 * prev
        prev = dn
        out.append(dn)
    return out

def to_z(dn):
    return 1.0 / (dn * (1.0 / Z_NEAR - 1.0 / Z_FAR) + 1.0 / Z_FAR)

def backproject(rgb, dn):
    s = STRIDE
    dn_s = cv2.medianBlur((dn * 255).astype(np.uint8), 5).astype(np.float32) / 255.0
    dn_s = cv2.resize(dn_s, (W // s, H // s), interpolation=cv2.INTER_AREA)
    col = cv2.resize(rgb, (W // s, H // s), interpolation=cv2.INTER_AREA)
    z = to_z(dn_s)
    h, w = z.shape
    fx = W / 2 / np.tan(np.radians(HFOV_DEG / 2))
    fy = fx
    u, v = np.meshgrid(np.arange(w) * s + s / 2, np.arange(h) * s + s / 2)
    gz = np.hypot(*np.gradient(np.log(z)))
    valid = np.isfinite(z) & (v < HOOD_ROW) & (dn_s > 0.07) & (gz < 0.1)
    x = (u - W / 2) * z / fx
    y = (v - H / 2) * z / fx
    P = np.stack([x, -y, z], -1)[valid]
    C = col[valid]
    return (P.astype(np.float32), C)

def look_at(eye, target):
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, f])

def project(Pw, eye, Rm, fx, cx, cy):
    pc = (Pw - eye) @ Rm.T
    zc = pc[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        px = fx * pc[:, 0] / zc + cx
        py = cy - fx * pc[:, 1] / zc
    return (px, py, zc)

def render_cloud(P, C, t, size=(1048, 915)):
    w, h = size
    img = np.full((h, w, 3), (16, 16, 18), np.uint8)
    az = np.radians(22) * np.sin(2 * np.pi * t * 0.5)
    el = np.radians(21)
    tgt = np.array([0, -1.0, 24.0])
    R = 42.0
    eye = tgt + R * np.array([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)])
    Rm = look_at(eye, tgt)
    fx = w / 2 / np.tan(np.radians(48 / 2))
    cx, cy = (w / 2, h / 2 + 40)
    Pw = P.copy()
    Pw[:, 1] -= 0.0
    px, py, zc = project(Pw, eye, Rm, fx, cx, cy)
    ok = zc > 1.0
    px, py, zc, Cc = (px[ok], py[ok], zc[ok], C[ok])

    def seg(a, b, colr, th=1):
        pa = project(np.array([a], np.float32), eye, Rm, fx, cx, cy)
        pb = project(np.array([b], np.float32), eye, Rm, fx, cx, cy)
        if pa[2][0] > 0.5 and pb[2][0] > 0.5:
            cv2.line(img, (int(pa[0][0]), int(pa[1][0])), (int(pb[0][0]), int(pb[1][0])), colr, th, cv2.LINE_AA)
    gy = -1.5
    for gx in range(-40, 41, 10):
        seg((gx, gy, 2), (gx, gy, 62), (46, 46, 52))
    for gz in range(2, 63, 10):
        seg((-40, gy, gz), (40, gy, gz), (46, 46, 52))
    k = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    order = np.argsort(-zc)
    px, py, Cc = (px[order], py[order], Cc[order])
    X = (np.round(px)[None, :] + k[:, 0:1]).astype(np.int32)
    Y = (np.round(py)[None, :] + k[:, 1:2]).astype(np.int32)
    inb = (X >= 0) & (X < w) & (Y >= 0) & (Y < h)
    Xf, Yf = (X.T.reshape(-1), Y.T.reshape(-1))
    inf = inb.T.reshape(-1)
    Cf = np.repeat(Cc, 9, axis=0)
    img[Yf[inf], Xf[inf]] = Cf[inf]
    O = (0, gy, 0)
    seg(O, (14, gy, 0), (60, 60, 255), 2)
    seg(O, (0, gy + 14, 0), (80, 230, 80), 2)
    seg(O, (0, gy, 14), (255, 170, 60), 2)
    for name, p, colr in (('X', (15.5, gy, 0), (60, 60, 255)), ('Y', (0, gy + 15.5, 0), (80, 230, 80)), ('Z', (0, gy, 15.5), (255, 170, 60))):
        q = project(np.array([p], np.float32), eye, Rm, fx, cx, cy)
        if q[2][0] > 0.5:
            cv2.putText(img, name, (int(q[0][0]) - 6, int(q[1][0]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colr, 2, cv2.LINE_AA)
    for gz in (10, 20, 30, 40, 50):
        q = project(np.array([(-1.2, gy, gz)], np.float32), eye, Rm, fx, cx, cy)
        if q[2][0] > 0.5:
            cv2.putText(img, f'Z {gz}', (int(q[0][0]) - 40, int(q[1][0])), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 140), 1, cv2.LINE_AA)
    return img
FONT = cv2.FONT_HERSHEY_SIMPLEX

def label(img, text, x, y):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.48, 1)
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x + tw + 16, y + th + 14), (10, 10, 12), -1)
    cv2.addWeighted(ov, 0.7, img, 0.3, 0, img)
    cv2.putText(img, text, (x + 8, y + th + 6), FONT, 0.48, (235, 235, 235), 1, cv2.LINE_AA)

def watermark(img, margin=24):
    ov = img.copy()
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(WATERMARK, FONT, 0.6, 1)
    cv2.putText(ov, WATERMARK, (w - tw - margin, h - margin), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)

def colourbar(img, x, y, hgt=130, wd=10):
    g = np.linspace(255, 0, hgt).astype(np.uint8)[:, None].repeat(wd, 1)
    img[y:y + hgt, x:x + wd] = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)
    cv2.putText(img, 'NEAR', (x - 40, y + 10), FONT, 0.36, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(img, 'FAR', (x - 30, y + hgt), FONT, 0.36, (230, 230, 230), 1, cv2.LINE_AA)

def depth_panel(dn):
    return cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

def compose(rgb, dn, cloud, npts):
    c = np.full((1080, 1920, 3), (14, 14, 16), np.uint8)
    cv2.putText(c, 'MONOCULAR 3D SCENE RECONSTRUCTION', (24, 48), FONT, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(c, 'Depth Anything V2 (Small)  |  RGB-D projection  |  relative depth, assumed intrinsics (HFOV 90 deg)', (24, 72), FONT, 0.45, (150, 150, 160), 1, cv2.LINE_AA)
    pw, ph = (800, 443)
    top = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA)
    bot = cv2.resize(depth_panel(dn), (pw, ph), interpolation=cv2.INTER_AREA)
    c[84:84 + ph, 24:24 + pw] = top
    c[84 + ph + 29:84 + 2 * ph + 29, 24:24 + pw] = bot
    c[84:84 + 915, 848:848 + 1048] = cloud
    for x, y, w_, h_ in ((24, 84, pw, ph), (24, 84 + ph + 29, pw, ph), (848, 84, 1048, 915)):
        cv2.rectangle(c, (x - 1, y - 1), (x + w_, y + h_), (95, 95, 105), 1)
    label(c, 'ORIGINAL RGB FRAME', 34, 94)
    label(c, 'AI PREDICTED DEPTH', 34, 84 + ph + 29 + 10)
    label(c, f'RECONSTRUCTED 3D POINT CLOUD  |  {npts:,} pts  |  relative scale (not metric)', 858, 94)
    colourbar(c, 24 + pw - 24, 84 + ph + 29 + 14)
    cv2.putText(c, 'Relative monocular depth - not metric depth, not LiDAR. Research & educational purposes only.', (24, 1050), FONT, 0.42, (130, 130, 140), 1, cv2.LINE_AA)
    watermark(c)
    return c

def card(lines, alpha):
    c = np.full((1080, 1920, 3), (12, 12, 14), np.uint8)
    for txt, scale, y, colr in lines:
        (tw, _), _ = cv2.getTextSize(txt, FONT, scale, 2 if scale > 1 else 1)
        cv2.putText(c, txt, ((1920 - tw) // 2, y), FONT, scale, colr, 2 if scale > 1 else 1, cv2.LINE_AA)
    watermark(c)
    return (c * alpha).astype(np.uint8)

def enc(path, w, h, fps=FPS_OUT):
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path], stdin=subprocess.PIPE)

def frame_data(i, files, dns):
    rgb = cv2.imread(files[i])
    P, C = backproject(rgb, dns[i])
    cloud = render_cloud(P, C, i / FPS_PROC)
    return (rgb, dns[i], cloud, len(P))

def main():
    files = run_depth()
    n = len(files)
    dns = load_normalised(n)
    if '--preview' in sys.argv:
        i = int(sys.argv[sys.argv.index('--preview') + 1])
        cv2.imwrite(f'{ROOT}/cache/preview_{i}.png', compose(*frame_data(i, files, dns)))
        return
    os.makedirs(f'{ROOT}/output', exist_ok=True)
    main_p = enc(f'{ROOT}/output/3d_scene_reconstruction.mp4', 1920, 1080)
    dep_p = enc(f'{ROOT}/output/depth_visualization.mp4', 960, 532)
    pc_p = enc(f'{ROOT}/output/point_cloud_visualization.mp4', 1048, 914)
    for k in range(75):
        a = min(1, k / 15, (74 - k) / 15)
        main_p.stdin.write(card([('MONOCULAR 3D SCENE RECONSTRUCTION', 1.4, 500, (245, 245, 245)), ('AI DEPTH ESTIMATION  |  RGB-D PROJECTION  |  3D POINT CLOUD', 0.7, 560, (170, 170, 180))], a).tobytes())
    prev = None
    t0 = time.time()

    def emit(cur_c, cur_d, cur_p, nxt):
        for k in range(FPS_OUT // FPS_PROC):
            wgt = k / (FPS_OUT // FPS_PROC) if nxt is not None else 0
            blend = lambda a, b: a if nxt is None else cv2.addWeighted(a, 1 - wgt, b, wgt, 0)
            main_p.stdin.write(blend(cur_c, nxt[0] if nxt else None).tobytes())
            dep_p.stdin.write(blend(cur_d, nxt[1] if nxt else None).tobytes())
            pc_p.stdin.write(blend(cur_p, nxt[2] if nxt else None).tobytes())
            if nxt is None:
                break

    def pack(i):
        rgb, dn, cloud, npts = frame_data(i, files, dns)
        d = cv2.resize(depth_panel(dn), (960, 532))
        label(d, 'AI PREDICTED DEPTH', 10, 10)
        colourbar(d, 920, 20)
        watermark(d, 12)
        p = cloud[:914].copy()
        label(p, f'3D POINT CLOUD  |  {npts:,} pts  |  relative scale', 10, 10)
        watermark(p, 12)
        return (compose(rgb, dn, cloud, npts), d, p)
    prev = pack(0)
    for i in range(1, n):
        cur = pack(i)
        emit(*prev, cur)
        prev = cur
        if i % 20 == 0:
            print(f'render {i}/{n}  {time.time() - t0:.0f}s', flush=True)
    emit(*prev, None)
    for k in range(75):
        a = min(1, k / 15, (74 - k) / 15)
        main_p.stdin.write(card([('Developed by tubakhxn', 1.3, 520, (245, 245, 245)), ('Research & Educational Purposes Only.', 0.7, 580, (170, 170, 180))], a).tobytes())
    for p in (main_p, dep_p, pc_p):
        p.stdin.close()
        p.wait()
    print('DONE', flush=True)
if __name__ == '__main__':
    main()
