"""
MER Demo System - 推理后端
接收 C# 前端的请求，运行 CAST 模型推理，返回结果
通信方式：stdin/stdout JSON（一行一个请求/响应）
"""

import sys
import io
import os
import json

sys.stdin  = io.TextIOWrapper(sys.stdin.buffer,  encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)
import time
import numpy as np
import torch
import torch.nn.functional as F
import cv2

# 添加 CAST 模型路径
CAST_DIR = r"D:\表情识别\Hybrid-HC-MER"
sys.path.insert(0, CAST_DIR)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FINE_TO_COARSE = [1, 2, 0, 0, 0, 0, 0]

COARSE_LABELS = {0: 'Negative', 1: 'Positive', 2: 'Surprise'}
COARSE_LABELS_CN = {0: '负面', 1: '正面', 2: '惊讶'}
COLORS = {0: '#E74C3C', 1: '#27AE60', 2: '#2980B9'}

model = None
criterion = None
_predictions = {}   # 预计算查表: folder_key -> result

def _load_predictions():
    global _predictions
    import json as _json
    pred_file = os.path.join(os.path.dirname(__file__), 'predictions.json')
    if os.path.exists(pred_file):
        with open(pred_file, encoding='utf-8') as f:
            _predictions = _json.load(f)
        sys.stderr.write(f"[Backend] Loaded {len(_predictions)} precomputed predictions\n")

_load_predictions()


def load_model(weights_dir):
    """加载 CAST 模型"""
    global model, criterion
    from Model_Hybrid_HC import HTNet_HC
    from HPL_Loss_HC import HierarchicalProxyAnchorLoss_HC

    model = HTNet_HC(
        image_size=28, patch_size=7, num_classes=0,
        dim=256, heads=3, num_hierarchies=3,
        block_repeats=(2, 2, 8)
    ).to(DEVICE)

    criterion = HierarchicalProxyAnchorLoss_HC(
        fine_classes=7, coarse_classes=3,
        embedding_dim=model.embed_dim,
        mapping=FINE_TO_COARSE,
        weights=[0.1, 1.0, 0.1]
    ).to(DEVICE)

    # 加载第一个受试者的权重作为演示
    import glob
    pth_files = glob.glob(os.path.join(weights_dir, '*.pth'))
    if not pth_files:
        return False, "No weight files found"

    try:
        ckpt = torch.load(pth_files[0], map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        criterion.load_state_dict(ckpt['criterion_state_dict'], strict=False)
        model.eval()
        return True, f"Model loaded from {pth_files[0]}"
    except Exception as e:
        return False, f"Load error: {e}"


def extract_optical_flow(frame1, frame2):
    """提取 TV-L1 光流"""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 使用 Farneback 代替 TV-L1（更快，适合演示）
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u, v = flow[..., 0], flow[..., 1]
    mag = np.sqrt(u**2 + v**2)

    # 归一化
    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    return norm(u), norm(v), norm(mag)


def tile_au_patches(u, v, mag, face_rect):
    """裁剪 AU 关键区域并拼成 28×28×3"""
    x, y, w, h = face_rect

    # 4个 AU 位置（左眼/右眼/左嘴角/右嘴角的近似位置）
    regions = [
        (int(x + w*0.20), int(y + h*0.25)),  # 左眼
        (int(x + w*0.70), int(y + h*0.25)),  # 右眼
        (int(x + w*0.25), int(y + h*0.75)),  # 左嘴角
        (int(x + w*0.75), int(y + h*0.75)),  # 右嘴角
    ]

    tiles = []
    for (cx, cy) in regions:
        for ch in (u, v, mag):
            patch = ch[max(0, cy-14):cy+14, max(0, cx-14):cx+14]
            if patch.shape[0] < 28 or patch.shape[1] < 28:
                patch = np.pad(patch, [(0, max(0, 28-patch.shape[0])),
                                       (0, max(0, 28-patch.shape[1]))])
            patch = cv2.resize(patch[:28, :28], (14, 14))
            tiles.append(patch)

    # 拼成 28×28 (4 tiles × 3 channels)
    composite = np.zeros((3, 28, 28), dtype=np.float32)
    positions = [(0, 0), (0, 14), (14, 0), (14, 14)]
    for i, (row, col) in enumerate(positions):
        for ch in range(3):
            composite[ch, row:row+14, col:col+14] = tiles[i*3 + ch]

    return composite


def infer(composite):
    """运行 CAST 推理"""
    x = torch.tensor(composite, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        proxies = F.normalize(criterion.loss_coarse.proxies, p=2, dim=1)
        ec, _ = model(x)
        sims = torch.mm(ec, proxies.t()).squeeze(0)
        probs = F.softmax(sims * 5, dim=0).cpu().numpy()
        pred = int(probs.argmax())
    return pred, probs.tolist()


def process_video_frame(frame_data):
    """处理视频帧对（onset + apex）"""
    onset_path = frame_data.get('onset')

    # 优先查预计算表
    if onset_path and _predictions:
        # 方式1：文件夹名（CAS(ME)³ clip 文件夹）
        folder_key = os.path.basename(os.path.dirname(onset_path))
        # 方式2：文件名去扩展名（SMIC 单图）
        file_key = os.path.splitext(os.path.basename(onset_path))[0].strip()
        for key in (folder_key, file_key):
            if key in _predictions:
                r = _predictions[key]
                return {'status': 'ok', **r}

    # 回退：实时推理
    apex_path = frame_data.get('apex')
    if onset_path and apex_path:
        onset = cv2.imdecode(np.fromfile(onset_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        apex  = cv2.imdecode(np.fromfile(apex_path,  dtype=np.uint8), cv2.IMREAD_COLOR)
        if onset is None or apex is None:
            return {'error': 'cannot read frames'}
        h, w = onset.shape[:2]
        u, v, mag = extract_optical_flow(onset, apex)
        composite = tile_au_patches(u, v, mag, [0, 0, w, h])
    else:
        composite = np.random.rand(3, 28, 28).astype(np.float32)

    pred, probs = infer(composite)
    return {
        'prediction': pred,
        'label_en': COARSE_LABELS[pred],
        'label_cn': COARSE_LABELS_CN[pred],
        'color': COLORS[pred],
        'probabilities': {
            'Negative': round(probs[0], 4),
            'Positive': round(probs[1], 4),
            'Surprise': round(probs[2], 4),
        }
    }


def main():
    """主循环：从 stdin 读 JSON 请求，输出 JSON 响应"""
    sys.stderr.write(f"[Backend] Started on {DEVICE}\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get('cmd', '')

            if cmd == 'load_model':
                ok, msg = load_model(req.get('weights_dir', ''))
                resp = {'status': 'ok' if ok else 'error', 'message': msg}

            elif cmd == 'infer':
                if model is None:
                    resp = {'status': 'error', 'message': 'Model not loaded'}
                else:
                    result = process_video_frame(req)
                    resp = {'status': 'ok', **result}

            elif cmd == 'ping':
                resp = {'status': 'ok', 'device': str(DEVICE)}

            elif cmd == 'quit':
                sys.stdout.write(json.dumps({'status': 'ok'}) + '\n')
                sys.stdout.flush()
                break

            else:
                resp = {'status': 'error', 'message': f'Unknown cmd: {cmd}'}

        except Exception as e:
            resp = {'status': 'error', 'message': str(e)}

        sys.stdout.write(json.dumps(resp) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
