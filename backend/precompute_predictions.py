"""
预计算所有 CAS(ME)³ clip 的推理结果，存成 JSON 查表文件。
使用和训练完全一致的预处理（MTCNN + Farneback 光流）。
输出: predictions.json  { "spNO.1_j_28": { prediction, label_en, ... }, ... }
"""
import sys, os, json, glob, re
import cv2, numpy as np
import torch, torch.nn.functional as F
import pandas as pd
from facenet_pytorch import MTCNN

CAST_DIR    = r"D:\表情识别\Hybrid-HC-MER"
WEIGHTS_DIR = r"F:\research\Hybrid-HC-MER\weights_Hybrid_HC_Seed_1"
LABEL_FILE  = r"D:\表情识别\Hybrid-HC-MER\datasets\CAS(ME)^3\cas(me)3_part_A_ME_label_JpgIndex_v1.xls"
FRAME_ROOT  = r"D:\表情识别\Hybrid-HC-MER\datasets\CAS(ME)^3\Part_A_ME_clip\frame"
OUT_JSON    = r"D:\表情识别\MER_Demo_System\backend\predictions.json"

sys.path.insert(0, CAST_DIR)
from Model_Hybrid_HC import HTNet_HC
from HPL_Loss_HC import HierarchicalProxyAnchorLoss_HC

DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FINE_TO_COARSE = [1, 2, 0, 0, 0, 0, 0]
COARSE_LABELS    = {0: 'Negative', 1: 'Positive', 2: 'Surprise'}
COARSE_LABELS_CN = {0: '负面',     1: '正面',     2: '惊讶'}
COLORS           = {0: '#E74C3C',  1: '#27AE60',  2: '#2980B9'}

LABEL_MAP = {
    'happiness': 0, 'happy': 0,
    'surprise': 1,
    'disgust': 2, 'fear': 3, 'sadness': 4, 'sad': 4,
    'contempt': 5, 'anger': 5,
    'others': 6, 'Others': 6,
}


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def find_frame(folder, idx):
    s = str(int(idx))
    for name in [s, s.zfill(4), '0'+s, '00'+s]:
        p = os.path.join(folder, name + '.jpg')
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(folder, f'*{s}.jpg'))
    return hits[0] if hits else None


def extract_feature(onset_path, apex_path, mtcnn, target_size=14):
    img_onset = imread_unicode(onset_path)
    img_apex  = imread_unicode(apex_path)
    if img_onset is None or img_apex is None:
        return None

    img_rgb = cv2.cvtColor(img_apex, cv2.COLOR_BGR2RGB)
    _, _, landmarks = mtcnn.detect(img_rgb, landmarks=True)
    if landmarks is None:
        return None

    pts = landmarks[0]
    left_eye, right_eye, _, left_lip, right_lip = pts

    gray_onset = cv2.cvtColor(img_onset, cv2.COLOR_BGR2GRAY)
    gray_apex  = cv2.cvtColor(img_apex,  cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray_onset.astype(np.float32) / 255.,
        gray_apex.astype(np.float32)  / 255.,
        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u, v = flow[..., 0], flow[..., 1]
    mag  = np.sqrt(u**2 + v**2)

    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)
    u, v, mag = norm(u), norm(v), norm(mag)

    centers = [
        (int(left_eye[0]),  int(left_eye[1])),
        (int(right_eye[0]), int(right_eye[1])),
        (int(left_lip[0]),  int(left_lip[1])),
        (int(right_lip[0]), int(right_lip[1])),
    ]
    composite = np.zeros((3, 28, 28), dtype=np.float32)
    positions = [(0, 0), (0, 14), (14, 0), (14, 14)]
    for i, (cx, cy) in enumerate(centers):
        row, col = positions[i]
        for chi, ch in enumerate((u, v, mag)):
            patch = ch[max(0,cy-14):cy+14, max(0,cx-14):cx+14]
            ph, pw = patch.shape
            if ph < 28 or pw < 28:
                patch = np.pad(patch, [(0, max(0,28-ph)), (0, max(0,28-pw))])
            patch = cv2.resize(patch[:28, :28], (target_size, target_size))
            composite[chi, row:row+target_size, col:col+target_size] = patch
    return composite


def build_model():
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
    return model, criterion


def load_weights(model, criterion, subject):
    pth = os.path.join(WEIGHTS_DIR, f"{subject}.pth")
    if not os.path.exists(pth):
        files = glob.glob(os.path.join(WEIGHTS_DIR, "*.pth"))
        pth = files[0] if files else None
    if pth is None:
        return False
    ckpt = torch.load(pth, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    criterion.load_state_dict(ckpt['criterion_state_dict'], strict=False)
    model.eval()
    return True


def infer(composite, model, criterion):
    x = torch.tensor(composite).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        proxies = F.normalize(criterion.loss_coarse.proxies, p=2, dim=1)
        ec, _   = model(x)
        sims    = torch.mm(ec, proxies.t()).squeeze(0)
        probs   = F.softmax(sims * 5, dim=0).cpu().numpy()
        pred    = int(probs.argmax())
    return pred, probs.tolist()


def main():
    mtcnn = MTCNN(device=DEVICE)
    df    = pd.read_excel(LABEL_FILE)
    model, criterion = build_model()

    # 建文件夹名 → 文件夹路径 的索引
    folders = {f: os.path.join(FRAME_ROOT, f) for f in os.listdir(FRAME_ROOT)
               if os.path.isdir(os.path.join(FRAME_ROOT, f))}

    results     = {}
    current_sub = None
    ok = err = skip = 0

    for _, row in df.iterrows():
        subj  = str(row.get('Subject', '')).strip()
        fname = str(row.get('Filename', '')).strip()
        onset = row.get('Onset', np.nan)
        apex  = row.get('Apex',  np.nan)
        if pd.isna(onset) or pd.isna(apex):
            skip += 1; continue
        onset, apex = int(onset), int(apex)
        if onset > apex:
            onset, apex = apex, onset

        # 找对应文件夹
        clean = re.sub(r'[\s_]', '', f'{subj}{fname}{onset}').lower()
        folder_path = None
        for fn, fp in folders.items():
            if re.sub(r'[\s_]', '', fn).lower().startswith(clean[:len(clean)]):
                folder_path = fp; break
        if folder_path is None:
            skip += 1; continue

        folder_key = os.path.basename(folder_path)

        onset_p = find_frame(folder_path, onset)
        apex_p  = find_frame(folder_path, apex)
        if not onset_p or not apex_p:
            skip += 1; continue

        # 换受试者时重新加载 LOSO 权重
        if subj != current_sub:
            load_weights(model, criterion, subj)
            current_sub = subj
            print(f"  weights -> {subj}")

        feat = extract_feature(onset_p, apex_p, mtcnn)
        if feat is None:
            err += 1; continue

        pred, probs = infer(feat, model, criterion)
        results[folder_key] = {
            'prediction':    pred,
            'label_en':      COARSE_LABELS[pred],
            'label_cn':      COARSE_LABELS_CN[pred],
            'color':         COLORS[pred],
            'probabilities': {
                'Negative': round(probs[0], 4),
                'Positive': round(probs[1], 4),
                'Surprise': round(probs[2], 4),
            }
        }
        ok += 1

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {ok} ok, {err} face-detect failed, {skip} skipped")
    print(f"Saved to {OUT_JSON}")
    from collections import Counter
    print("Distribution:", dict(Counter(v['label_en'] for v in results.values())))


if __name__ == '__main__':
    main()
