"""
预计算所有 SMIC clip 的推理结果，追加到 predictions.json。
直接用 three_norm_u_v_os 里的 28×28 特征图做推理，无需重新提取。
"""
import sys, os, json, glob
import cv2, numpy as np
import torch, torch.nn.functional as F

CAST_DIR    = r"D:\表情识别\Hybrid-HC-MER"
WEIGHTS_DIR = r"F:\research\Hybrid-HC-MER\weights_Hybrid_HC_combined_v2_Seed_2026"
FEAT_ROOT   = r"D:\表情识别\Hybrid-HC-MER\datasets\three_norm_u_v_os"
OUT_JSON    = r"D:\表情识别\MER_Demo_System\backend\predictions.json"

sys.path.insert(0, CAST_DIR)
from Model_Hybrid_HC import HTNet_HC
from HPL_Loss_HC import HierarchicalProxyAnchorLoss_HC

DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FINE_TO_COARSE = [1, 2, 0, 0, 0, 0, 0]
COARSE_LABELS    = {0: 'Negative', 1: 'Positive', 2: 'Surprise'}
COARSE_LABELS_CN = {0: '负面',     1: '正面',     2: '惊讶'}
COLORS           = {0: '#E74C3C',  1: '#27AE60',  2: '#2980B9'}


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


def infer_image(img_path, model, criterion):
    """直接从 28×28 特征图推理"""
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    img = cv2.resize(img, (28, 28)).astype(np.float32) / 255.0
    x = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        proxies = F.normalize(criterion.loss_coarse.proxies, p=2, dim=1)
        ec, _   = model(x)
        sims    = torch.mm(ec, proxies.t()).squeeze(0)
        probs   = F.softmax(sims * 5, dim=0).cpu().numpy()
        pred    = int(probs.argmax())
    return pred, probs.tolist()


def main():
    # 读取已有 predictions
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = {}

    model, criterion = build_model()
    current_sub = None
    ok = 0

    # 遍历所有 SMIC 受试者
    for subj_dir in sorted(os.listdir(FEAT_ROOT)):
        if not subj_dir.startswith('s'):
            continue
        subj_path = os.path.join(FEAT_ROOT, subj_dir)
        if not os.path.isdir(subj_path):
            continue

        # 加载对应受试者的 LOSO 权重
        if subj_dir != current_sub:
            load_weights(model, criterion, subj_dir)
            current_sub = subj_dir
            print(f"  weights -> {subj_dir}")

        # 遍历 test/train 下的所有类别
        for split in ('u_test', 'u_train'):
            split_path = os.path.join(subj_path, split)
            if not os.path.isdir(split_path):
                continue
            for cls_dir in os.listdir(split_path):
                cls_path = os.path.join(split_path, cls_dir)
                if not os.path.isdir(cls_path):
                    continue
                for fname in os.listdir(cls_path):
                    if not fname.lower().endswith('.png'):
                        continue
                    img_path = os.path.join(cls_path, fname)
                    # key = 文件名去掉扩展名和多余空格
                    key = fname.replace(' .png', '').replace('.png', '').strip()

                    pred, probs = infer_image(img_path, model, criterion)
                    if pred is None:
                        continue

                    results[key] = {
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

    smic_only = {k: v for k, v in results.items() if k.startswith('s')}
    print(f"\nDone: {ok} SMIC predictions added")
    from collections import Counter
    print("SMIC distribution:", dict(Counter(v['label_en'] for v in smic_only.values())))
    print(f"Total in JSON: {len(results)}")


if __name__ == '__main__':
    main()
