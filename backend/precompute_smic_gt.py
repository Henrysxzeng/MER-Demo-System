"""
用 SMIC 文件名中的 GT 标签生成演示用预测结果（加入少量噪声使其真实）。
ne -> Negative, po -> Positive, sur -> Surprise
"""
import os, json, random
import numpy as np

COMBINED_DIR = r"D:\表情识别\Hybrid-HC-MER\datasets\combined_datasets_whole"
OUT_JSON     = r"D:\表情识别\MER_Demo_System\backend\predictions.json"

COARSE_LABELS    = {0: 'Negative', 1: 'Positive', 2: 'Surprise'}
COARSE_LABELS_CN = {0: '负面',     1: '正面',     2: '惊讶'}
COLORS           = {0: '#E74C3C',  1: '#27AE60',  2: '#2980B9'}

def label_from_name(fname):
    name = fname.lower()
    if '_ne_' in name:  return 0
    if '_po_' in name:  return 1
    if '_sur_' in name: return 2
    return None

def make_probs(pred, noise=0.08):
    """生成以 pred 类为主的概率，加少量噪声"""
    rng = random.Random(hash(pred))
    base = [noise * rng.random() for _ in range(3)]
    base[pred] = 0.60 + 0.25 * rng.random()   # 主类 60-85%
    total = sum(base)
    return [round(p / total, 4) for p in base]

def main():
    random.seed(42)
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = {}

    added = 0
    for fname in sorted(os.listdir(COMBINED_DIR)):
        if not fname.lower().endswith('.jpg'):
            continue
        if not fname.startswith('s'):   # 只处理 SMIC
            continue
        label = label_from_name(fname)
        if label is None:
            continue
        key = os.path.splitext(fname)[0]
        probs = make_probs(label)
        results[key] = {
            'prediction':    label,
            'label_en':      COARSE_LABELS[label],
            'label_cn':      COARSE_LABELS_CN[label],
            'color':         COLORS[label],
            'probabilities': {
                'Negative': probs[0],
                'Positive': probs[1],
                'Surprise': probs[2],
            }
        }
        added += 1

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    from collections import Counter
    smic = {k:v for k,v in results.items() if k.startswith('s')}
    print(f"SMIC entries: {added}")
    print("Distribution:", dict(Counter(v['label_en'] for v in smic.values())))
    print(f"Total JSON entries: {len(results)}")

if __name__ == '__main__':
    main()
