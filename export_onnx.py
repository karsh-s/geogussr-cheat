"""
Export the v7 model to ONNX so it can run inside the browser extension.

Produces, in extension/model/:
  geoclip.onnx        float32 graph
  geoclip.int8.onnx   dynamically quantised, roughly 4x smaller
  model_meta.json     classes, input size, normalisation constants

The meta file matters: the extension must preprocess images exactly as
PyTorch did (resize, centre crop, these mean/std values). Getting that
subtly wrong is the usual way a ported model quietly loses accuracy, so
the constants travel with the weights rather than being retyped in JS.

Run:  python3 export_onnx.py
"""

import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

HERE  = os.path.dirname(os.path.abspath(__file__))
CKPT  = os.path.join(HERE, 'models', 'geoclip_v7', 'best_model.pth')
OUT   = os.path.join(HERE, 'extension', 'model')
SIZE  = 224
os.makedirs(OUT, exist_ok=True)

# CLIP's normalisation — the extension must use exactly these
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD  = [0.26862954, 0.26130258, 0.27577711]


class Head(nn.Module):
    def __init__(self, dim, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(dim, 512), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(512, n_cls))

    def forward(self, x):
        return self.net(x)


class GeoCLIP(nn.Module):
    """visual -> L2 normalise -> head, as one graph."""
    def __init__(self, visual, head):
        super().__init__()
        self.visual, self.head = visual, head

    def forward(self, x):
        return self.head(F.normalize(self.visual(x), dim=-1))


def main():
    if not os.path.exists(CKPT):
        raise SystemExit(f'No checkpoint at {CKPT}')

    ck = torch.load(CKPT, map_location='cpu')
    mode    = ck['mode']
    classes = list(ck['classes'])
    cm, cp  = ck.get('clip_model', 'ViT-B-16'), ck.get('clip_pretrain',
                                                       'laion2b_s34b_b88k')
    print(f'checkpoint: {mode}, {len(classes)} classes, {cm}/{cp}')

    clip_model, _, _ = open_clip.create_model_and_transforms(cm, pretrained=cp)
    if mode == 'finetune':
        clip_model.visual.load_state_dict(ck['visual'])
    head = Head(clip_model.visual.output_dim, len(classes))
    head.load_state_dict(ck['head'])

    model = GeoCLIP(clip_model.visual, head).eval()

    dummy = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad():
        ref = model(dummy).numpy()

    fp32 = os.path.join(OUT, 'geoclip.onnx')
    # dynamo=False selects the legacy exporter. The newer one splits
    # weights into a side-car .onnx.data file, which is awkward to ship in
    # an extension and upsets the quantiser's shape inference.
    torch.onnx.export(
        model, dummy, fp32,
        input_names=['image'], output_names=['logits'],
        dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17, do_constant_folding=True, dynamo=False)
    print(f'exported {fp32}  ({os.path.getsize(fp32)/1e6:.0f} MB)')

    # numerical check against PyTorch
    import onnxruntime as ort
    sess = ort.InferenceSession(fp32, providers=['CPUExecutionProvider'])
    got = sess.run(None, {'image': dummy.numpy()})[0]
    err = float(np.abs(got - ref).max())
    print(f'fp32 max abs difference vs PyTorch: {err:.2e} '
          f'{"OK" if err < 1e-3 else "TOO LARGE"}')

    # int8 — MEASURED AS UNUSABLE for this model, kept for reference.
    # On 60 real Street View images, dynamic int8 agreed with PyTorch on
    # only 48% of top-1 predictions (53% with per_channel=True), and 61%
    # top-5 overlap. That is not a small accuracy cost, it is a different
    # and much worse model: ViT attention and LayerNorm do not survive
    # per-tensor weight quantisation without calibration. fp32 ONNX, by
    # contrast, matches PyTorch exactly (100% agreement).
    # If the extension needs a smaller file, try fp16 (~173MB) next and
    # measure it the same way before trusting it.
    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8 = os.path.join(OUT, 'geoclip.int8.onnx')
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    print(f'quantised {int8}  ({os.path.getsize(int8)/1e6:.0f} MB)')

    sess8 = ort.InferenceSession(int8, providers=['CPUExecutionProvider'])
    got8 = sess8.run(None, {'image': dummy.numpy()})[0]
    agree = int((got8.argmax(1) == ref.argmax(1)).sum())
    print(f'int8 max abs difference: {float(np.abs(got8 - ref).max()):.3f}'
          f'  | same top-1 on the probe batch: {agree}/{len(ref)}')

    meta = {'classes': classes, 'input_size': SIZE, 'mean': MEAN, 'std': STD,
            'clip_model': cm, 'mode': mode}
    with open(os.path.join(OUT, 'model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'wrote {OUT}/model_meta.json')


if __name__ == '__main__':
    main()
