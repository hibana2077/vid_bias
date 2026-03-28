from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch
from torchcodec.decoders import VideoDecoder

processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")

# video_url = 'https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/high_jump/3sBYgcb4bEY_000003_000013.mp4'
video_url = 'https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/marching/1m-Kdky1y84_000022_000032.mp4'
# video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"

vr = VideoDecoder(video_url)

num_frames = len(vr)
num_samples = 8

frame_idx = np.linspace(0, num_frames - 1, num_samples).astype(int)

video = vr.get_frames_at(indices=frame_idx).data  # (T, C, H, W)

inputs = processor(list(video), return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

logits = outputs.logits
attn = outputs.attentions

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

# 看一下 shape
print("pixel_values shape:", inputs["pixel_values"].shape)
print("num attention layers:", len(attn))
print("first attention shape:", attn[0].shape)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

layer_idx = 5
A_all = attn[layer_idx]

print(A_all.shape)

# 判斷 frame 數
if A_all.ndim == 4 and A_all.shape[-1] == 197:
    num_frames = A_all.shape[0]
elif A_all.ndim == 4:
    num_frames = 1   # 不是 per-frame attention
else:
    raise ValueError(f"Unexpected attention shape: {A_all.shape}")

for frame_idx in range(num_frames):

    A = A_all

    # 取對應 frame
    if A.ndim == 4 and A.shape[-1] == 197:
        A = A[frame_idx]          # [heads, 197, 197]
    elif A.ndim == 4:
        A = A[0]                  # [heads, seq, seq]

    # 平均 heads
    A = A.mean(dim=0)

    if A.shape != (197, 197):
        print(f"Skip frame {frame_idx}, shape = {A.shape}")
        continue

    cls_to_patch = A[0, 1:]
    heatmap = cls_to_patch.reshape(14, 14)

    # 取 frame
    frame = video[frame_idx].permute(1, 2, 0).cpu().numpy()

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    H, W = frame.shape[:2]

    heatmap = heatmap.unsqueeze(0).unsqueeze(0).float()
    heatmap = F.interpolate(
        heatmap,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )[0, 0].cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.title(f"Layer {layer_idx} - Frame {frame_idx}")
    plt.imshow(frame)
    plt.imshow(heatmap, alpha=0.5, cmap="jet")
    plt.axis("off")
    plt.show()