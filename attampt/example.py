import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, TimesformerForVideoClassification

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import av
except ImportError:
    av = None

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

try:
    import cv2
except ImportError:
    cv2 = None


DEFAULT_MODEL = "facebook/timesformer-base-finetuned-k400"
DEFAULT_BASE_URL = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val"


@dataclass
class VideoEntry:
    label: str
    filename: str
    source: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    attempt_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Iterate over videos from list.md and save DAAM visualizations."
    )
    parser.add_argument("--list-path", type=Path, default=attempt_dir / "list.md")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "vis" / "daam")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--label", default=None, help="Only process one markdown section label.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto when omitted.")
    return parser.parse_args()


def parse_video_list(list_path: Path, base_url: str) -> list[VideoEntry]:
    entries: list[VideoEntry] = []
    current_label: str | None = None

    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            current_label = line[3:].strip()
            continue
        if line.startswith("- ") and current_label:
            filename = line[2:].strip()
            source = f"{base_url}/{current_label}/{filename}"
            entries.append(VideoEntry(label=current_label, filename=filename, source=source))

    return entries


def get_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_frame_indices(total_frames: int, num_samples: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("Video has no frames.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if total_frames == 1:
        return np.zeros((num_samples,), dtype=np.int64)
    return np.linspace(0, total_frames - 1, num_samples).astype(np.int64)


def select_sampled_frames(frames: list[np.ndarray], num_samples: int) -> list[np.ndarray]:
    frame_indices = sample_frame_indices(len(frames), num_samples)
    return [frames[index] for index in frame_indices]


def read_video_with_pyav(video_source: str, num_samples: int) -> list[np.ndarray]:
    if av is None:
        raise RuntimeError("PyAV is not available.")
    container = av.open(video_source)
    try:
        decoded_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    finally:
        container.close()
    return select_sampled_frames(decoded_frames, num_samples)


def read_video_with_imageio(video_source: str, num_samples: int) -> list[np.ndarray]:
    if imageio is None:
        raise RuntimeError("imageio is not available.")
    reader = imageio.get_reader(video_source, format="ffmpeg")
    try:
        decoded_frames = [frame for frame in reader]
    finally:
        reader.close()
    return select_sampled_frames(decoded_frames, num_samples)


def read_video_with_opencv(video_source: str, num_samples: int) -> list[np.ndarray]:
    if cv2 is None:
        raise RuntimeError("OpenCV is not available.")

    capture = cv2.VideoCapture(video_source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video source: {video_source}")

    decoded_frames: list[np.ndarray] = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            decoded_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    return select_sampled_frames(decoded_frames, num_samples)


def load_video_frames(video_source: str, num_samples: int) -> list[np.ndarray]:
    errors: list[str] = []

    if av is not None:
        try:
            return read_video_with_pyav(video_source, num_samples)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"pyav failed: {exc}")

    if imageio is not None:
        try:
            return read_video_with_imageio(video_source, num_samples)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"imageio failed: {exc}")

    if cv2 is not None:
        try:
            return read_video_with_opencv(video_source, num_samples)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"opencv failed: {exc}")

    if not errors:
        raise RuntimeError("No video backend is available. Install PyAV, imageio-ffmpeg, or opencv-python.")
    raise RuntimeError("; ".join(errors))


def reshape_tokens_to_map(tensor: torch.Tensor) -> torch.Tensor:
    batch_size, num_tokens, hidden_size = tensor.shape
    side = int(math.sqrt(num_tokens - 1))
    if side * side != num_tokens - 1:
        raise ValueError(f"Unexpected token count for DAAM map: {num_tokens}")
    return tensor[:, 1:, :].reshape(batch_size, side, side, hidden_size).permute(0, 3, 1, 2)


class TimesformerDAAM:
    def __init__(self, model: TimesformerForVideoClassification, device: torch.device):
        self.model = model
        self.device = device
        self.activations: list[torch.Tensor] = []
        self.gradients: list[torch.Tensor] = []
        self.handles = []

        for layer in self.model.timesformer.encoder.layer:
            self.handles.append(layer.attention.attention.register_forward_hook(self._save_activation))
            self.handles.append(layer.attention.output.register_forward_hook(self._register_gradient_hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _save_activation(self, module, inputs, outputs) -> None:
        hidden_states = inputs[0]
        if not hidden_states.requires_grad:
            return

        batch_size, token_count, channels = hidden_states.shape
        qkv = (
            module.qkv(hidden_states)
            .reshape(batch_size, token_count, 3, module.num_heads, channels // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        _, key, value = qkv[0], qkv[1], qkv[2]
        attention = (qkv[0] @ key.transpose(-2, -1)) * module.scale
        attention = module.attn_drop(attention.softmax(dim=-1))
        cls_focus = attention[:, :, 0].unsqueeze(-1)
        activation = (cls_focus * value).transpose(1, 2).reshape(batch_size, token_count, channels)
        self.activations.append(activation.detach().cpu())

    def _register_gradient_hook(self, module, inputs, outputs) -> None:
        projection_input = inputs[0]
        if not projection_input.requires_grad:
            return
        projection_input.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = [grad[:, 0, :].detach().cpu()] + self.gradients

    def generate(self, pixel_values: torch.Tensor, target_index: int | None = None) -> tuple[np.ndarray, int, torch.Tensor]:
        self.activations.clear()
        self.gradients.clear()

        pixel_values = pixel_values.to(self.device)
        pixel_values = pixel_values.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits

        if target_index is None:
            target_index = logits.argmax(dim=-1).item()

        logits[:, target_index].sum().backward()

        if not self.activations or not self.gradients:
            raise RuntimeError("DAAM hooks did not capture activations or gradients.")

        cams = []
        for gradient, activation in zip(self.gradients, self.activations):
            activation_map = reshape_tokens_to_map(activation).numpy()
            weights = gradient.unsqueeze(-1).unsqueeze(-1).numpy()
            weighted = np.maximum(weights, 0.0) * activation_map
            cam = np.maximum(weighted.sum(axis=1), 0.0)
            cams.append(self._normalize_per_frame(cam))

        accumulated = np.sum(np.stack(cams, axis=0), axis=0)
        final_heatmaps = self._nonlinear_normalize(accumulated)
        return final_heatmaps, target_index, logits.detach().cpu()

    @staticmethod
    def _normalize_per_frame(cam: np.ndarray) -> np.ndarray:
        minimum = cam.min(axis=(1, 2), keepdims=True)
        maximum = cam.max(axis=(1, 2), keepdims=True)
        return (cam - minimum) / (maximum - minimum + 1e-10)

    @staticmethod
    def _nonlinear_normalize(cam: np.ndarray) -> np.ndarray:
        global_max = max(float(cam.max()), 1e-10)
        cam = np.maximum(cam, 0.0) / global_max
        cam = 1.0 / (1.0 + np.exp(-5.0 * cam)) - 0.5
        minimum = cam.min(axis=(1, 2), keepdims=True)
        maximum = cam.max(axis=(1, 2), keepdims=True)
        return (cam - minimum) / (maximum - minimum + 1e-10)


def resize_heatmap(heatmap: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    tensor = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return resized[0, 0].numpy()


def apply_heatmap(frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    frame_float = frame.astype(np.float32) / 255.0
    heatmap_rgb = plt.get_cmap("jet")(heatmap)[..., :3].astype(np.float32)
    blended = 0.55 * frame_float + 0.45 * heatmap_rgb
    return np.clip(blended, 0.0, 1.0)


def save_visualization_grid(
    frames: Iterable[np.ndarray],
    heatmaps: np.ndarray,
    output_path: Path,
    source_label: str,
    predicted_label: str,
) -> None:
    frames = list(frames)
    num_frames = len(frames)
    fig, axes = plt.subplots(2, num_frames, figsize=(3 * num_frames, 6), squeeze=False)
    fig.suptitle(f"{source_label} -> {predicted_label}", fontsize=14)

    for index, frame in enumerate(frames):
        resized_heatmap = resize_heatmap(heatmaps[index], frame.shape[:2])
        overlay = apply_heatmap(frame, resized_heatmap)

        axes[0, index].imshow(frame)
        axes[0, index].set_title(f"Frame {index}")
        axes[0, index].axis("off")

        axes[1, index].imshow(overlay)
        axes[1, index].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(rows: list[dict[str, str]], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "filename", "source", "predicted_label", "top5", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)


def format_topk(logits: torch.Tensor, id2label: dict[int, str], topk: int = 5) -> str:
    scores = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(scores[0], k=min(topk, scores.shape[-1]))
    parts = []
    for value, index in zip(values.tolist(), indices.tolist()):
        parts.append(f"{id2label[index]}:{value:.4f}")
    return " | ".join(parts)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    entries = parse_video_list(args.list_path, args.base_url)
    if args.label:
        entries = [entry for entry in entries if entry.label == args.label]
    if args.max_videos is not None:
        entries = entries[: args.max_videos]
    if not entries:
        raise ValueError("No videos found to process.")

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = TimesformerForVideoClassification.from_pretrained(args.model_name).to(device)
    model.eval()

    num_samples = args.num_samples or model.config.num_frames
    daam = TimesformerDAAM(model, device)
    summary_rows: list[dict[str, str]] = []

    try:
        for index, entry in enumerate(entries, start=1):
            source_label = f"{entry.label}/{entry.filename}"
            print(f"[{index}/{len(entries)}] Processing {source_label}")

            try:
                frames = load_video_frames(entry.source, num_samples)
                inputs = processor(frames, return_tensors="pt")
                heatmaps, predicted_index, logits = daam.generate(inputs["pixel_values"])
                predicted_label = model.config.id2label[predicted_index]

                output_path = args.output_dir / entry.label / f"{Path(entry.filename).stem}_daam.png"
                save_visualization_grid(
                    frames=frames,
                    heatmaps=heatmaps,
                    output_path=output_path,
                    source_label=source_label,
                    predicted_label=predicted_label,
                )

                top5 = format_topk(logits, model.config.id2label)
                summary_rows.append(
                    {
                        "label": entry.label,
                        "filename": entry.filename,
                        "source": entry.source,
                        "predicted_label": predicted_label,
                        "top5": top5,
                        "status": "ok",
                    }
                )
                print(f"  predicted={predicted_label}")
                print(f"  saved={output_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"  failed={exc}")
                summary_rows.append(
                    {
                        "label": entry.label,
                        "filename": entry.filename,
                        "source": entry.source,
                        "predicted_label": "",
                        "top5": "",
                        "status": f"error: {exc}",
                    }
                )
    finally:
        daam.close()

    write_summary(summary_rows, args.output_dir / "summary.csv")
    print(f"Summary written to {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
