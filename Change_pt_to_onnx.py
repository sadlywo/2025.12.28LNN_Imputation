import argparse
from typing import Dict, Optional

import torch

from config import DataConfig, ModelConfig
from models import build_model


def _unwrap_state_dict(checkpoint: Dict) -> Dict:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _infer_model_name(state_dict: Dict) -> str:
    keys = state_dict.keys()
    if any(k.startswith("physics_coupling") for k in keys) or "gyro_noise_scale" in keys:
        return "physics"
    if "input_proj.weight" in keys or any(k.startswith("encoder.layers") for k in keys):
        return "transformer"
    if any(k.startswith("rnn.") for k in keys):
        return "gru"
    if any(k.startswith("cfc.") for k in keys):
        return "cfc"
    return "cfc"


class _ExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            return outputs[0], outputs[1]
        return outputs


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    model_name: Optional[str],
    hidden_units: int,
    seq_len: int,
    input_dim: int,
    device: str,
    opset: int,
) -> None:
    device = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _unwrap_state_dict(checkpoint)
    inferred_name = _infer_model_name(state_dict)
    final_name = model_name or inferred_name

    model = build_model(model_name=final_name, input_dim=input_dim, hidden_dim=hidden_units, output_dim=6)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    export_model = _ExportWrapper(model)
    dummy_input = torch.randn(1, seq_len, input_dim, device=device)

    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["inputs"],
        output_names=["pred", "uncertainty"],
        dynamic_axes={
            "inputs": {0: "batch", 1: "time"},
            "pred": {0: "batch", 1: "time"},
            "uncertainty": {0: "batch", 1: "time"},
        },
    )
    print(f"[ONNX] Exported {final_name} model to {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    parser = argparse.ArgumentParser(description="Export trained IMU imputation model to ONNX")
    parser.add_argument("--checkpoint", default="best_model.pt", help="Path to .pt checkpoint")
    parser.add_argument("--output", default="best_model.onnx", help="Output ONNX file path")
    parser.add_argument("--model-name", default=None, help="Override model name (cfc/physics/gru/transformer)")
    parser.add_argument("--hidden-units", type=int, default=model_cfg.hidden_units, help="Hidden units")
    parser.add_argument("--seq-len", type=int, default=data_cfg.seq_len, help="Sequence length")
    parser.add_argument("--input-dim", type=int, default=model_cfg.input_dim, help="Input feature dim")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Export device")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_name=args.model_name,
        hidden_units=args.hidden_units,
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        device=args.device,
        opset=args.opset,
    )
