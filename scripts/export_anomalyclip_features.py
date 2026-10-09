from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AnomalyCLIP features and anomaly maps to .npz files.")
    parser.add_argument("--anomalyclip-root", default="external/AnomalyCLIP-main")
    parser.add_argument("--dataset", choices=["mvtec", "visa"], required=True)
    parser.add_argument("--data-path", required=True, help="Dataset root containing AnomalyCLIP meta.json.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--modes", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--splits", nargs="*", default=None, help="Optional few-shot split JSON files; exports their calibration+test union.")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--clip-backbone", default="ViT-L/14@336px")
    parser.add_argument("--clip-cache-dir", default="external/clip_cache")
    parser.add_argument("--features-list", nargs="+", type=int, default=[24])
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n-ctx", type=int, default=12)
    parser.add_argument("--t-n-ctx", type=int, default=4)
    parser.add_argument("--feature-layer-start", type=int, default=0)
    parser.add_argument("--prototype-layer", type=int, default=-1, help="Patch feature layer to export; -1 means last.")
    parser.add_argument("--dpam-layer", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N sample limit for debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anomalyclip_root = Path(args.anomalyclip_root).resolve()
    if str(anomalyclip_root) not in sys.path:
        sys.path.insert(0, str(anomalyclip_root))

    import numpy as np
    import torch
    from scipy.ndimage import gaussian_filter
    from tqdm import tqdm

    import AnomalyCLIP_lib
    from dataset import Dataset
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform

    setup_seed(args.seed, torch, np)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else default_checkpoint(anomalyclip_root, args.dataset)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    design = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }
    model, _ = AnomalyCLIP_lib.load(
        args.clip_backbone,
        device=device,
        design_details=design,
        download_root=args.clip_cache_dir,
    )
    model.eval()

    preprocess, target_transform = get_transform(args)
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), design)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=args.dpam_layer)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    allowed_paths = load_split_paths(args.splits) if args.splits else None
    count = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for mode in args.modes:
            dataset = Dataset(
                root=args.data_path,
                transform=preprocess,
                target_transform=target_transform,
                dataset_name=args.dataset,
                mode=mode,
            )
            if allowed_paths is not None:
                dataset.data_all = filter_data_all(dataset.data_all, args.data_path, allowed_paths)
                dataset.length = len(dataset.data_all)
            if len(dataset) == 0:
                print(f"mode={mode}: no samples selected")
                continue

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
            for items in tqdm(dataloader, desc=f"export:{mode}"):
                if args.limit is not None and count >= args.limit:
                    break
                count += export_one_sample(
                    items=items,
                    args=args,
                    model=model,
                    text_features=text_features,
                    output_dir=output_dir,
                    manifest=manifest,
                    device=device,
                    torch=torch,
                    np=np,
                    gaussian_filter=gaussian_filter,
                    AnomalyCLIP_lib=AnomalyCLIP_lib,
                )
            if args.limit is not None and count >= args.limit:
                break

    print(f"exported {count} samples")
    print(f"manifest: {manifest_path}")


def export_one_sample(
    *,
    items,
    args: argparse.Namespace,
    model,
    text_features,
    output_dir: Path,
    manifest,
    device: str,
    torch,
    np,
    gaussian_filter,
    AnomalyCLIP_lib,
) -> int:
    image = items["img"].to(device)
    gt_mask = items["img_mask"]
    gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
    image_path = str(items["img_path"][0])
    class_name = str(items["cls_name"][0])
    anomaly_label = int(items["anomaly"][0])

    with torch.no_grad():
        image_features, patch_features = model.encode_image(
            image,
            args.features_list,
            DPAM_layer=args.dpam_layer,
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_probs = image_features @ text_features.permute(0, 2, 1)
        text_probs = (text_probs / 0.07).softmax(-1)
        image_score = float(text_probs[:, 0, 1].detach().cpu().item())

        selected_patch_features = []
        anomaly_map_list = []
        for layer_idx, patch_feature in enumerate(patch_features):
            if layer_idx < args.feature_layer_start:
                continue
            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
            selected_patch_features.append(patch_feature)
            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
            similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
            anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
            anomaly_map_list.append(anomaly_map)

        if not selected_patch_features:
            raise RuntimeError("No patch features selected; check --features-list and --feature-layer-start.")

        base_map = torch.stack(anomaly_map_list).sum(dim=0)
        base_map = torch.stack(
            [torch.from_numpy(gaussian_filter(m, sigma=args.sigma)) for m in base_map.detach().cpu()],
            dim=0,
        )[0].numpy().astype("float32")

        patch_feature = selected_patch_features[args.prototype_layer]
        features = patch_tokens_to_grid(patch_feature[:, 1:, :].detach().cpu().numpy()[0]).astype("float32")
        mask = gt_mask.detach().cpu().numpy()[0]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        mask = mask.astype("uint8")

    feature_path = feature_output_path(output_dir, args.data_path, image_path)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_path,
        features=features,
        base_map=base_map,
        mask=mask,
        image_path=np.array(image_path),
        class_name=np.array(class_name),
        image_score=np.array(image_score, dtype=np.float32),
        anomaly_label=np.array(anomaly_label, dtype=np.int64),
    )
    manifest.write(
        json.dumps(
            {
                "image_path": image_path,
                "feature_path": str(feature_path.resolve()),
                "class_name": class_name,
                "anomaly": anomaly_label,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    return 1


def load_split_paths(split_paths: list[str]) -> set[str]:
    allowed: set[str] = set()
    for split_path in split_paths:
        with Path(split_path).open("r", encoding="utf-8") as handle:
            split = json.load(handle)
        for section in ("calibration", "test"):
            for item in split.get(section, []):
                allowed.add(str(Path(item["image_path"]).resolve()))
    return allowed


def filter_data_all(data_all: list[dict[str, object]], data_root: str, allowed_paths: set[str]) -> list[dict[str, object]]:
    root = Path(data_root).resolve()
    return [
        item
        for item in data_all
        if str((root / str(item["img_path"])).resolve()) in allowed_paths
    ]


def setup_seed(seed: int, torch_module, np_module) -> None:
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    np_module.random.seed(seed)
    random.seed(seed)
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False


def default_checkpoint(anomalyclip_root: Path, dataset: str) -> Path:
    # The official test.sh evaluates MVTec with the VisA-trained checkpoint and VisA with the MVTec-trained one.
    name = "9_12_4_multiscale" if dataset == "mvtec" else "9_12_4_multiscale_visa"
    return anomalyclip_root / "checkpoints" / name / "epoch_15.pth"


def patch_tokens_to_grid(tokens):
    side = int(tokens.shape[0] ** 0.5)
    if side * side != tokens.shape[0]:
        return tokens
    return tokens.reshape(side, side, tokens.shape[-1])


def feature_output_path(output_dir: Path, data_root: str, image_path: str) -> Path:
    root = Path(data_root).resolve()
    path = Path(image_path).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)
    return output_dir / relative.with_suffix(".npz")


if __name__ == "__main__":
    main()
