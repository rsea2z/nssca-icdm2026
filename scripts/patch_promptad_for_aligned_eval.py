from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch external PromptAD checkout for aligned evaluation bookkeeping.")
    parser.add_argument("--promptad-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.promptad_root)
    patch_visa_loader(root / "datasets" / "visa.py")
    patch_metrics(root / "utils" / "metrics.py")
    patch_csv_utils(root / "utils" / "csv_utils.py")
    patch_train_cls(root / "train_cls.py")
    patch_test_cls(root / "test_cls.py")
    patch_downloader_ssl(root / "PromptAD" / "CLIPAD" / "pretrained.py")
    print(f"patched {root}")


def patch_visa_loader(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace('glob.glob(os.path.join(root_path, defect_type) + "/*.JPG")', 'glob.glob(os.path.join(root_path, defect_type) + "/*.*")')
    text = text.replace("os.path.basename(s)[:-4] + '.png'", "os.path.basename(s)[:-4] + '_mask.png'")
    path.write_text(text, encoding="utf-8")


def patch_metrics(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    old = """    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)

    result_dict = {'i_roc': img_roc_auc * 100}
"""
    new = """    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    anomaly_scores = img_scores[gt_list == 1]
    normal_scores = img_scores[gt_list == 0]
    if anomaly_scores.size == 0 or normal_scores.size == 0:
        normal_fpr_at_95_tpr = np.nan
    else:
        threshold = np.quantile(anomaly_scores, 0.05)
        normal_fpr_at_95_tpr = np.mean(normal_scores >= threshold)

    result_dict = {'i_roc': img_roc_auc * 100, 'i_fpr95': normal_fpr_at_95_tpr * 100}
"""
    if old in text:
        text = text.replace(old, new)
    elif "'i_fpr95'" not in text:
        raise RuntimeError(f"Could not patch metric_cal_img in {path}.")
    path.write_text(text, encoding="utf-8")


def patch_csv_utils(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace("keys = ['i_roc', 'p_roc']", "keys = ['i_roc', 'p_roc', 'i_fpr95']")
    path.write_text(text, encoding="utf-8")


def patch_train_cls(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    pixel_block = """        pixel_result_dict = metric_cal_pix(np.array(score_maps), gt_mask_list)\n        result_dict.update(pixel_result_dict)\n"""
    while pixel_block + pixel_block in text:
        text = text.replace(pixel_block + pixel_block, pixel_block)
    if pixel_block in text:
        path.write_text(text, encoding="utf-8")
        return
    old = """        test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))\n        result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps))\n"""
    new = """        test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))\n        result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps))\n        pixel_result_dict = metric_cal_pix(np.array(score_maps), gt_mask_list)\n        result_dict.update(pixel_result_dict)\n"""
    if old not in text:
        raise RuntimeError(f"Could not patch train_cls result block in {path}.")
    text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def patch_test_cls(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace("p_roc = round(metrics['i_roc'], 2)", "p_roc = round(metrics['p_roc'], 2)")
    pixel_block = """    pixel_result_dict = metric_cal_pix(np.array(score_maps), gt_mask_list)\n    result_dict.update(pixel_result_dict)\n"""
    while pixel_block + pixel_block in text:
        text = text.replace(pixel_block + pixel_block, pixel_block)
    if pixel_block in text:
        path.write_text(text, encoding="utf-8")
        return
    old = """    test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list,\n                                                             resolution=(args.resolution, args.resolution))\n    result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps))\n\n    return result_dict\n"""
    new = """    test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list,\n                                                             resolution=(args.resolution, args.resolution))\n    result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps))\n    pixel_result_dict = metric_cal_pix(np.array(score_maps), gt_mask_list)\n    result_dict.update(pixel_result_dict)\n\n    return result_dict\n"""
    if old not in text:
        raise RuntimeError(f"Could not patch test_cls result block in {path}.")
    text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def patch_downloader_ssl(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "import ssl" not in text:
        text = text.replace("import urllib", "import urllib\nimport ssl")
    old = "with urllib.request.urlopen(url) as source, open(download_target, \"wb\") as output:"
    new = "with urllib.request.urlopen(url, context=ssl._create_unverified_context()) as source, open(download_target, \"wb\") as output:"
    if old in text:
        text = text.replace(old, new)
    elif new not in text:
        raise RuntimeError(f"Could not patch urllib downloader in {path}.")
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
