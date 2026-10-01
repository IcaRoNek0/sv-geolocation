#!/usr/bin/env python
"""Experimental visual car/year branch. Explicit panorama or visible-car crop only."""
import argparse
import json
from pathlib import Path

import numpy as np

from data.inference_views import load_image, to_views
from data.prepare import ViewConfig
from data.vehicle import car_view, mask_vehicle_crop
from inference_onnx import quiet_stderr
from models.fusion import fuse, top_counties
from models.vehicle_prior import coverage_prior, blend_vehicle_prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--backend", choices=("torch", "onnx"), default="torch")
    ap.add_argument("--mode", choices=("panorama", "vehicle_crop"), required=True,
                    help="vehicle_crop must actually contain the car; no automatic visibility detector")
    ap.add_argument("--prior", type=Path)
    ap.add_argument("--env-json", type=Path, help="Full probability JSON from inference.py")
    ap.add_argument("--strength", type=float, default=0.0,
                    help="Experimental fusion, disabled until validated; at most 0.35")
    ap.add_argument("--min-confidence", type=float, default=0.6)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    meta = json.loads((args.run / "classes.json").read_text(encoding="utf-8"))
    if meta.get("task") != "vehicle":
        raise ValueError("Expected a vehicle checkpoint export")
    cfg = ViewConfig(**meta["views"])
    img, _ = load_image(args.image, "panorama" if args.mode == "panorama" else "screenshot")
    if args.mode == "panorama":
        views = car_view(img, cfg, np.random.default_rng(0))
    else:
        views, _, _ = to_views(img, False, 1, cfg.fov, cfg.size, 1)
        views[0] = mask_vehicle_crop(views[0])
    x = views[None].transpose(0, 1, 4, 2, 3)
    mask = np.ones((1, 1), dtype=bool)
    if args.backend == "torch":
        import torch
        from models.env_model import EnvModel
        model = EnvModel(len(meta["adcodes"]), len(meta["cities"]), len(meta["provinces"]),
                         backbone=meta["backbone"], pretrained=False)
        checkpoint = args.run / "best.pt"
        if not checkpoint.exists():
            checkpoint = args.run / "last.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("run_meta") and state["run_meta"] != meta:
            raise ValueError("Checkpoint metadata mismatch")
        model.load_state_dict(state["model"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        with torch.inference_mode():
            out = model(torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device))
        logits = out["county"].float().cpu().numpy()
        years = out["city"].float().cpu().numpy()
    else:
        import onnxruntime as ort
        with quiet_stderr():
            sess = ort.InferenceSession(str(args.run / "model.onnx"), providers=["CPUExecutionProvider"])
        logits, years = sess.run(["county", "city"], {"views": x, "vmask": mask})
    vp = fuse(logits[0], adcodes=meta["adcodes"], from_logits=True)
    yp = fuse(years[0], adcodes=meta["cities"], from_logits=True)
    result = {"vehicle_top5": top_counties(vp), "year_top5": top_counties(yp),
              "vehicle_probabilities": vp, "year_probabilities": yp,
              "fusion_applied": False, "experimental": True}
    if args.env_json:
        if not args.prior:
            raise ValueError("--env-json requires --prior")
        env = json.loads(args.env_json.read_text(encoding="utf-8"))["county_probabilities"]
        coverage = json.loads(args.prior.read_text(encoding="utf-8"))["coverage"]
        prior, mass = coverage_prior(vp, yp, coverage, list(env))
        probs, applied = blend_vehicle_prior(env, prior, max(vp.values()), mass,
                                             args.strength, args.min_confidence)
        result.update(county_probabilities=probs, county_top5=top_counties(probs),
                      supported_mass=mass, fusion_applied=applied)
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("probabilities")},
                     ensure_ascii=False, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
