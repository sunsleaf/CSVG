import argparse
import json
import os
import re
import subprocess
from contextlib import redirect_stdout

import numpy as np

from scannet_utils import BoundingBox3D, ScanNetScene


def parse_bbox(data: dict | list) -> BoundingBox3D | list[BoundingBox3D]:
    if isinstance(data, list):
        result = []
        for item in data:
            assert isinstance(item, dict)
            assert "pmin" in item
            assert "pmax" in item
            result.append(
                BoundingBox3D(
                    pmin0=np.array(item["pmin"]),
                    pmax0=np.array(item["pmax"]),
                )
            )
        return result

    else:
        assert isinstance(data, dict)
        assert "pmin" in data
        assert "pmax" in data
        return BoundingBox3D(pmin0=np.array(data["pmin"]), pmax0=np.array(data["pmax"]))


def format_query(query: str) -> str:
    query = re.sub(r"[,.:;?!]", "", query)
    query = re.sub(r" +", " ", query)
    query = query.lower().strip()
    return query


def get_gt_query(formatted_query: str, gt_eval_results: dict) -> str | None:
    for eval_res in gt_eval_results:
        gt_query = format_query(eval_res["text"])
        if gt_query == formatted_query:
            return eval_res


def get_zsvg_query(formatted_query: str, zsvg_eval_results: dict) -> str | None:
    for eval_res in zsvg_eval_results:
        zsvg_query = format_query(eval_res["caption"])
        if zsvg_query == formatted_query:
            return eval_res


parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True)
parser.add_argument("--gt-file", type=str)
parser.add_argument("--zsvg3d-file", type=str)
parser.add_argument("--no-server", action="store_true")
parser.add_argument("--viz-port", type=int, default=8889)
parser.add_argument("--zsvg-fail", action="store_true")
parser.add_argument("--csvg-fail", action="store_true")
parser.add_argument("--anchor-not-required", action="store_true")
parser.add_argument("--experiment-name", type=str)
parser.add_argument("--distractor-required", action="store_true")
parser.add_argument("--func", type=str, action="append")
parser.add_argument("--no-func", type=str, action="append")
args = parser.parse_args()

eval_results_file = args.file
assert os.path.isfile(eval_results_file)
with open(eval_results_file) as f:
    eval_results = json.load(f)

if args.gt_file:
    gt_eval_results_file = args.gt_file
    assert os.path.isfile(gt_eval_results_file)
    with open(gt_eval_results_file) as f:
        gt_eval_results = json.load(f)

if args.zsvg3d_file:
    zsvg_eval_results_file = args.zsvg3d_file
    assert os.path.isfile(zsvg_eval_results_file)
    with open(zsvg_eval_results_file) as f:
        zsvg_eval_results = json.load(f)

scannet_root = "./data"
viz_dir_root = "./data/visualize_eval"
assert os.path.isdir(scannet_root)
if not os.path.isdir(viz_dir_root):
    os.system(f"mkdir -p {viz_dir_root}")

zsvg_should_work = not args.zsvg_fail
csvg_mask3d_should_work = not args.csvg_fail
csvg_gt_should_work = True

conf_str = f"csvg_{0 if args.csvg_fail else 1}"
if args.zsvg3d_file:
    conf_str += f"_zsvg_{0 if args.zsvg_fail else 1}"
if args.distractor_required:
    conf_str += "_with_distractor"
if args.experiment_name:
    conf_str += f"_{args.experiment_name}"

print()
print(f"config str: {conf_str}")
input("press enter to continue...")
print()

for i, eval_res in enumerate(eval_results):
    print(f"progress: {i} / {len(eval_results)}.")
    # if (
    #     eval_res["acc05"] is False
    #     and eval_res["acc025"] is False
    #     and "predicted_bbox" in eval_res
    # ):

    if "anchor_bboxes" not in eval_res or not eval_res["anchor_bboxes"]:
        if not args.anchor_not_required:
            continue

    if eval_res["acc05"] == csvg_mask3d_should_work:
        # assert eval_res["acc025"] is True

        query = format_query(eval_res["text"])

        if args.gt_file:
            gt_eval_res = get_gt_query(query, gt_eval_results)
            assert gt_eval_res is not None

            if gt_eval_res["acc05"] != csvg_gt_should_work:
                continue

        if args.zsvg3d_file:
            zsvg_eval_res = get_zsvg_query(query, zsvg_eval_results)
            if zsvg_eval_res is None:
                continue

            if (
                "pred_box" not in zsvg_eval_res
                or zsvg_eval_res["acc05"] != zsvg_should_work
            ):
                continue

        scene_id = eval_res["scene_id"]
        scene_path = os.path.join(scannet_root, "scans", scene_id)

        # print(json.dumps(gt_eval_res, sort_keys=True, indent=4))
        # print(json.dumps(zsvg_eval_res, sort_keys=True, indent=4))

        scene = ScanNetScene(
            scene_path,
            add_room_center=False,
            add_room_corners=False,
            cache_root="./data/instance_cache",
        )

        if args.distractor_required:
            target_id = eval_res["target_id"]
            if not scene.get_distractors(target_id):
                continue

        if args.func:
            if not any(func in eval_res["program"] for func in args.func):
                continue

        if args.no_func:
            if any(func in eval_res["program"] for func in args.no_func):
                continue

        print()
        print("=" * 30)
        print(f"scene id    : {scene_id}")
        print(f"target id   : {eval_res['target_id']}")
        print(f"query       : {eval_res['text']}")
        print(f"target label: {eval_res['target_label']}")
        print()
        print("program     :")
        print(">" * 10)
        print(eval_res["program"])
        print("<" * 10)
        print()
        print("csp         :")
        print(">" * 10)
        print(eval_res["csp_desc"])
        print("<" * 10)
        if args.gt_file:
            print()
            print("csvg gt program")
            print(">" * 10)
            print(gt_eval_res["program"])
            print("<" * 10)
        if args.zsvg3d_file:
            print()
            print("zsvg3d program")
            print(">" * 10)
            print(zsvg_eval_res["program"])
            print("<" * 10)
        print()
        print(f"query       : {eval_res['text']}")
        print()
        print("=" * 30)
        print()

        if args.zsvg3d_file:
            zsvg_bbox = BoundingBox3D(
                pcenter0=zsvg_eval_res["pred_box"][:3],
                psize0=zsvg_eval_res["pred_box"][3:],
            )

        gt_color = [0.4, 0.7, 1.0]
        pred_color_correct = [0.3, 1.0, 0.3]
        pred_color_wrong = [1.0, 0.2, 0.2]
        anchor_color = [1.0, 0.7, 0.0]
        distractor_color = [1.0, 0.3, 0.7]

        extra_bboxes = []

        if args.gt_file:
            csvg_gt_bbox = parse_bbox(gt_eval_res["predicted_bbox"])
            extra_bboxes.append(
                ScanNetScene.BBoxInfo(
                    pmin=csvg_gt_bbox.pmin,
                    pmax=csvg_gt_bbox.pmax,
                    color=pred_color_correct
                    if csvg_gt_should_work
                    else pred_color_wrong,
                    name="csvg-gt-pred",
                ),
            )

        if args.zsvg3d_file:
            extra_bboxes.append(
                ScanNetScene.BBoxInfo(
                    pmin=zsvg_bbox.pmin,
                    pmax=zsvg_bbox.pmax,
                    color=pred_color_correct if zsvg_should_work else pred_color_wrong,
                    name="zsvg-pred",
                ),
            )

        if args.gt_file:
            for name, bbox in gt_eval_res["anchor_bboxes"].items():
                bbox = parse_bbox(bbox)
                extra_bboxes.append(
                    ScanNetScene.BBoxInfo(
                        pmin=bbox.pmin,
                        pmax=bbox.pmax,
                        color=anchor_color,
                        name=f"csvg-gt-anchor-{name}",
                    )
                )

        pred_bbox = parse_bbox(eval_res["predicted_bbox"])
        extra_bboxes.append(
            ScanNetScene.BBoxInfo(
                pmin=pred_bbox.pmin,
                pmax=pred_bbox.pmax,
                color=pred_color_correct
                if csvg_mask3d_should_work
                else pred_color_wrong,
                name="csvg-mask3d-pred",
            )
        )

        anchor_bboxes = {
            name: parse_bbox(bbox) for name, bbox in eval_res["anchor_bboxes"].items()
        }
        for name, bbox in anchor_bboxes.items():
            assert isinstance(bbox, BoundingBox3D)
            extra_bboxes.append(
                ScanNetScene.BBoxInfo(
                    pmin=bbox.pmin,
                    pmax=bbox.pmax,
                    color=anchor_color,
                    name="csvg-mask3d-anchor",
                )
            )

        if args.distractor_required:
            target_id = eval_res["target_id"]
            distractors = scene.get_distractors(target_id)
            for inst in distractors:
                extra_bboxes.append(
                    ScanNetScene.BBoxInfo(
                        pmin=inst.bbox.pmin,
                        pmax=inst.bbox.pmax,
                        color=distractor_color,
                        name="distractor",
                    )
                )

        viz_dir = scene.visualize_pyviz3d(
            viz_dir_root,
            target_id=eval_res["target_id"],
            target_color=gt_color,
            extra_bboxes=extra_bboxes,
        )

        if not args.no_server:
            try:
                proc = subprocess.Popen(
                    ["python", "-m", "http.server", str(args.viz_port), "-d", viz_dir],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"serving visualization at http://0.0.0.0:{args.viz_port}/")
                print()
                cmd = input("press enter to continue...\n")
                print()

                if cmd.lower().strip() == "store":
                    file_path = f"./figure_data_{conf_str}.txt"
                    with open(file_path, "a") as f:
                        with redirect_stdout(f):
                            print()
                            print("=" * 30)
                            print(f"scene id    : {scene_id}")
                            print(f"target id   : {eval_res['target_id']}")
                            print(f"query       : {eval_res['text']}")
                            print(f"target label: {eval_res['target_label']}")
                            print()
                            print("program     :")
                            print(">" * 10)
                            print(eval_res["program"])
                            print("<" * 10)
                            print()
                            print("csp         :")
                            print(">" * 10)
                            print(eval_res["csp_desc"])
                            print("<" * 10)
                            if args.gt_file:
                                print()
                                print("csvg gt program")
                                print(">" * 10)
                                print(gt_eval_res["program"])
                                print("<" * 10)
                            if args.zsvg3d_file:
                                print()
                                print("zsvg3d program")
                                print(">" * 10)
                                print(zsvg_eval_res["program"])
                                print("<" * 10)
                            print()
                            print("=" * 30)
                            print()
                    print()
                    print(f"query info stored to {file_path}")
                    input("press enter to continue...")
                    print()

            except Exception:
                pass
            finally:
                proc.kill()
