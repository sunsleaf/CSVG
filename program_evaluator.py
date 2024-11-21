# run the generated programs and evalute the grounding results

import argparse
import functools
import json
import os
import random
import sys
import traceback
from dataclasses import dataclass
from multiprocessing import Pool

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Self

from program_functions_csp import reset_csp_solver, run_csp_solver
from scannet_utils import ObjInstance, ScanNetScene
from scope_env import (
    TargetInfo,
    get_eval_scope,
    set_instance_map,
    set_room_center,
)


@dataclass
class EvalSingleResultScanRefer:
    acc05: bool = False
    acc05_potential: bool = False
    acc025: bool = False
    acc025_potential: bool = False
    is_unique: bool = True
    eval_result: dict = None
    error: bool = False
    llm_used: bool = False


@dataclass
class EvalSingleResultNr3D:
    eval_result: dict = None
    is_hard: bool = False
    is_view_dependent: bool = False
    success: bool = False
    could_success: bool = False
    error: bool = False
    llm_used: bool = False


def eval_single_scanrefer(
    query_info: tuple[int, dict],
    mask3d_pred_path: str | None = None,
    maskcluster_pred_path: str | None = None,
    cache_root: str | None = None,
    solver_type: str = "default",
    select_solution: str = "min_dist",
    verbose: int = 0,
) -> EvalSingleResultScanRefer:
    query_id, query = query_info
    result = EvalSingleResultScanRefer()
    result.eval_result = query.copy()
    result.eval_result["query_id"] = query_id
    result.eval_result["acc05"] = False
    result.eval_result["acc025"] = False

    # todo:
    # loading the scene for every query is slow!
    # we should load all relevant scens at the beginning.

    scene_id = query["scene_id"]
    scene = ScanNetScene(
        f"./data/scans/{scene_id}",
        mask3d_pred_path=mask3d_pred_path,
        maskcluster_pred_path=maskcluster_pred_path,
        add_room_center=True,
        add_room_corners=True,
        cache_root=cache_root,
    )
    # scene.visualize()

    # used to get groundtruth bounding boxes
    gt_scene = ScanNetScene(
        f"./data/scans/{scene_id}",
        mask3d_pred_path=None,
        maskcluster_pred_path=None,
        add_room_center=False,
        add_room_corners=False,
        cache_root=cache_root,
    )

    # check if the query is "unique" or "multiple"
    result.is_unique = gt_scene.is_unique_label(
        query["target_label"].replace("_", " ").lower().strip()
    )

    if verbose >= 2:
        print(f"==================> vvv Query {query_id} vvv <==================")
        print(query["text"])
        print(query["scene_id"])
        print(query["target_id"])
        print(query["target_label"])
        print(query["program"])
        print()

    use_min_dist_heuristic = True
    query_text = query["text"].lower()
    if any(
        x in query_text
        for x in {"far", "across", "opposite", "away", "remote", "distant"}
    ):
        use_min_dist_heuristic = False

    try:
        # this is such a mess... it would be better to implement an interpreter manually,
        # which will provide better error messages and debugging support

        set_instance_map(scene.get_instance_map())
        set_room_center(scene.get_room_center())
        reset_csp_solver()
        TargetInfo.reset()

        exec(query["program"], get_eval_scope(use_type_check_funcs=False))
        run_csp_solver(
            query=query["text"],
            solver_type=solver_type,
            select_solution=select_solution,
        )

        # check acc@0.25/0.5
        gt_bbox = gt_scene.get_instance_bbox(str(query["target_id"]))
        if TargetInfo.best_instance is not None:
            iou_best = TargetInfo.best_instance.bbox.iou(gt_bbox)
            iou_others = [
                inst.bbox.iou(gt_bbox) for inst in TargetInfo.candidate_instances
            ]

            result.eval_result["predicted_bbox"] = {
                "pmin": [float(x) for x in list(TargetInfo.best_instance.bbox.pmin)],
                "pmax": [float(x) for x in list(TargetInfo.best_instance.bbox.pmax)],
            }

            result.eval_result["anchor_bboxes"] = {
                name: {
                    "pmin": [float(x) for x in list(inst.bbox.pmin)],
                    "pmax": [float(x) for x in list(inst.bbox.pmax)],
                }
                if isinstance(inst, ObjInstance)
                else [
                    {
                        "pmin": [float(x) for x in list(y.bbox.pmin)],
                        "pmax": [float(x) for x in list(y.bbox.pmax)],
                    }
                    for y in inst
                ]
                for name, inst in TargetInfo.anchor_instances.items()
            }

            result.eval_result["csp_desc"] = TargetInfo.csp_desc
            result.llm_used = TargetInfo.llm_used

            if iou_best >= 0.5:
                result.acc05 = True
                result.eval_result["acc05"] = True
            if iou_best >= 0.25:
                result.acc025 = True
                result.eval_result["acc025"] = True

            if (result.acc05 or result.acc025) and verbose >= 2:
                print()
                print("*** SUCCESSFUL ***")
                print()

            if any(x >= 0.5 for x in iou_others):
                result.acc05_potential = True
                # if result.acc05 is False:
                #     result.eval_result["far"] = True
            if any(x >= 0.25 for x in iou_others):
                result.acc025_potential = True

            if (result.acc05_potential or result.acc025_potential) and verbose >= 2:
                print()
                print("*** COULD BE SUCCESSFUL ***")
                print()

        if verbose >= 2:
            # print(TargetInfo.label)
            # for inst in TargetInfo.instances:
            #     print(inst.inst_id, inst.label)
            print(f"==================> ^^^ Query {query_id} ^^^ <==================")
            print()
            print()
            print()

    except Exception as e:
        # raise e
        if verbose >= 2:
            print()
            print("============> Query Failed <============")
            print(query["program"])
            print("========================================")
            print()

        if verbose >= 1:
            print()
            print(f"QUERY {query_id} THROWS EXCEPTION!")
            print()
            print(os.linesep.join([s for s in query["program"].splitlines() if s]))
            print()
            traceback.print_exc(file=sys.stdout)
            print()

        result.acc025 = False
        result.acc025_potential = False
        result.acc05 = False
        result.acc05_potential = False

        result.eval_result = query.copy()
        result.eval_result["acc05"] = False
        result.eval_result["acc025"] = False

        result.error = True

    return result


def eval_single_nr3d(
    query_info: tuple[int, dict],
    mask3d_pred_path: str | None = None,
    maskcluster_pred_path: str | None = None,
    cache_root: str | None = None,
    solver_type: str = "default",
    select_solution: str = "min_dist",
    verbose: int = 0,
) -> EvalSingleResultNr3D:
    query_id, query = query_info
    result = EvalSingleResultNr3D()
    result.eval_result = query.copy()
    result.eval_result["success"] = False

    assert "view_dependent" in query
    assert "hard" in query
    result.is_hard = query["hard"]
    result.is_view_dependent = query["view_dependent"]

    scene_id = query["scene_id"]
    scene = ScanNetScene(
        f"./data/scans/{scene_id}",
        mask3d_pred_path=mask3d_pred_path,
        maskcluster_pred_path=maskcluster_pred_path,
        add_room_center=True,
        add_room_corners=True,
        cache_root=cache_root,
    )
    # scene.visualize()

    # used to get groundtruth bounding boxes
    gt_scene = ScanNetScene(
        f"./data/scans/{scene_id}",
        mask3d_pred_path=None,
        maskcluster_pred_path=None,
        add_room_center=False,
        add_room_corners=False,
        cache_root=cache_root,
    )

    if verbose >= 2:
        print(f"==================> vvv Query {query_id} vvv <==================")
        print(query["text"])
        print(query["scene_id"])
        print(query["target_id"])
        print(query["target_label"])
        print(query["program"])
        print()

    use_min_dist_heuristic = True
    query_text = query["text"].lower()
    if any(
        x in query_text
        for x in {"far", "across", "opposite", "away", "remote", "distant"}
    ):
        use_min_dist_heuristic = False

    try:
        # this is such a mess... it would be better to implement an interpreter manually,
        # which will provide better error messages and debugging support

        set_instance_map(scene.get_instance_map())
        set_room_center(scene.get_room_center())
        reset_csp_solver()
        TargetInfo.reset()

        exec(query["program"], get_eval_scope(use_type_check_funcs=False))
        run_csp_solver(
            query=query["text"],
            solver_type=solver_type,
            select_solution=select_solution,
        )

    except Exception as e:
        # raise e
        if verbose >= 2:
            print()
            print("============> Query Failed <============")
            print(query["program"])
            print("========================================")
            print()

        if verbose >= 1:
            print()
            print(f"QUERY {query_id} THROWS EXCEPTION!")
            print()
            print(
                os.linesep.join(
                    [s for s in query["program"].splitlines() if s.rstrip()]
                )
            )
            print()
            traceback.print_exc(file=sys.stdout)
            print()

        result.error = True

    if TargetInfo.best_instance is not None:
        if str(TargetInfo.best_instance.inst_id) == str(query["target_id"]):
            result.success = True
            result.eval_result["success"] = True
            if verbose >= 2:
                print()
                print("*** SUCCESSFUL ***")
                print()

        if str(TargetInfo.best_instance.inst_id) in set(
            str(inst.inst_id) for inst in TargetInfo.candidate_instances
        ):
            result.could_success = True
            if verbose >= 2:
                print()
                print("*** COULD BE SUCCESSFUL ***")
                print()

        result.eval_result["predicted_bbox"] = {
            "pmin": [float(x) for x in list(TargetInfo.best_instance.bbox.pmin)],
            "pmax": [float(x) for x in list(TargetInfo.best_instance.bbox.pmax)],
        }

        result.eval_result["anchor_bboxes"] = {
            name: {
                "pmin": [float(x) for x in list(inst.bbox.pmin)],
                "pmax": [float(x) for x in list(inst.bbox.pmax)],
            }
            if isinstance(inst, ObjInstance)
            else [
                {
                    "pmin": [float(x) for x in list(y.bbox.pmin)],
                    "pmax": [float(x) for x in list(y.bbox.pmax)],
                }
                for y in inst
            ]
            for name, inst in TargetInfo.anchor_instances.items()
        }

        result.eval_result["csp_desc"] = TargetInfo.csp_desc
        result.llm_used = TargetInfo.llm_used

    if verbose >= 2:
        # print(TargetInfo.label)
        # for inst in TargetInfo.instances:
        #     print(inst.inst_id, inst.label)
        print(f"==================> ^^^ Query {query_id} ^^^ <==================")
        print()
        print()
        print()

    assert isinstance(result, EvalSingleResultNr3D)
    return result


def main():
    # register predefined functions
    # import program_functions as _
    import program_functions_csp as _

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver", type=str, choices=["default", "non_csp"], default="default"
    )
    parser.add_argument(
        "--select-solution",
        type=str,
        choices=["min_dist", "max_dist", "random", "first"],
        default="min_dist",
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--seg", type=str, choices=["gt", "mask3d"], required=True)
    parser.add_argument("--num-threads", type=int, default=30)
    parser.add_argument("--num-queries", type=int)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--print-if-succeed", action="store_true")
    args = parser.parse_args()

    tags = []

    if args.seg == "gt":
        print("using gt segmentation")
        tags.append("gt")

    mask3d_pred = None
    maskcluster_pred = None

    if args.seg == "mask3d":
        print("using mask3d segmentation")
        mask3d_pred = "./data/eval_output/mask3d_val"
        tags.append("mask3d")

    if args.seg == "maskcluster":
        print("using maskclustering segmentation")
        maskcluster_pred = "./data/eval_output/maskcluster"
        tags.append("maskcluster")
        raise NotImplementedError()

    assert len(tags) == 1

    if args.dataset == "scanrefer":
        print("using scanrefer dataset")
        tags.append("scanrefer")

    if args.dataset == "nr3d":
        print("using nr3d dataset")
        tags.append("nr3d")

    if args.dataset == "custom":
        print("using custom dataset")
        tags.append("custom")

    if len(tags) < 2:
        assert args.dataset
        print(f"using custom dataset: {args.dataset}")
        tags.append(args.dataset)

    assert len(tags) == 2

    if args.experiment_name:
        print(f"using experiment name: {args.experiment_name}")
        tags.append(args.experiment_name)

    cache_root = "./data/instance_cache"

    eval_file_path = f"./output/eval_data_{'_'.join(tags)}.json"
    eval_results_file_path = f"./output/eval_results_{'_'.join(tags)}.json"

    print(f"loading eval file:          [{eval_file_path}].")
    print(f"will write results to file: [{eval_results_file_path}].")

    if not os.path.isfile(eval_file_path):
        print("eval file not found.")
        return

    print(f"using csp solver: {args.solver}.")

    with open(eval_file_path) as f:
        eval_data = json.load(f)

    if args.num_queries is not None:
        random.seed()
        random.shuffle(eval_data)
        eval_data = eval_data[0 : args.num_queries]

    num_threads = args.num_threads
    num_errors = 0
    num_llm_used = 0
    num_llm_correct = 0
    verbose = args.verbose

    print()

    if args.dataset != "nr3d":
        # accumulate some statistics
        num_acc05 = 0
        num_acc025 = 0
        num_acc05_potential = 0
        num_acc025_potential = 0
        num_evals = 0

        num_acc05_unique = 0
        num_acc025_unique = 0
        num_acc05_potential_unique = 0
        num_acc025_potential_unique = 0
        num_evals_unique = 0

        num_acc05_multiple = 0
        num_acc025_multiple = 0
        num_acc05_potential_multiple = 0
        num_acc025_potential_multiple = 0
        num_evals_multiple = 0

        good_query_indices = []

        # start evaluation
        eval_results = []
        eval_data = eval_data[:]

        eval_single_func = functools.partial(
            eval_single_scanrefer,
            mask3d_pred_path=mask3d_pred,
            maskcluster_pred_path=maskcluster_pred,
            cache_root=cache_root,
            solver_type=args.solver,
            select_solution=args.select_solution,
            verbose=verbose,
        )

        with Pool(num_threads) as pool:
            for result in tqdm(
                pool.imap_unordered(
                    eval_single_func,
                    zip(range(len(eval_data)), eval_data),
                ),
                total=len(eval_data),
            ):
                if args.print_if_succeed:
                    print()
                    print(f"query: {result.eval_result['text']}")
                    print(f"program:\n{result.eval_result['program']}")
                    print("success" if result.acc05 else "failure")
                    print()

                num_errors += int(result.error)
                if result.llm_used:
                    num_llm_used += 1
                    num_llm_correct += int(result.acc05)

                num_acc05 += int(result.acc05)
                num_acc025 += int(result.acc025)
                num_acc05_potential += int(result.acc05_potential)
                num_acc025_potential += int(result.acc025_potential)
                num_evals += 1

                if result.is_unique:
                    num_acc05_unique += int(result.acc05)
                    num_acc025_unique += int(result.acc025)
                    num_acc05_potential_unique += int(result.acc05_potential)
                    num_acc025_potential_unique += int(result.acc025_potential)
                    num_evals_unique += 1
                else:
                    num_acc05_multiple += int(result.acc05)
                    num_acc025_multiple += int(result.acc025)
                    num_acc05_potential_multiple += int(result.acc05_potential)
                    num_acc025_potential_multiple += int(result.acc025_potential)
                    num_evals_multiple += 1

                assert result.eval_result
                eval_results.append(result.eval_result)

                if result.acc05:
                    good_query_indices.append(result.eval_result["query_id"])

        print()
        print("========================= evalutation results =========================")
        print()

        # fmt: off
        print(f"acc@0.5:      {num_acc05 / num_evals:.4f} ({num_acc05} / {num_evals})")
        print(f"acc@0.25:     {num_acc025 / num_evals:.4f} ({num_acc025} / {num_evals})")
        print(f"acc@0.5  (?): {num_acc05_potential / num_evals:.4f} ({num_acc05_potential} / {num_evals})")
        print(f"acc@0.25 (?): {num_acc025_potential / num_evals:.4f} ({num_acc025_potential} / {num_evals})")

        if num_evals_unique > 0:
            print(f"acc@0.5  (u): {num_acc05_unique / num_evals_unique:.4f} ({num_acc05_unique} / {num_evals_unique})")
            print(f"acc@0.25 (u): {num_acc025_unique / num_evals_unique:.4f} ({num_acc025_unique} / {num_evals_unique})")

        if num_evals_multiple > 0:
            print(f"acc@0.5  (m): {num_acc05_multiple / num_evals_multiple:.4f} ({num_acc05_multiple} / {num_evals_multiple})")
            print(f"acc@0.25 (m): {num_acc025_multiple / num_evals_multiple:.4f} ({num_acc025_multiple} / {num_evals_multiple})")

        print(f"errors:       {num_errors}")
        print(f"llm:          {num_llm_correct} / {num_llm_used}")
        # fmt: on

        # print(sorted(good_query_indices))

        print()
        print("========================= evalutation results =========================")
        print()

    if args.dataset == "nr3d":
        num_overall = 0
        num_overall_success = 0

        num_easy = 0
        num_easy_success = 0

        num_hard = 0
        num_hard_success = 0

        num_view_dep = 0
        num_view_dep_success = 0

        num_view_indep = 0
        num_view_indep_success = 0

        # start evaluation
        eval_results = []
        eval_data = eval_data[:]

        eval_single_func = functools.partial(
            eval_single_nr3d,
            mask3d_pred_path=mask3d_pred,
            maskcluster_pred_path=maskcluster_pred,
            cache_root=cache_root,
            solver_type=args.solver,
            select_solution=args.select_solution,
            verbose=verbose,
        )

        with Pool(num_threads) as pool:
            for result in tqdm(
                pool.imap_unordered(
                    eval_single_func,
                    zip(range(len(eval_data)), eval_data),
                ),
                total=len(eval_data),
            ):
                if args.print_if_succeed:
                    print()
                    print(f"query: {result.eval_result['text']}")
                    print("success" if result.acc05 else "failure")
                    print()

                num_errors += int(result.error)
                if result.llm_used:
                    num_llm_used += 1
                    num_llm_correct += int(result.acc05)

                num_overall += 1
                num_overall_success += int(result.success)

                if result.is_hard:
                    num_hard += 1
                    num_hard_success += int(result.success)
                else:
                    num_easy += 1
                    num_easy_success += int(result.success)

                if result.is_view_dependent:
                    num_view_dep += 1
                    num_view_dep_success += int(result.success)
                else:
                    num_view_indep += 1
                    num_view_indep_success += int(result.success)

                assert result.eval_result
                eval_results.append(result.eval_result)

        print()
        print("========================= evalutation results =========================")
        print()

        # fmt: off
        print(f"overall:    {num_overall_success / num_overall:.4f} ({num_overall_success} / {num_overall})")
        print(f"easy:       {num_easy_success / num_easy:.4f} ({num_easy_success} / {num_easy})")
        print(f"hard:       {num_hard_success / num_hard:.4f} ({num_hard_success} / {num_hard})")
        print(f"view dep:   {num_view_dep_success / num_view_dep:.4f} ({num_view_dep_success} / {num_view_dep})")
        print(f"view indep: {num_view_indep_success / num_view_indep:.4f} ({num_view_indep_success} / {num_view_indep})")
        print(f"errors:     {num_errors}")
        print(f"llm:        {num_llm_correct} / {num_llm_used}")
        # fmt: on

        print()
        print("========================= evalutation results =========================")
        print()

    with open(eval_results_file_path, "w") as f:
        json.dump(eval_results, f)


if __name__ == "__main__":
    # random.seed(0)
    main()
