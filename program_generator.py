#
# generate pseudo python programs from queries
#

import argparse
import asyncio
import json
import os
import random
import re
import sys
import textwrap
import time
import traceback

import pandas as pd
import requests
import tqdm
import tqdm.asyncio
from openai import AsyncOpenAI, OpenAI

from program_validator import validate_program
from scannet_utils import ScanNetScene
from scope_env import (
    get_eval_scope,
    get_predef_func_sigs,
    set_relevant_obj_map,
)
from score_funcs import SCORE_FUNCTIONS

# set to None to use all queries
FIRST_QUERY = 0
LAST_QUERY = None
REQUEST_GROUP_SIZE = 50
USE_CSP_PROGRAM = True
PHASE_3_NUM_RETRIES = 10
PHASE_1_NUM_RETRIES = 10
RANDOM_SEED = 31
MODEL_TEMPERATURE = 0.3


def load_scanrefer_queries():
    scanrefer_queries = []
    # with open("./data/scanrefer/ScanRefer_filtered_train.json") as f:
    with open("./data/scanrefer/ScanRefer_filtered_val.json") as f:
        j_data = json.load(f)
        random.shuffle(j_data)
        for j_query in j_data[FIRST_QUERY:LAST_QUERY]:
            query = {}
            query["text"] = j_query["description"].lower().strip()
            query["scene_id"] = j_query["scene_id"]
            query["target_id"] = j_query["object_id"]
            query["target_label"] = j_query["object_name"]
            scanrefer_queries.append(query)
    return scanrefer_queries


def load_custom_queries(query_path: str = ""):
    queries = []

    if not query_path:
        query_path = "./data/custom_queries.json"
    assert os.path.isfile(query_path), query_path

    with open(query_path) as f:
        j_data = json.load(f)
        random.shuffle(j_data)
        for j_query in j_data[FIRST_QUERY:LAST_QUERY]:
            query = {}
            query["text"] = j_query["description"].lower().strip()
            query["scene_id"] = j_query["scene_id"]
            query["target_id"] = j_query["object_id"]
            query["target_label"] = j_query["object_name"]
            queries.append(query)
    return queries


def load_nr3d_queries():
    def is_hard(row):
        """
        Split into scene_id, instance_label, # objects, target object id,
        distractors object id.

        :param s: the stimulus string

        modified from ReferIt3D: https://github.com/referit3d/referit3d/tree/eccv
        """
        s = row.stimulus_id
        if len(s.split("-", maxsplit=4)) == 4:
            scene_id, instance_label, n_objects, target_id = s.split("-", maxsplit=4)
            distractors_ids = ""
        else:
            scene_id, instance_label, n_objects, target_id, distractors_ids = s.split(
                "-", maxsplit=4
            )

        instance_label = instance_label.replace("_", " ")
        n_objects = int(n_objects)
        target_id = int(target_id)
        distractors_ids = [int(i) for i in distractors_ids.split("-") if i != ""]
        assert len(distractors_ids) == n_objects - 1

        # return scene_id, instance_label, n_objects, target_id, distractors_ids
        return n_objects > 2

    def is_view_dep(row):
        """
        :param df: pandas dataframe with "tokens" columns
        :return: a boolean mask

        modified from ReferIt3D: https://github.com/referit3d/referit3d/tree/eccv
        """
        target_words = {
            "front",
            "behind",
            "back",
            "right",
            "left",
            "facing",
            "leftmost",
            "rightmost",
            "looking",
            "across",
        }
        return len(set(eval(row.tokens)).intersection(target_words)) > 0

    with open("./data/scannetv2_val.txt") as f:
        scannet_val_scenes = [line.strip() for line in f if line.strip()]
    assert len(scannet_val_scenes) == 312
    scannet_val_scenes = set(scannet_val_scenes)

    nr3d_queries = []
    dataset = pd.read_csv("./data/nr3d.csv")

    dataset = dataset[dataset.scan_id.isin(scannet_val_scenes)]
    dataset = dataset[dataset.mentions_target_class == True]
    dataset = dataset[dataset.correct_guess == True]

    for i, row in dataset.iloc[FIRST_QUERY:LAST_QUERY].iterrows():
        assert row.mentions_target_class
        assert row.correct_guess
        assert row.scan_id in scannet_val_scenes

        query = {}
        query["text"] = row.utterance.lower().strip()
        query["scene_id"] = row.scan_id
        query["target_id"] = str(row.target_id)
        query["target_label"] = str(row.instance_type)
        query["view_dependent"] = is_view_dep(row)
        query["hard"] = is_hard(row)
        nr3d_queries.append(query)

    return nr3d_queries


def load_scene_objects(
    scene_ids: list[str],
    mask3d_pred_path: str | None = None,
    maskcluster_pred_path: str | None = None,
    cache_root: str | None = None,
):
    assert not mask3d_pred_path or not maskcluster_pred_path
    scene_obj_labels = {}

    for scene_id in tqdm.tqdm(scene_ids, desc="loading labels", leave=False):
        if scene_id not in scene_obj_labels:
            scene = ScanNetScene(
                scene_path=f"./data/scans/{scene_id}",
                mask3d_pred_path=mask3d_pred_path,
                maskcluster_pred_path=maskcluster_pred_path,
                add_room_center=True,
                add_room_corners=True,
                cache_root=cache_root,
            )
            scene_obj_labels[scene_id] = sorted(list(scene.get_instance_map().keys()))

    # print(scene_obj_labels)
    return scene_obj_labels


def load_prompt(file_path, num_queries: int | None = None):
    with open(file_path) as f:
        msg_started = False
        msg_role = None
        msg = None
        prompt_dialog = []
        for line in f:
            if not msg_started:
                line = line.strip()
                if line in {"<[SYSTEM]>", "<[USER]>", "<[ASSISTANT]>"}:
                    msg_started = True
                    msg_role = line[2:-2].lower().strip()
                    msg = []
            else:
                assert msg_role in {"system", "user", "assistant"}
                if line.strip() in {"<[SYSTEM]>", "<[USER]>", "<[ASSISTANT]>"}:
                    assert msg
                    prompt_dialog.append(
                        {"role": msg_role, "content": "\n".join(msg).strip()}
                    )
                    msg_role = line.strip()[2:-2].lower().strip()
                    msg = []
                else:
                    msg.append(line.rstrip())
        if msg:
            prompt_dialog.append({"role": msg_role, "content": "\n".join(msg).strip()})

    if num_queries is not None:
        prompt_dialog = prompt_dialog[: 1 + 2 * num_queries]

    # validate the dialog
    assert len(prompt_dialog) > 0
    assert prompt_dialog[0]["role"] == "system"
    for i, turn in enumerate(prompt_dialog[1:]):
        if i % 2 == 0:
            assert turn["role"] == "user"
        else:
            assert turn["role"] == "assistant"
    assert prompt_dialog[-1]["role"] == "assistant"

    return prompt_dialog


def print_dialog_full(dialog, header="DIALOG"):
    assert len(dialog) > 0
    print(f"==> ↓↓↓  {header} ↓↓↓  <==")
    print()
    for turn in dialog:
        print(f"<[{turn['role']}]>")
        print(turn["content"])
        print()
    print(f"==> ↑↑↑  {header} ↑↑↑  <==")
    print()


async def generate_program_single(
    query_text: str,
    scene_objs: list[str],
    client: AsyncOpenAI,
    model_name: str,
    prompt_dialog_filter_obj: list[dict[str, str]],
    prompt_dialog_gen_prog: list[dict[str, str]],
    query_id: int,
    verbose: int = 0,
    use_first_phase: bool = True,
):
    def print_dialog(dialog):
        assert len(dialog) > 0
        print()
        print(f"==> ↓↓↓  QUERY {query_id} ↓↓↓  <==")
        print()
        for turn in dialog[-2:]:
            print(turn["content"])
            print()
        print(f"==> ↑↑↑  QUERY {query_id} ↑↑↑  <==")
        print()
        input()

    # dialog history
    results = []

    async def get_completion(prompt, seed):
        nonlocal results
        while True:
            try:
                completion = await client.chat.completions.create(
                    model=model_name,
                    messages=prompt,
                    # max_tokens=2048,
                    # temperature=MODEL_TEMPERATURE,
                    # seed=seed,
                )
            except Exception:
                # print("\nretry...\n")
                time.sleep(2.0)
            else:
                break

        # print(completion.model)
        answer = completion.choices[0].message.content.strip()
        results.append({"input": prompt[-1]["content"], "output": answer})
        return answer

    scene_objs = sorted(list(scene_objs))

    if use_first_phase:
        labels_str = "\n".join([f"[{i}] {label}" for i, label in enumerate(scene_objs)])

        prompt1 = prompt_dialog_filter_obj.copy()
        invalid_label_err_msgs = []

        phase_1_tries = 0
        phase_1_success = False
        for i_phase1 in range(PHASE_1_NUM_RETRIES):
            if i_phase1 == 0:
                prompt1.append(
                    {
                        "role": "user",
                        "content": f"QUERY:\n{query_text}\n\nOBJECTS IN 3D SCENE:\n{labels_str}",
                    }
                )
            else:
                assert len(invalid_label_err_msgs) > 0
                error_msgs = "\n".join(invalid_label_err_msgs)
                correction_prompt = (
                    "Your output contains invalid label IDs or labels. The error messages are:\n"
                    f"{error_msgs}\n\n"
                    "Please reason about why these errors happen and fix them. You should strictly follow the format of your previous answers. Please DO NOT repeat the query and objects in the scene. You should only output your reasoning and the corrected version of relevant objects.\n"
                    "Beware that you should only use the objects given to you in the list. Please DO NOT invent new objects or extract objects from the query.\n"
                )

                prompt1.append(
                    {
                        "role": "user",
                        "content": textwrap.dedent(correction_prompt).strip(),
                    }
                )

            # # custom llm server...
            # llm_api = "http://localhost:8000/chat/quiet"
            # responses = requests.post(llm_api, json={"data": prompt_batch})
            # answers: list[str] = [x["output"] for x in responses.json()]

            if verbose >= 1:
                print(f"query [{query_id}] starts phase 1 ({i_phase1})")

            answer1 = await get_completion(prompt1, seed=0)
            phase_1_tries += 1

            if verbose >= 2:
                dialog = prompt1.copy()
                dialog.append({"role": "assistant", "content": answer1})
                print_dialog(dialog)

            # parse relevant objects from the llm response
            relevant_obj_labels = []
            for line in answer1.strip().split("\n"):
                line = line.strip()
                if line and (match := re.match(r"@obj\s*\[(\d+)\]\s*(.+)$", line)):
                    assert len(match.groups()) == 2
                    relevant_obj_labels.append((match[1].strip(), match[2].strip()))

            # validate output relevant object labels
            invalid_label_err_msgs = []
            for label_id, label in relevant_obj_labels:
                if not label_id.isdigit():
                    invalid_label_err_msgs.append(
                        f"error: label_id [{label_id}] is not an integer!"
                    )
                    continue

                label_id = int(label_id)
                if label_id >= len(scene_objs) or label_id < 0:
                    invalid_label_err_msgs.append(
                        f"error: label_id [{label_id}] has invalid value!"
                    )
                elif scene_objs[label_id] != label:
                    invalid_label_err_msgs.append(
                        f"error: label_id [{label_id}] and label {label} does not match!"
                    )

            if len(invalid_label_err_msgs) == 0:
                if len(relevant_obj_labels) > 0:
                    phase_1_success = True
                    break
                else:
                    print()
                    print("===>>EMPTY RELEVANT OBJECTS <<===")
                    print()
                    print(results[-1]["input"])
                    print()
                    print("    >>>")
                    print()
                    print(results[-1]["output"])
                    print()
                    print("===>>EMPTY RELEVANT OBJECTS <<===")
                    print()
                    invalid_label_err_msgs.append(
                        "error: you did not output any relevant objects!"
                    )

            prompt1.append({"role": "assistant", "content": answer1})

        relevant_obj_labels = [label for label_id, label in relevant_obj_labels]
    else:
        phase_1_tries = 0
        phase_1_success = True
        relevant_obj_labels = scene_objs.copy()

    assert len(relevant_obj_labels) > 0

    relevant_obj_labels = sorted(list(set(relevant_obj_labels)))
    relevant_object_map = {i: label for i, label in enumerate(relevant_obj_labels)}

    labels_str = "\n".join(
        [f"[{i}] {label}" for i, label in enumerate(relevant_obj_labels)]
    )

    prompt2 = prompt_dialog_gen_prog.copy()
    prompt2.append(
        {
            "role": "user",
            "content": f"QUERY:\n{query_text}\n\nRELEVANT OBJECT LABELS:\n{labels_str}",
        }
    )

    # responses = requests.post(llm_api, json={"data": prompt_batch})
    # answers: list[str] = [x["output"] for x in responses.json()]

    if verbose >= 1:
        print(f"query [{query_id}] starts phase 2")
    answer2 = await get_completion(prompt2, seed=1)

    if verbose >= 2:
        dialog = prompt2.copy()
        dialog.append({"role": "assistant", "content": answer2})
        print_dialog(dialog)

    # # validate generated program
    # obj_labels_set = set(scene_objs)
    # good = True
    # for rel in json.loads(answer)["constraints"]:
    #     if rel["target"] not in obj_labels_set or rel["anchor"] not in obj_labels_set:
    #         good = False
    #         break

    # refine program until all errors are corrected
    prompt3 = prompt2.copy()
    phase_3_success = False
    phase_3_tries = 0
    i_phase3 = 0
    generated_program = None
    # first_error = True

    from program_functions_csp import check_target, reset_target

    while True:
        error_msg = None
        answer2 = validate_program(answer2)

        # print()
        # print()
        # print("================= Validated Program =================")
        # print()
        # print(answer2)
        # print()
        # print("=====================================================")
        # print()
        # print()

        try:
            reset_target()
            set_relevant_obj_map(relevant_object_map)
            exec(answer2, get_eval_scope(use_type_check_funcs=True))

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            stack = traceback.extract_tb(exc_tb)
            # find the stack frame of generated program
            err_lineno = None
            for i in range(1, len(stack) + 1):
                # print(stack[-i][0])
                if stack[-i].filename.strip() == "<string>":
                    err_lineno = stack[-i].lineno
            if err_lineno is not None:
                err_line = answer2.strip().split("\n")[err_lineno - 1]
            else:
                err_lineno = stack[-1].lineno
                err_line = stack[-1].line
            error_msg = [f"Error at line {err_lineno}: {err_line}", repr(e)]
            # print("\n\n===> Exception! <===\n\n")

        if error_msg is None and not check_target():
            error_msg = ["Error: ", "you did not specify an target object!"]

        # error_msg = None  # disable semantic check
        if error_msg is None:
            phase_3_success = True
            generated_program = answer2
            break

        # retry phase 3
        i_phase3 += 1
        if i_phase3 >= PHASE_3_NUM_RETRIES:  # adjust maximum number of retries
            results.append({"input": "", "output": "\n".join(error_msg)})
            break

        # if first_error:
        #     first_error = False
        # else:
        #     prompt3.pop()
        #     prompt3.pop()

        prompt3.append({"role": "assistant", "content": answer2})

        correction_prompt = f"""
            Your output contains error. The error message is:
            {error_msg[0]}
            {error_msg[1]}
            
            Please reason about why this error occurs and regenerate a correct program without any errors. Please be as concise as possible and do not give too verbose reasoning. Please make sure your output is a valid python program, i.e. comment all your explanations.
            
            If you think you cannot correct some errors, you can simplify the program by ignoring that function and removing it.
            
            Please DO NOT follow markdown convention. PLEASE DO NOT ENCLOSE PYTHON CODE WITH ```!
            """
        prompt3.append(
            {"role": "user", "content": textwrap.dedent(correction_prompt).strip()}
        )

        if verbose >= 1:
            print(f"query [{query_id}] starts phase 3 ({i_phase3})")

        answer3 = await get_completion(prompt3, seed=2)
        phase_3_tries += 1

        if verbose >= 2:
            dialog = prompt3.copy()
            dialog.append({"role": "assistant", "content": answer3})
            print_dialog(dialog)

        answer2 = answer3

        # maybe only retry once...
        # break

    if verbose >= 1 and (not phase_3_success or not phase_1_success):
        print(f"QUERY [{query_id}] FAILED <=== [!]")

    if phase_1_success and phase_3_success:
        assert generated_program is not None

    return {
        "history": results,
        "phase_1_tries": phase_1_tries,
        "phase_3_tries": phase_3_tries,
        "query_id": query_id,
        "query_text": query_text,
        "phase_1_success": phase_1_success,
        "phase_3_success": phase_3_success,
        "success": phase_1_success and phase_3_success,
        "program": generated_program,
    }


def print_result(query_id, dialog, success=None):
    assert len(dialog) >= 1

    print()
    if success is None:
        print(f"==> ↓↓↓  QUERY {query_id} ↓↓↓  <==")
    elif success is True:
        print(f"==> ↓↓↓  QUERY {query_id} SUCCESSFUL ↓↓↓  <==")
    else:
        print(f"==> ↓↓↓  QUERY {query_id} FAILED ↓↓↓  <==")
    print()

    for turn in dialog:
        print(f"{turn['input']}\n\n>>>>>>\n\n{turn['output']}")
        print()
        print(" " * 10 + "*****")
        print()

    if success is None:
        print(f"==> ↑↑↑  QUERY {query_id} ↑↑↑  <==")
    elif success is True:
        print(f"==> ↑↑↑  QUERY {query_id} SUCCESSFUL ↑↑↑  <==")
    else:
        print(f"==> ↑↑↑  QUERY {query_id} FAILED ↑↑↑  <==")
    print()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-query", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--use-first-phase", action="store_true")
    parser.add_argument("--llm", type=str, choices=["openai", "local"], default="local")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--num-prompt-example", type=str)
    parser.add_argument("--no-minmax", action="store_true")
    parser.add_argument("--print-prompt-only", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mask3d", action="store_true")
    # group.add_argument("--maskcluster", action="store_true")
    args = parser.parse_args()

    #
    # register predefined functions
    #

    def _register_csp_funcs():
        # disable some csp functions
        import csp_func_control as cfc

        cfc.DISABLE_MINMAX = args.no_minmax
        # cfc.DISABLE_NEGATION = (
        #     args.no_counting_negation or args.no_counting_negation_minmax
        # )

        # register predefined functions
        import program_functions_csp as _

    _register_csp_funcs()

    #
    # load prompts
    #

    prompt_dialog_filter_obj = load_prompt("./prompts/filter_relevant_objects.txt")

    header_text = "PHASE 2 PROMPT (CSP)"
    # if args.no_counting:
    #     prompt_dialog_gen_prog = load_prompt(
    #         "./prompts/generate_program_csp_no_counting.txt"
    #     )
    #     header_text += " (no counting)"
    # elif args.no_counting_negation:
    #     prompt_dialog_gen_prog = load_prompt(
    #         "./prompts/generate_program_csp_no_counting_negation.txt"
    #     )
    #     header_text += " (no counting/negation)"
    # elif args.no_counting_negation_minmax:
    #     prompt_dialog_gen_prog = load_prompt(
    #         "./prompts/generate_program_csp_no_counting_negation_minmax.txt"
    #     )
    #     header_text += " (no counting/negation/minmax)"
    # else:
    if args.no_minmax:
        prompt_dialog_gen_prog = load_prompt(
            "./prompts/generate_program_csp_no_minmax.txt",
            args.num_prompt_example,
        )
        header_text += " (no min/max)"
    else:
        prompt_dialog_gen_prog = load_prompt(
            "./prompts/generate_program_csp.txt",
            args.num_prompt_example,
        )

    reg_funcs = get_predef_func_sigs()
    reg_funcs_str = "\n".join(
        [
            f"{fname}{fsig} # {fdoc}" if fdoc else f"{fname}{fsig}"
            for fname, fsig, fdoc in reg_funcs
        ]
    )

    reg_score_funcs_str = "\n".join(SCORE_FUNCTIONS.keys())

    assert prompt_dialog_gen_prog[0]["role"] == "system"
    raw_sys_prompt: str = prompt_dialog_gen_prog[0]["content"]

    prompt_dialog_gen_prog[0]["content"] = raw_sys_prompt.replace(
        "<[REGISTERED_FUNCTIONS_PLACEHOLDER]>",
        reg_funcs_str,
    ).replace(
        "<[REGISTERED_SCORE_FUNCTIONS_PLACEHOLDER]>",
        reg_score_funcs_str,
    )

    print_dialog_full(prompt_dialog_gen_prog, header=header_text)
    if args.print_prompt_only:
        return

    # input()

    #
    # parse args
    #

    if args.random_seed is not None:
        print()
        print(f"random seed: {args.random_seed}")
        print()
        random.seed(args.random_seed)
    else:
        print()
        print("random seed: current time")
        print()
        random.seed()

    if args.group_size is not None:
        global REQUEST_GROUP_SIZE
        REQUEST_GROUP_SIZE = args.group_size

    # select segmentation method

    global LAST_QUERY
    if args.num_query is not None:
        LAST_QUERY = args.num_query
        print()
        print(f"using first {LAST_QUERY} querys.")
        print()
    else:
        LAST_QUERY = None
        print()
        print("using all queries")
        print()

    mask3d_pred = "./data/eval_output/mask3d_val" if args.mask3d else None
    maskcluster_pred = "./data/eval_output/maskcluster" if args.maskcluster else None

    cache_root = "./data/instance_cache"

    generation_tags = []

    if args.mask3d:
        print()
        print("using mask3d segmentations.")
        print()
        generation_tags.append("mask3d")

    if args.maskcluster:
        print()
        print("using maskcluster segmentations.")
        print()
        generation_tags.append("maskcluster")

    if mask3d_pred is None and maskcluster_pred is None:
        print()
        print("using gt segmentations.")
        print()
        generation_tags.append("gt")

    if args.dataset == "scanrefer":
        print()
        print("loading scanrefer dataset.")
        print()
        all_queries = load_scanrefer_queries()
        generation_tags.append("scanrefer")

    if args.dataset == "nr3d":
        print()
        print("loading nr3d dataset.")
        print()
        all_queries = load_nr3d_queries()
        generation_tags.append("nr3d")

    if args.dataset == "custom":
        print()
        print("loading custom queries.")
        print()
        all_queries = load_custom_queries()
        generation_tags.append("custom")

    if len(generation_tags) < 2:
        assert args.dataset
        print()
        print(f"loading non-predefined queries: {args.dataset}")
        print()
        all_queries = load_custom_queries(f"./data/{args.dataset}_queries.json")
        generation_tags.append(args.dataset)

    if "all_queries" not in locals() or len(all_queries) == 0:
        print()
        print("no queries are loaded. exit.")
        print()
        return

    if args.experiment_name:
        generation_tags.append(args.experiment_name)
        print()
        print(f"using experiment name: {args.experiment_name}")
        print()

    assert len(generation_tags) in {2, 3}
    eval_file_path = f"./output/eval_data_{'_'.join(generation_tags)}.json"
    print()
    print(f"writing to [{eval_file_path}].")
    print()

    group_size = REQUEST_GROUP_SIZE
    group_count = len(all_queries) // group_size
    if len(all_queries) % group_size != 0:
        group_count += 1

    # or use gpt (need to add rate control)
    if args.llm == "local":
        openai_client = AsyncOpenAI(
            api_key="db72ad53ea0db1354d46405703546670",
            base_url="http://127.0.0.1:2242/v1",
            timeout=3600.0,
        )
        # model_name = "qwen2-72b-instruct-exl"
        # model_name = "llama-3.1-70b-instruct-exl2"
        # model_name = "mistral-large-instruct-2407-123b-exl2"
        model_name = "mistral-large-instruct-2407-awq"
        model_name = "/cvhci/temp/qyuan/" + model_name

        print()
        print("using local llm.")
        print()

    elif args.llm == "openai":
        openai_client = AsyncOpenAI(
            api_key="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",
        )
        model_name = "gpt-4o"

        print()
        print("using openai server.")
        input("are you sure?")
        print()

    else:
        raise SystemError()

    # start generation

    eval_data = []
    num_success = 0
    num_phase_1_retry = 0
    num_phase_1_failed = 0
    num_phase_3_retry = 0
    num_phase_3_failed = 0

    for i_group in tqdm.tqdm(range(group_count), "total progress"):
        # get queries for the current group
        queries = all_queries[
            i_group * group_size : min(len(all_queries), (i_group + 1) * group_size)
        ]

        scene_objs_map = load_scene_objects(
            scene_ids=[q["scene_id"] for q in queries],
            mask3d_pred_path=mask3d_pred,
            maskcluster_pred_path=maskcluster_pred,
            cache_root=cache_root,
        )

        tasks = [
            generate_program_single(
                query_text=q["text"],
                scene_objs=scene_objs_map[q["scene_id"]],
                client=openai_client,
                model_name=model_name,
                prompt_dialog_filter_obj=prompt_dialog_filter_obj,
                prompt_dialog_gen_prog=prompt_dialog_gen_prog,
                query_id=i_group * group_size + i,
                verbose=0,
                use_first_phase=args.use_first_phase,
            )
            for i, q in enumerate(queries)
        ]

        sequential = False
        if not sequential:
            for result in await tqdm.asyncio.tqdm.gather(
                *tasks, desc="group progress", leave=False
            ):
                print_result(
                    query_id=result["query_id"],
                    dialog=result["history"],
                    success=result["success"],
                )

                if result["success"]:
                    num_success += 1
                if not result["phase_1_success"]:
                    num_phase_1_failed += 1
                if not result["phase_3_success"]:
                    num_phase_3_failed += 1
                if result["phase_1_tries"] > 1:
                    num_phase_1_retry += 1
                if result["phase_3_tries"] > 1:
                    num_phase_3_retry += 1

                # export generated programs for evaluation
                if result["success"]:
                    query = all_queries[result["query_id"]].copy()
                    query["program"] = result["program"]
                    eval_data.append(query)

        else:
            for task in tasks:
                result = await task
                print_result(
                    query_id=result["query_id"],
                    dialog=result["history"],
                    success=result["success"],
                )
                input()

    print()
    print(f"writing generation results to: {eval_file_path}")
    print()
    with open(eval_file_path, "w") as f:
        f.write(json.dumps(eval_data))

    print()
    print()
    print()
    print("======> SUMMARY <======")
    print(f"successful queries: {num_success} / {len(all_queries)}")
    print(f"phase 1 [retry: {num_phase_1_retry}] [failure: {num_phase_1_failed}]")
    print(f"phase 3 [retry: {num_phase_3_retry}] [failure: {num_phase_3_failed}]")
    print()


if __name__ == "__main__":
    asyncio.run(main())
