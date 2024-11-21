import functools
import random
import re
import sys
import textwrap
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Generator

import numpy as np
from openai import OpenAI
from typing_extensions import Self

from scannet_utils import ObjInstance
from scope_env import GlobalState, set_target_info

ID_COUNTER = 0


def reset_var_counter():
    global ID_COUNTER
    ID_COUNTER = 0


class CSPVar:
    """a CSP variable that represents a single instance in the scene"""

    def __init__(self, labels: list[str] = []):
        global ID_COUNTER
        ID_COUNTER += 1

        self.labels = labels
        self.obj_id = str(ID_COUNTER)
        self.is_target = False  # only CSPVar can be a target
        self.negative = False

    def __hash__(self) -> int:
        return hash(self.obj_id)

    def __eq__(self, other: Self) -> bool:
        return self.obj_id == other.obj_id

    def set_as_target(self):
        self.is_target = True

    def get_identifier(self) -> str:
        return f"{self.obj_id}-{'-'.join(self.labels)}".replace(" ", "_")


Solution = dict[CSPVar, ObjInstance]


def solution_to_str(sol_dict: Solution) -> str:
    text = []
    for csp_var, inst in sol_dict.items():
        text.append(f"[{csp_var.get_identifier()} -> {inst.inst_id}: {inst.label}]")
    return "{" + " ".join(text) + "}"


def load_prompt(file_path):
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

    # validate the dialog
    assert len(prompt_dialog) > 0
    assert prompt_dialog[0]["role"] == "system"
    for i, turn in enumerate(prompt_dialog[1:]):
        if i % 2 == 0:
            assert turn["role"] == "user"
        else:
            assert turn["role"] == "assistant"
    assert prompt_dialog[-1]["role"] in {"assistant", "system"}

    return prompt_dialog


MODEL_NAME = "mistral-large-instruct-2407-awq"
MODEL_NAME = "/cvhci/temp/qyuan/" + MODEL_NAME

PROMPT = load_prompt("./prompts/select_solution.txt")


def select_best_solution(
    query: str,
    csp_desc: str,
    valid_solutions: list[Solution],
) -> Solution:
    # def get_completion(prompt):
    #     assert len(prompt) > 0
    #     assert prompt[-1]["role"] == "user"

    #     openai_client = OpenAI(
    #         api_key="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",
    #         # base_url="http://127.0.0.1:2242/v1",
    #         timeout=3600.0,
    #     )

    #     while True:
    #         try:
    #             completion = openai_client.chat.completions.create(
    #                 # model=MODEL_NAME,
    #                 model="gpt-4o-mini",
    #                 messages=prompt,
    #                 # max_tokens=2048,
    #                 # temperature=0.7,
    #             )

    #             # print(completion.model)
    #             answer = completion.choices[0].message.content.strip()

    #         except Exception:
    #             # print("retry...")
    #             time.sleep(2.0)
    #         else:
    #             break

    #     return answer

    def get_completion(prompt):
        assert len(prompt) > 0
        assert prompt[-1]["role"] == "user"

        openai_client = OpenAI(
            api_key="db72ad53ea0db1354d46405703546670",
            base_url="http://127.0.0.1:2242/v1",
            timeout=3600.0,
        )

        completion = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=prompt,
            # max_tokens=2048,
            # temperature=0.7,
        )

        # print(completion.model)
        answer = completion.choices[0].message.content.strip()
        return answer

    assert query
    assert csp_desc
    assert 2 <= len(valid_solutions) <= 4

    objects_str = []
    solutions_str = []

    for csp_var in valid_solutions[0].keys():
        csp_var_id = f"{'_'.join(csp_var.labels).replace(' ', '_')}_{csp_var.obj_id}"
        objects_str.append(csp_var_id)

    objects_str = ", ".join(objects_str)

    for i, sol_dict in enumerate(valid_solutions):
        solutions_str.append(f"solution {i}")
        for csp_var, inst in sol_dict.items():
            csp_var_id = (
                f"{'_'.join(csp_var.labels).replace(' ', '_')}_{csp_var.obj_id}"
            )
            solutions_str.append(
                f"{csp_var_id}: "
                f"center=[{inst.bbox.center[0]:.3f}, {inst.bbox.center[1]:.3f}, {inst.bbox.center[2]:.3f}]; "
                f"size=[{inst.bbox.size[0]:.3f}, {inst.bbox.size[1]:.3f}, {inst.bbox.size[2]:.3f}]"
            )
        solutions_str.append("")

    solutions_str = "\n".join(solutions_str)

    user_prompt = (
        f"@QUERY\n{query}\n\n" f"@OBJECTS\n{objects_str}\n\n@SOLUTIONS\n{solutions_str}"
    )

    prompt = PROMPT.copy()
    prompt.append({"role": "user", "content": user_prompt.strip()})

    for i in range(5):
        response = get_completion(prompt)

        # print()
        # print("=" * 40)
        # print()
        # print(user_prompt)
        # print()
        # print("=" * 40)
        # print()
        # print(response)
        # print()
        # print("=" * 40)
        # print()

        matches = re.findall(
            r"^@BEGIN\{ANSWER\}\ncorrect solution index: \[(\d+)\]\n@END\{ANSWER\}",
            response,
            re.MULTILINE,
        )
        if len(matches) != 1:
            prompt.append({"role": "assistant", "content": response})
            prompt.append(
                {
                    "role": "user",
                    "content": (
                        "Your output is valid. You should include\n"
                        "@BEGIN{ANSWER}\ncorrect solution index: [index]\n@END{ANSWER}\n"
                        "exactly once!"
                    ),
                }
            )
            continue

        assert len(matches) == 1

        # print()
        # print(matches[0])
        # print()

        break

    return valid_solutions[int(matches[0])]


def print_solution(sol_dict: Solution):
    print()
    print("======= vvv SOLUTION vvv =======")
    for csp_var, cand_insts in sol_dict.items():
        print(
            f"{type(csp_var)} {csp_var.obj_id} {csp_var.labels} => "
            f"{[(inst.inst_id, inst.label) for inst in cand_insts]}"
        )
    print("======= ^^^ SOLUTION ^^^ =======")
    print()


class CSPConstraint(ABC):
    """
    base class for all constraints
    """

    def __init__(self):
        global ID_COUNTER
        ID_COUNTER += 1

        # a unique constraint id
        # self.con_id = str(uuid.uuid4())
        self.con_id = "con-" + str(ID_COUNTER)

        # the function name used in the generated program
        self.apparent_name: str = "(not set)"

    def __hash__(self) -> int:
        return hash(self.con_id)

    def __eq__(self, other: Self) -> bool:
        return self.con_id == other.con_id

    def set_apparent_name(self, name: str):
        self.apparent_name = name

    @abstractmethod
    def get_desc_str(self) -> str:
        """return a string describing the variables involved in this constraint"""

    @abstractmethod
    def get_target_var(self) -> CSPVar:
        """return the target variable of this constraint"""

    @abstractmethod
    def get_vars(self) -> set[CSPVar]:
        """return all variables invovled in this constraint"""

    @abstractmethod
    def check_solution(self, solution_dict: Solution) -> bool:
        """check if the given solution satisfies this constraint"""


def solution_generator(
    csp_vars: list[CSPVar],
) -> Generator[Solution, None, None]:
    """generate all possible combinations of assignments for free variables"""
    assert all(isinstance(x, CSPVar) for x in csp_vars)
    csp_vars.sort(key=lambda x: x.get_identifier())

    solution_dict: Solution = {}
    used_instances: set[ObjInstance] = set()

    def gen_func(i_var: int) -> Generator[Solution, None, None]:
        nonlocal solution_dict

        if i_var >= len(csp_vars):
            yield solution_dict.copy()
            return

        cur_var = csp_vars[i_var]
        assert not cur_var.negative

        for inst in GlobalState.get_cand_insts(cur_var.labels):
            if inst in used_instances:
                continue

            solution_dict[cur_var] = inst

            used_instances.add(inst)
            yield from gen_func(i_var + 1)
            used_instances.remove(inst)

    yield from gen_func(0)


def check_solution(sol_dict: Solution) -> bool:
    """check if an instance is assigned to multiple variables"""
    used_insts = set()

    for csp_var, cand_inst in sol_dict.items():
        if cand_inst in used_insts:
            return False

        used_insts.add(cand_inst)

    return True


def get_solution_heuristic_score(sol_dict: Solution) -> tuple[float, str]:
    instances = list(sol_dict.values())
    assert len(instances) > 0

    total_dist = 0
    for i in range(len(instances)):
        for j in range(i):
            total_dist += np.linalg.norm(
                instances[i].bbox.center - instances[j].bbox.center
            )

    solution_str = "-".join(
        [csp_var.get_identifier() for csp_var, _ in sol_dict.items()]
    )

    return (total_dist, solution_str)

    # # compute the center of all instances
    # center = np.mean([inst.bbox.center for inst in instances], axis=0)

    # # compute the average distance to the center
    # return np.average([np.linalg.norm(inst.bbox.center - center) for inst in instances])


def get_solution_heuristic_score_2(
    csp_var_groups: set[tuple[CSPVar, ...]],
    sol_dict: Solution,
) -> float:
    instances = list(sol_dict.values())
    assert len(instances) > 0

    total_dist = 0
    for grp in csp_var_groups:
        inst_centers = [sol_dict[csp_var].bbox.center for csp_var in grp]
        center = np.mean(inst_centers, axis=0)
        total_dist += np.sum([np.linalg.norm(c - center) for c in inst_centers])

    return total_dist


class CSPSolver:
    """
    the CSP (Constraint Satisfication Problem) whose solution will give the solution to
    the 3D visual grounding problem.
    """

    def __init__(self):
        self.variables: set[CSPVar] = set()
        self.normal_constraints: set[CSPConstraint] = set()
        self.minmax_constraints: set[CSPConstraint] = set()

        # # only used to ensure each variable has at most one min/max constraint
        # self.minmax_variables: set[CSPVar] = set()

    def create_var(self, labels: list[str] = [], negative: bool = False) -> CSPVar:
        csp_var = CSPVar(labels=labels)
        csp_var.negative = negative
        self.variables.add(csp_var)
        return csp_var

    def add_constraint(self, constraint: CSPConstraint):
        assert hasattr(constraint, "FUNC_NAME")

        func_names: set[str] = set()
        if isinstance(constraint.FUNC_NAME, str):
            func_names.add(constraint.FUNC_NAME)
        else:
            func_names |= set(constraint.FUNC_NAME)

        if func_names & {"CONSTRAINT_MIN_OF", "CONSTRAINT_MAX_OF"}:
            self.minmax_constraints.add(constraint)
            # # a variable can only have a single min/max constraint
            # assert constraint.get_target_var() not in self.minmax_variables
            # self.minmax_variables.add(constraint.get_target_var())
        else:
            self.normal_constraints.add(constraint)

    def get_constraints(self, csp_var: CSPVar) -> set[CSPConstraint]:
        """return all constraints related to the given variable"""
        return {con for con in self.normal_constraints if csp_var in con.get_vars()}

    def solve(
        self,
        query: str,
        select_solution: str = "min_dist",
        verbose: bool = False,
    ):
        assert query
        assert self.variables
        # assert self.normal_constraints or self.minmax_constraints

        if verbose:
            print()
            print("=" * 30)
            print()
            print("solving csp:")
            print(self.get_desc_str())

        normal_vars = [csp_var for csp_var in self.variables if not csp_var.negative]
        negative_vars = [csp_var for csp_var in self.variables if csp_var.negative]

        for con in self.minmax_constraints:
            assert all(not csp_var.negative for csp_var in con.get_vars())

        # find the target variable
        target_var_set = set(var for var in self.variables if var.is_target)
        assert len(target_var_set) == 1
        target_var = next(iter(target_var_set))

        # if the target variable has only a single candidate, skip the searching
        target_var_cand_insts = GlobalState.get_cand_insts(target_var.labels)
        if len(target_var_cand_insts) == 1:
            if verbose:
                print()
                print("target is unique.")
                print()
                print("=" * 30)
                print()

            set_target_info(
                best_instance=target_var_cand_insts[0],
                candidate_instances=target_var_cand_insts,
                anchor_instances={},
                csp_desc=self.get_desc_str(),
                llm_used=False,
            )
            return

        # iterate through possible solutions
        valid_solutions: list[Solution] = []
        solution_counter = 0

        for sol_dict in solution_generator(csp_vars=normal_vars):
            solution_counter += 1
            if solution_counter > 1000:
                break

            # discard this solution if an instance is used twice
            if not check_solution(sol_dict):
                continue

            # we should have an assignment for all variables by now
            assert len(sol_dict) + len(negative_vars) == len(self.variables)
            assert all(var in sol_dict for var in normal_vars)

            con_failed = False
            for con in self.normal_constraints:
                neg_vars = {csp_var for csp_var in con.get_vars() if csp_var.negative}

                if neg_vars:
                    assert len(neg_vars) == 1
                    neg_var = next(iter(neg_vars))
                    assert neg_var not in sol_dict

                    neg_var_cand_insts = GlobalState.get_cand_insts(neg_var.labels)
                    assert len(neg_var_cand_insts) >= 1

                    for inst in neg_var_cand_insts:
                        sol_dict[neg_var] = inst
                        if con.check_solution(sol_dict):
                            con_failed = True
                            break

                    del sol_dict[neg_var]

                else:
                    if not con.check_solution(sol_dict):
                        con_failed = True

                if con_failed:
                    break

            if not con_failed:
                valid_solutions.append(sol_dict)

        if valid_solutions:
            assert all(target_var in sol_dict for sol_dict in valid_solutions)

            if verbose:
                print()
                print("valid solutions:")
                for sol in valid_solutions:
                    print(solution_to_str(sol))

            minmax_constraints = sorted(self.minmax_constraints, key=lambda x: x.con_id)
            normal_constraints = sorted(self.normal_constraints, key=lambda x: x.con_id)

            # print()
            # print(self.get_desc_str())
            # print()

            # print()
            # print("num minmax_con:", len(minmax_constraints))
            # for con in minmax_constraints:
            #     print(con.con_id, con.get_desc_str())
            # print()

            # print()
            # for sol in valid_solutions:
            #     for csp_var, inst in sol.items():
            #         print(
            #             csp_var.get_identifier(),
            #             f"{inst.label}: {inst.inst_id}",
            #             end=" || ",
            #         )
            #     print()
            # print()

            for minmax_con in minmax_constraints:
                con_target_var = minmax_con.get_target_var()
                con_anchor_vars: set[CSPVar] = set()

                for normal_con in normal_constraints:
                    if normal_con.get_target_var() == con_target_var:
                        con_anchor_vars |= normal_con.get_vars() - {con_target_var}

                anchor_var_groups: dict[tuple[ObjInstance, ...], list[Solution]] = (
                    defaultdict(lambda: [])
                )

                # print()
                for sol in valid_solutions:
                    key = tuple(
                        sorted(
                            [sol[csp_var] for csp_var in con_anchor_vars],
                            key=lambda x: x.inst_id,
                        )
                    )
                    # print([f"{inst.inst_id}-{inst.label}" for inst in key])
                    anchor_var_groups[key].append(sol)
                # print()

                # print()
                # print("num groups:", len(anchor_var_groups))
                # print()
                valid_solutions: list[Solution] = []
                for _, sols in anchor_var_groups.items():
                    valid_solutions += minmax_con.filter_solutions(sols)

            if verbose:
                print()
                print("valid solutions after handling min/max constraints:")
                for sol in valid_solutions:
                    print(solution_to_str(sol))

            # print()
            # for sol in valid_solutions:
            #     for csp_var, inst in sol.items():
            #         print(
            #             csp_var.get_identifier(),
            #             f"{inst.label}: {inst.inst_id}",
            #             end=" || ",
            #         )
            #     print()
            # print()

            # select one solution with a heuristic...
            first_valid_solution = valid_solutions[0].copy()
            valid_solutions.sort(key=get_solution_heuristic_score)
            # print()
            # print("2:", len(valid_solutions))
            # print("2:", valid_solutions[0][target_var].inst_id)
            # print()

            best_solution_0 = valid_solutions[0]
            candidate_instances = {best_solution_0[target_var]}
            good_solutions = [best_solution_0]

            if len(valid_solutions) >= 2:
                best_solution_1 = valid_solutions[-1]
                candidate_instances.add(best_solution_1[target_var])
                good_solutions.append(best_solution_1)

            if len(valid_solutions) >= 3:
                best_solution_2 = valid_solutions[1]
                candidate_instances.add(best_solution_2[target_var])
                good_solutions.append(best_solution_2)

            # if len(valid_solutions) >= 4:
            #     best_solution_3 = valid_solutions[-2]
            #     candidate_instances.add(best_solution_3[target_var])
            #     good_solutions.append(best_solution_3)

            # target_var_insts = {sol_dict[target_var] for sol_dict in valid_solutions}
            llm_used = False
            # query_words = set(query.lower().strip().split(" "))
            # view_dep_words = {
            #     "front",
            #     "behind",
            #     "back",
            #     "right",
            #     "left",
            #     "facing",
            #     "leftmost",
            #     "rightmost",
            #     "looking",
            #     "across",
            # }
            # view_dep = len(query_words & view_dep_words) > 0
            # if len(target_var_insts) >= 2 and not view_dep:
            #     best_solution_0 = select_best_solution(
            #         query=query,
            #         csp_desc=self.get_desc_str(),
            #         valid_solutions=good_solutions,
            #     )
            #     llm_used = True

            if select_solution == "max_dist":
                best_solution_0, best_solution_1 = best_solution_1, best_solution_0
            elif select_solution == "random":
                best_solution_0 = random.choice(valid_solutions)
            elif select_solution == "first":
                best_solution_0 = first_valid_solution
            else:
                assert select_solution == "min_dist"

            if verbose:
                print()
                print("best solution:")
                print(solution_to_str(best_solution_0))

            set_target_info(
                best_instance=best_solution_0[target_var],
                candidate_instances=candidate_instances,
                anchor_instances={
                    csp_var.get_identifier(): inst
                    for csp_var, inst in best_solution_0.items()
                    if csp_var != target_var
                },
                csp_desc=self.get_desc_str(),
                llm_used=llm_used,
            )

        print()
        print("=" * 30)
        print()

    def solve_naive(self):
        raise NotImplementedError()

    def solve_non_csp(self):
        def generate_solutions(
            csp_vars: list[CSPVar],
            csp_var_insts: dict[CSPVar, set[ObjInstance]],
        ) -> Generator[Solution, None, None]:
            csp_vars = list(set(csp_vars))
            total_sols = 0

            solution_dict: Solution = {}
            used_instances: set[ObjInstance] = set()

            def gen_func(i_var: int) -> Generator[Solution, None, None]:
                nonlocal solution_dict
                nonlocal total_sols
                if total_sols >= 1000:
                    return

                if i_var >= len(csp_vars):
                    total_sols += 1
                    yield solution_dict.copy()
                    return

                cur_var = csp_vars[i_var]
                for inst in csp_var_insts[cur_var]:
                    if inst in used_instances:
                        continue

                    solution_dict[cur_var] = inst

                    used_instances.add(inst)
                    yield from gen_func(i_var + 1)
                    used_instances.remove(inst)

            yield from gen_func(0)

        csp_var_dep_cons: dict[CSPVar, set[CSPConstraint]] = {
            csp_var: {
                con
                for con in self.normal_constraints
                if con.get_target_var() == csp_var
            }
            for csp_var in self.variables
        }

        csp_var_minmax_cons: dict[CSPVar, set[CSPConstraint]] = {
            csp_var: {
                con
                for con in self.minmax_constraints
                if con.get_target_var() == csp_var
            }
            for csp_var in self.variables
        }

        csp_var_insts: dict[CSPVar, set[ObjInstance]] = {
            csp_var: set(GlobalState.get_cand_insts(csp_var.labels))
            for csp_var in self.variables
        }

        constraints: set[CSPConstraint] = self.normal_constraints.copy()

        for i in range(100):
            if not constraints:
                break

            con_processed = False

            processed_cons: set[CSPConstraint] = set()
            for con in constraints:
                con_target_var = con.get_target_var()
                con_anchor_vars = con.get_vars() - {con_target_var}

                if all(
                    len(csp_var_dep_cons[csp_var]) == 0 for csp_var in con_anchor_vars
                ):
                    con_processed = True
                    target_var_insts: set[ObjInstance] = set()

                    for sol_dict in generate_solutions(con.get_vars(), csp_var_insts):
                        assert con_target_var in sol_dict
                        assert all(csp_var in sol_dict for csp_var in con_anchor_vars)
                        if con.check_solution(sol_dict):
                            target_var_insts.add(sol_dict[con_target_var])

                    csp_var_insts[con_target_var] = target_var_insts.copy()
                    csp_var_dep_cons[con_target_var].remove(con)
                    processed_cons.add(con)

            constraints -= processed_cons

            processed_vars: set[CSPVar] = set()
            for csp_var, minmax_cons in csp_var_minmax_cons.items():
                if not minmax_cons or csp_var_dep_cons[csp_var]:
                    continue

                # assert len(minmax_cons) == 1
                minmax_con = next(iter(minmax_cons))

                con_target_var = minmax_con.get_target_var()
                con_anchor_vars = minmax_con.get_vars() - {con_target_var}

                if con_anchor_vars and any(
                    csp_var_dep_cons[anchor_var] for anchor_var in con_anchor_vars
                ):
                    continue

                con_processed = True
                solutions = list(
                    generate_solutions(minmax_con.get_vars(), csp_var_insts)
                )
                solutions: list[Solution] = minmax_con.filter_solutions(solutions)
                csp_var_insts[con_target_var] = {
                    sol[con_target_var] for sol in solutions
                }

                processed_vars.add(csp_var)

            for csp_var in processed_vars:
                del csp_var_minmax_cons[csp_var]

            if not con_processed:
                return

        target_var_set = set(var for var in self.variables if var.is_target)
        assert len(target_var_set) == 1
        target_var = next(iter(target_var_set))

        # if the target variable has only a single candidate, skip the searching
        target_var_cand_insts = csp_var_insts[target_var]
        if target_var_cand_insts:
            set_target_info(
                best_instance=next(iter(target_var_cand_insts)),
                candidate_instances=target_var_cand_insts,
                anchor_instances={},
                csp_desc=self.get_desc_str(),
                llm_used=False,
            )

    def get_desc_str(self) -> str:
        desc = []

        for csp_var in self.variables:
            desc.append(csp_var.get_identifier())
            if csp_var.is_target:
                desc[-1] += " (target)"
            if csp_var.negative:
                desc[-1] += " (negative)"

        for con in self.normal_constraints:
            desc.append(f"{con.apparent_name} {con.get_desc_str()}")

        for con in self.minmax_constraints:
            desc.append(f"{con.apparent_name} {con.get_desc_str()}")

        return "\n".join(desc)
