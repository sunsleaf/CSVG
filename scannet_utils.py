import json
import os
from collections import Counter
from dataclasses import dataclass
from random import random, seed
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import pyviz3d
import pyviz3d.visualizer
from plyfile import PlyData
from sklearn.cluster import DBSCAN
from typing_extensions import Self

from data.scannet200_constants import CLASS_LABELS_200, VALID_CLASS_IDS_200


def idx_2_label_200(idx):
    """copied from https://github.com/ripl/Transcrib3D/blob/main/preprocessing/gen_obj_list.py"""
    return CLASS_LABELS_200[VALID_CLASS_IDS_200.index(idx)]


def read_mesh_vertices(mesh_file: str, load_color: bool) -> npt.NDArray:
    """read XYZ (and RGB) for each vertex."""
    assert os.path.isfile(mesh_file), mesh_file
    with open(mesh_file, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        num_cols = 6 if load_color else 3
        vertices = np.zeros(shape=[num_verts, num_cols], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        if load_color:
            vertices[:, 3] = plydata["vertex"].data["red"] / 255.0
            vertices[:, 4] = plydata["vertex"].data["green"] / 255.0
            vertices[:, 5] = plydata["vertex"].data["blue"] / 255.0
    return vertices


def transform_vertices(
    meta_file: str,
    mesh_vertices: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """read alignment matrix and transform mesh vertices"""
    assert os.path.isfile(meta_file)
    lines = open(meta_file).readlines()

    axis_align_matrix = None
    for line in lines:
        if "axisAlignment" in line:
            axis_align_matrix = [
                float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
            ]

    if axis_align_matrix is not None:
        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
        # print(axis_align_matrix)

        pts = np.ones((mesh_vertices.shape[0], 4))
        pts[:, 0:3] = mesh_vertices[:, 0:3]
        pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4

        aligned_vertices = mesh_vertices.copy()
        aligned_vertices[:, 0:3] = pts[:, 0:3]
        return aligned_vertices, axis_align_matrix
    else:
        print()
        print("no axis alignment matrix!")
        print()
        return mesh_vertices, np.eye(4)


class RawInstance(TypedDict):
    inst_id: str
    label: str
    vertices: npt.NDArray
    score: float


class BoundingBox3D:
    def __init__(
        self,
        pmin0: npt.NDArray = None,
        pmax0: npt.NDArray = None,
        pcenter0: npt.NDArray = None,
        psize0: npt.NDArray = None,
    ):
        if pmin0 is not None and pmax0 is not None:
            assert pcenter0 is None and psize0 is None
            pmin0 = np.array(pmin0)
            pmax0 = np.array(pmax0)
        elif pcenter0 is not None and psize0 is not None:
            assert pmin0 is None and pmax0 is None
            pcenter0 = np.array(pcenter0)
            psize0 = np.array(psize0)
            pmin0 = pcenter0 - psize0 * 0.5
            pmax0 = pcenter0 + psize0 * 0.5

        assert pmin0.shape == (3,)
        assert pmax0.shape == (3,)
        pmin = np.minimum(pmin0, pmax0)
        pmax = np.maximum(pmin0, pmax0)
        self.pmin = pmin
        self.pmax = pmax
        self.center = 0.5 * (pmin + pmax)
        self.size = pmax - pmin
        self.max_extent = np.max(pmax - pmin)
        self.extents = {
            "x": pmax[0] - pmin[0],
            "y": pmax[1] - pmin[1],
            "z": pmax[2] - pmin[2],
        }

    def contains(self, p: npt.NDArray) -> bool:
        return np.all(p > self.pmin) and np.all(p < self.pmax)

    def intersect(self, other: Self) -> Self:
        return BoundingBox3D(
            pmin0=np.maximum(self.pmin, other.pmin),
            pmax0=np.minimum(self.pmax, other.pmax),
        )

    def union(self, other: Self) -> Self:
        return BoundingBox3D(
            pmin0=np.minimum(self.pmin, other.pmin),
            pmax0=np.maximum(self.pmax, other.pmax),
        )

    def volume(self) -> float:
        return np.prod(self.pmax - self.pmin)

    def iou(self, other: Self) -> float:
        if self.volume() == 0 or other.volume() == 0:
            return 0.0
        return self.intersect(other).volume() / self.union(other).volume()


def filter_raw_instances(inst_map: dict[str, RawInstance]) -> dict[str, RawInstance]:
    def filter_pointcould(points):
        """copied from the Transcrib3D repo"""
        # use dbscan to filter out outlier points
        dbscan = DBSCAN(eps=0.1, min_samples=20)
        if points.shape[1] == 3:
            dbscan.fit(points)
        else:
            dbscan.fit(points[:, 0:3])
        counter = Counter(dbscan.labels_)
        main_idx = counter.most_common(2)[0][0]
        if main_idx == -1:
            main_idx = counter.most_common(2)[-1][0]
        # print("counter:",counter)
        # print("main_idx:",main_idx)
        points_filtered = points[dbscan.labels_ == main_idx]
        return points_filtered

    def calc_iou(inst_a: RawInstance, inst_b: RawInstance) -> float:
        bbox_a = BoundingBox3D(
            pmin0=inst_a["vertices"][:, :3].min(axis=0),
            pmax0=inst_a["vertices"][:, :3].max(axis=0),
        )
        bbox_b = BoundingBox3D(
            pmin0=inst_b["vertices"][:, :3].min(axis=0),
            pmax0=inst_b["vertices"][:, :3].max(axis=0),
        )
        return bbox_a.iou(bbox_b)

    # filter the point cloud of each instance
    filtered_insts_1: dict[str, RawInstance] = {}
    for k, inst_0 in inst_map.items():
        inst_1 = inst_0.copy()
        inst_1["vertices"] = filter_pointcould(inst_0["vertices"])
        filtered_insts_1[k] = inst_1

    # filter out overlapped instances with lower scores
    filtered_insts_2: dict[str, RawInstance] = {}
    for k, inst_1 in filtered_insts_1.items():
        for inst in filtered_insts_1.values():
            iou = calc_iou(inst_1, inst)
            if iou >= 0.7 and inst_1["score"] < inst["score"]:
                break
        else:
            filtered_insts_2[k] = inst_1

    return filtered_insts_2


def read_instances(
    agg_file_path: str,
    seg_file_path: str,
    vertex_buffer: npt.NDArray,
) -> dict[str, RawInstance]:
    """return a dict: instance id -> instance info"""

    # read segments
    seg_to_verts = {}
    with open(seg_file_path) as f:
        data = json.load(f)
        num_verts = len(data["segIndices"])
        for i in range(num_verts):
            seg_id = data["segIndices"][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]

    # read instances
    inst_id_to_insts: dict[str, RawInstance] = {}
    with open(agg_file_path) as f:
        data = json.load(f)
        num_objects = len(data["segGroups"])
        for i in range(num_objects):
            inst = RawInstance()
            inst["id"] = str(data["segGroups"][i]["objectId"])
            inst["label"] = str(data["segGroups"][i]["label"])

            # assign vertices to the instance
            for seg in data["segGroups"][i]["segments"]:
                verts = seg_to_verts[seg]
                if "vertices" not in inst:
                    inst["vertices"] = vertex_buffer[verts, :]
                else:
                    inst["vertices"] = np.vstack(
                        [inst["vertices"], vertex_buffer[verts, :]]
                    )

            inst_id_to_insts[inst["id"]] = inst

    return inst_id_to_insts


def read_instances_mask3d(
    scene_id: str, pred_path: str, vertex_buffer: npt.NDArray
) -> dict[str, RawInstance]:
    assert os.path.isdir(pred_path), pred_path
    assert vertex_buffer.shape[1] in (3, 6), vertex_buffer.shape

    scene_pred_file = os.path.join(pred_path, scene_id + ".txt")
    assert os.path.isfile(scene_pred_file), scene_pred_file

    inst_id_to_insts: dict[str, RawInstance] = {}
    with open(scene_pred_file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            mask_rel_path, label_id, score = line.split()

            score = float(score)
            if score < 0.5:
                continue

            label_str = idx_2_label_200(int(label_id))
            mask_file = os.path.join(pred_path, mask_rel_path)
            assert os.path.isfile(mask_file), mask_file

            masks = np.loadtxt(mask_file, dtype=bool)
            assert masks.shape[0] == vertex_buffer.shape[0]
            inst_id_to_insts[str(i)] = RawInstance(
                id=str(i),
                label=label_str,
                vertices=vertex_buffer[masks, :],
                score=score,
            )

    return filter_raw_instances(inst_id_to_insts)


def read_instances_maskcluster(
    scene_id: str, pred_path: str, vertex_buffer: npt.NDArray
) -> dict[str, RawInstance]:
    assert os.path.isdir(pred_path), pred_path
    assert vertex_buffer.shape[1] in (3, 6), vertex_buffer.shape

    scene_pred_file = os.path.join(pred_path, scene_id + ".npz")
    pred = np.load(scene_pred_file)

    pred_masks = pred["pred_masks"]
    pred_scores = pred["pred_score"]
    pred_classes = pred["pred_classes"]

    num_instances = pred["pred_masks"].shape[1]
    assert pred_masks.shape[0] == vertex_buffer.shape[0]
    assert pred_scores.shape[0] == num_instances
    assert pred_classes.shape[0] == num_instances

    return filter_raw_instances(
        {
            str(i): RawInstance(
                id=str(i),
                label=idx_2_label_200(pred_classes[i]),
                vertices=vertex_buffer[pred_masks[:, i], :],
                score=pred_scores[i],
            )
            for i in range(num_instances)
        }
    )


class ObjInstance:
    """instance of an object of a certain label/category"""

    def __init__(self, instance_id: str, label: str, vertices: npt.NDArray):
        self.inst_id = str(instance_id)
        self.label = label
        self.vertices = vertices
        self.bbox = BoundingBox3D(
            pmin0=np.min(vertices[:, :3], axis=0),
            pmax0=np.max(vertices[:, :3], axis=0),
        )

    def __hash__(self) -> int:
        return hash(self.inst_id)

    def __eq__(self, other: Self) -> bool:
        return self.inst_id == other.inst_id


class ScanNetScene:
    def __init__(
        self,
        scene_path: str,
        mask3d_pred_path: str | None = None,
        maskcluster_pred_path: str | None = None,
        cache_root: str | None = None,
        add_room_center: bool = True,
        add_room_corners: bool = True,
    ):
        scene_path = os.path.normpath(scene_path)
        # assert os.path.isdir(scene_path)

        scene_id = os.path.basename(scene_path).strip()
        self.scene_id = scene_id
        self.viz_suffix = "gt"
        # print(f"loading {scene_id}.")

        scene_prefix = f"{scene_path}/{scene_id}"
        ply_file = f"{scene_prefix}_vh_clean_2.ply"
        agg_file = f"{scene_prefix}.aggregation.json"
        seg_file = f"{scene_prefix}_vh_clean_2.0.010000.segs.json"
        meta_file = f"{scene_prefix}.txt"
        # assert os.path.isfile(ply_file)
        # assert os.path.isfile(agg_file)
        # assert os.path.isfile(seg_file)
        # assert os.path.isfile(meta_file)

        instance_map: dict[str, RawInstance] = {}
        if mask3d_pred_path is not None:
            # print("loading mask3d.")
            assert maskcluster_pred_path is None
            self.viz_suffix = "mask3d"

            should_load_data = True
            cache_file = None

            if cache_root:
                cache_dir = os.path.join(cache_root, "instances_mask3d")
                cache_file = os.path.join(cache_dir, f"{scene_id}.npy")

                # create the cache folder if it does not exist
                if not os.path.isdir(cache_dir):
                    os.system(f"mkdir -p {cache_dir}")

                # if the cache file exists, load it directly
                elif os.path.isfile(cache_file):
                    should_load_data = False
                    instance_map = np.load(cache_file, allow_pickle=True).item()

            if should_load_data:
                # load scene data
                vert_buf = read_mesh_vertices(mesh_file=ply_file, load_color=True)
                vert_buf, _ = transform_vertices(
                    meta_file=meta_file, mesh_vertices=vert_buf
                )

                instance_map = read_instances_mask3d(
                    scene_id=scene_id,
                    pred_path=mask3d_pred_path,
                    vertex_buffer=vert_buf,
                )

                if cache_file:
                    np.save(cache_file, instance_map)

        elif maskcluster_pred_path is not None:
            # load scene data
            vert_buf = read_mesh_vertices(mesh_file=ply_file, load_color=True)
            vert_buf, _ = transform_vertices(
                meta_file=meta_file, mesh_vertices=vert_buf
            )

            # print("loading maskclustering.")
            assert mask3d_pred_path is None
            self.viz_suffix = "maskcluster"
            instance_map = read_instances_maskcluster(
                scene_id=scene_id,
                pred_path=maskcluster_pred_path,
                vertex_buffer=vert_buf,
            )

        else:
            self.viz_suffix = "gt"

            should_load_data = True
            cache_file = None

            if cache_root:
                cache_dir = os.path.join(cache_root, "instances_gt")
                cache_file = os.path.join(cache_dir, f"{scene_id}.npy")

                # create the cache folder if it does not exist
                if not os.path.isdir(cache_dir):
                    os.system(f"mkdir -p {cache_dir}")

                # if the cache file exists, load it directly
                elif os.path.isfile(cache_file):
                    should_load_data = False
                    instance_map = np.load(cache_file, allow_pickle=True).item()

            if should_load_data:
                # load scene data
                vert_buf = read_mesh_vertices(mesh_file=ply_file, load_color=True)
                vert_buf, _ = transform_vertices(
                    meta_file=meta_file, mesh_vertices=vert_buf
                )

                # print("loading groundtruth.")
                instance_map = read_instances(
                    agg_file_path=agg_file,
                    seg_file_path=seg_file,
                    vertex_buffer=vert_buf,
                )

                if cache_file:
                    np.save(cache_file, instance_map)

        # remove instances with too few vertices
        assert instance_map
        instance_map = {
            k: v for k, v in instance_map.items() if v["vertices"].shape[0] >= 10
        }
        # print(f"{len(instance_map)} instances loaded.")

        self.raw_instance_map: dict[str, ObjInstance] = {}
        self.instance_map: dict[str, list[ObjInstance]] = {}
        bboxes = []

        # build instance map from raw instance map...
        for inst in instance_map.values():
            obj_inst = ObjInstance(
                instance_id=inst["id"],
                label=inst["label"],
                vertices=inst["vertices"],
            )
            bboxes.append(obj_inst.bbox)
            assert obj_inst.bbox.volume() > 0

            assert obj_inst.inst_id not in self.raw_instance_map
            self.raw_instance_map[obj_inst.inst_id] = obj_inst

            if inst["label"] in self.instance_map:
                self.instance_map[inst["label"]].append(obj_inst)
            else:
                self.instance_map[inst["label"]] = [obj_inst]

        self.bbox = BoundingBox3D(
            pmin0=np.min([bbox.pmin for bbox in bboxes], axis=0),
            pmax0=np.max([bbox.pmax for bbox in bboxes], axis=0),
        )
        self.room_center = self.bbox.center

        if add_room_center:
            self.instance_map["room center"] = [
                ObjInstance(
                    instance_id=-1,
                    label="room center",
                    vertices=np.array(
                        [self.room_center - 1e-5, self.room_center + 1e-5]
                    ),
                )
            ]

        if add_room_corners:
            self.room_corners = [
                self.bbox.pmin,
                np.array([self.bbox.pmax[0], self.bbox.pmin[1], self.bbox.pmin[2]]),
                np.array([self.bbox.pmin[0], self.bbox.pmax[1], self.bbox.pmin[2]]),
                np.array([self.bbox.pmax[0], self.bbox.pmax[1], self.bbox.pmin[2]]),
            ]
            self.instance_map["room corner"] = [
                ObjInstance(
                    instance_id=-2,
                    label="room corner",
                    vertices=np.array([corner - 1e-5, corner + 1e-5]),
                )
                for corner in self.room_corners
            ]

        # TODO: add "room front", "room back", "room left" and "room right"

    def get_instance_map(self) -> dict[str, list[ObjInstance]]:
        return self.instance_map

    def get_raw_instance_map(self):
        return self.raw_instance_map

    def get_distractors(self, inst_id: str) -> list[ObjInstance]:
        inst_0 = self.raw_instance_map[str(inst_id).strip()]
        return [inst for inst in self.instance_map[inst_0.label] if inst != inst_0]

    def get_room_center(self) -> npt.NDArray:
        return self.room_center

    def get_room_corners(self) -> list[npt.NDArray]:
        return self.room_corners

    def get_instance_bbox(self, inst_id: str) -> BoundingBox3D:
        return self.raw_instance_map[str(inst_id)].bbox

    def is_unique_label(self, label: str) -> bool:
        return len(self.instance_map[label]) == 1

    @dataclass
    class BBoxInfo:
        pmin: npt.NDArray
        pmax: npt.NDArray
        color: npt.NDArray
        name: str

        def __post_init__(self):
            self.pmin = np.array(self.pmin)
            self.pmax = np.array(self.pmax)
            self.color = np.array(self.color)

            assert self.pmin.shape == (3,)
            assert self.pmax.shape == (3,)
            assert self.color.shape == (3,)

            assert np.all(self.pmax >= self.pmin)
            assert np.all((self.color >= 0) & (self.color <= 1))

            self.color = (self.color * 255).astype(np.uint8)

    def visualize_pyviz3d(
        self,
        viz_root_dir: str,
        target_id: int | str | None = None,
        target_color: tuple[float, float, float] | None = None,
        pred_bbox: BoundingBox3D | None = None,
        anchor_bboxes: dict[str, BoundingBox3D | list[BoundingBox3D]] | None = None,
        segments: bool = False,
        seg_colors: dict[str, tuple[float, float, float]] = {},
        bbox_highlights: dict[str, tuple[float, float, float]] = {},
        seg_highlights: dict[str, tuple[float, float, float]] = {},
        extra_bboxes: list[BBoxInfo] = [],
    ) -> str:
        assert not bbox_highlights or not seg_highlights
        viz = pyviz3d.visualizer.Visualizer()

        bbox_line_width = 0.02
        id_counter = 0

        if target_id is not None:
            target_id = str(target_id).strip()
            assert target_id.isdigit()

        if target_color is not None:
            target_color = np.array(target_color)
            assert target_color.shape == (3,)
            assert np.all((target_color >= 0) & (target_color <= 1))
            target_color = (target_color * 255).astype(np.uint8)

        for insts in self.instance_map.values():
            for inst in insts:
                # if inst.inst_id not in {"2", "7", "35", "37"}:
                #     continue

                if inst.vertices.shape[1] != 6:
                    continue

                point_positions = inst.vertices[:, :3]

                if seg_highlights:
                    if inst.inst_id in seg_highlights:
                        point_colors = (
                            np.array(
                                [seg_highlights[inst.inst_id]] * inst.vertices.shape[0]
                            )
                            * 255
                        ).astype(np.uint8)
                    else:
                        if segments:
                            point_colors = (
                                np.array([[225, 225, 225]] * inst.vertices.shape[0])
                            ).astype(np.uint8)
                        else:
                            point_colors = (inst.vertices[:, 3:] * 0.2 * 255).astype(
                                np.uint8
                            )

                else:
                    if segments:
                        if inst.inst_id in seg_colors:
                            color = seg_colors[inst.inst_id]
                            color[0] *= 255
                            color[1] *= 255
                            color[2] *= 255
                        else:
                            # seed(inst.inst_id)
                            color = [random() * 255, random() * 255, random() * 255]
                        point_colors = (
                            np.array([color] * inst.vertices.shape[0])
                        ).astype(np.uint8)
                    else:
                        point_colors = (inst.vertices[:, 3:] * 255).astype(np.uint8)

                viz.add_points(
                    f"inst-{inst.label}-{(id_counter := id_counter + 1)}",
                    point_positions,
                    point_colors,
                    point_size=50,
                )

                if target_id is not None and inst.inst_id == target_id:
                    viz.add_bounding_box(
                        f"bbox-target-{inst.label}-{inst.inst_id}-{(id_counter := id_counter + 1)}",
                        position=inst.bbox.center,
                        size=inst.bbox.size,
                        color=np.array([0, 255, 0])
                        if target_color is None
                        else target_color,
                        edge_width=bbox_line_width,
                    )

                if inst.inst_id in bbox_highlights:
                    viz.add_bounding_box(
                        f"bbox-highlight-{inst.label}-{inst.inst_id}-{(id_counter := id_counter + 1)}",
                        position=inst.bbox.center,
                        size=inst.bbox.size,
                        color=(np.array(bbox_highlights[inst.inst_id]) * 255).astype(
                            np.uint8
                        ),
                        edge_width=bbox_line_width,
                    )

        for bbox_info in extra_bboxes:
            bbox = BoundingBox3D(pmin0=bbox_info.pmin, pmax0=bbox_info.pmax)
            viz.add_bounding_box(
                f"bbox-{bbox_info.name}-{(id_counter := id_counter + 1)}",
                position=bbox.center,
                size=bbox.size,
                color=bbox_info.color,
                edge_width=bbox_line_width,
            )

        viz_dir = os.path.join(viz_root_dir, f"{self.scene_id}_{self.viz_suffix}")
        # print(f"viz_dir: {viz_dir}")
        if os.path.exists(viz_dir):
            os.system(f"rm -rf {viz_dir}")

        viz.save(viz_dir, verbose=False)

        return viz_dir

    def visualize_open3d(
        self,
        target_id: int | str | None = None,
        pred_bbox: BoundingBox3D | None = None,
        anchor_bboxes: dict[str, BoundingBox3D | list[BoundingBox3D]] | None = None,
        segments: bool = False,
    ):
        import open3d as o3d

        if target_id is not None:
            target_id = str(target_id).strip()
            assert target_id.isdigit()

        geometries = []

        for insts in self.instance_map.values():
            for inst in insts:
                # if inst.inst_id not in {"2", "7", "35", "37"}:
                #     continue

                if inst.vertices.shape[1] != 6:
                    continue

                point_positions = inst.vertices[:, :3]
                if segments:
                    point_colors = (
                        np.array(
                            [[random() * 255, random() * 255, random() * 255]]
                            * inst.vertices.shape[0]
                        )
                    ).astype(np.uint8)
                else:
                    point_colors = (inst.vertices[:, 3:] * 255).astype(np.uint8)

                point_cloud = o3d.t.geometry.PointCloud(point_positions)
                point_cloud.point.colors = point_colors
                geometries.append(point_cloud.to_legacy())

        o3d.visualization.draw_geometries(geometries)
