import argparse
import os

from scannet_utils import ScanNetScene

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, required=True)
parser.add_argument("--no-server", action="store_true")
parser.add_argument("--viz-port", type=int, default=8889)
parser.add_argument("--segs", action="store_true")


def parse_color(arg):
    if arg.count(":") != 3:
        raise argparse.ArgumentTypeError(
            "invalid bbox color mapping: missing component."
        )

    inst_id, r, g, b = arg.split(":")
    inst_id, r, g, b = inst_id.strip(), r.strip(), g.strip(), b.strip()

    if not inst_id or not r or not g or not b:
        raise argparse.ArgumentTypeError("invalid bbox color mapping: empty component.")

    if (
        (not r.replace(".", "", 1).isdigit())
        or (not g.replace(".", "", 1).isdigit())
        or (not b.replace(".", "", 1).isdigit())
    ):
        raise argparse.ArgumentTypeError("invalid bbox color mapping: rgb not float.")

    try:
        result = [inst_id, [float(r), float(g), float(b)]]
    except Exception:
        raise argparse.ArgumentTypeError("invalid bbox color mapping: i don't know.")

    if (
        result[1][0] < 0
        or result[1][0] > 1
        or result[1][1] < 0
        or result[1][1] > 1
        or result[1][2] < 0
        or result[1][2] > 1
    ):
        raise argparse.ArgumentTypeError(
            "invalid bbox color mapping: invalid rgb value."
        )

    return result


parser.add_argument(
    "-bh",
    "--bbox-highlight",
    metavar="inst_id:r,g,b",
    type=parse_color,
    action="append",
)
parser.add_argument(
    "-sh",
    "--seg-highlight",
    metavar="inst_id:r,g,b",
    type=parse_color,
    action="append",
)
parser.add_argument(
    "-c",
    "--color",
    metavar="inst_id:r,g,b",
    type=parse_color,
    action="append",
)

group = parser.add_mutually_exclusive_group()
group.add_argument("--mask3d", action="store_true")
group.add_argument("--maskcluster", action="store_true")

args = parser.parse_args()

scene_id = args.scene if "scene" in args.scene else "scene" + args.scene
scannet_root = "./data"
viz_dir_root = "./visualize"
mask3d_pred_root = "./data/eval_output/mask3d_val" if args.mask3d else None
maskcluster_pred_root = "./data/eval_output/maskcluster" if args.maskcluster else None

seg_colors = {} if args.color is None else {c[0]: c[1] for c in args.color}
bbox_highlights = (
    {} if args.bbox_highlight is None else {bh[0]: bh[1] for bh in args.bbox_highlight}
)
seg_highlights = (
    {} if args.seg_highlight is None else {sh[0]: sh[1] for sh in args.seg_highlight}
)

scene_path = os.path.join(scannet_root, "scans", scene_id)
scene = ScanNetScene(
    scene_path,
    mask3d_pred_path=mask3d_pred_root,
    maskcluster_pred_path=maskcluster_pred_root,
    add_room_center=False,
    add_room_corners=False,
)
viz_dir = scene.visualize_pyviz3d(
    viz_dir_root,
    segments=args.segs,
    seg_colors=seg_colors,
    bbox_highlights=bbox_highlights,
    seg_highlights=seg_highlights,
)

if not args.no_server:
    os.system(f"python -m http.server {args.viz_port} -d {viz_dir}")
