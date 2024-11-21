import numpy as np
import numpy.typing as npt


def lookat_matrix(
    eye: npt.NDArray, target: npt.NDArray, up: npt.NDArray = [0, 0, 1]
) -> npt.NDArray:
    """
    build a lookat matrix (4x4): global coordiantes -> camera/eye coordiantes
    (might still have bugs...)
    """
    mat = np.zeros((4, 4))
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    axis_z = target - eye
    axis_z /= np.linalg.norm(axis_z)

    up /= np.linalg.norm(up)
    axis_x = np.cross(up, axis_z)
    axis_x /= np.linalg.norm(axis_x)

    axis_y = np.cross(axis_z, axis_x)
    axis_y /= np.linalg.norm(axis_y)

    mat[0, :3] = axis_x
    mat[1, :3] = axis_y
    mat[2, :3] = axis_z
    mat[:3, 3] = -mat[:3, :3] @ eye
    mat[3, 3] = 1

    return mat


def check(expr: bool, msg: str = "") -> None:
    if not (expr):
        raise SystemError(msg)


def check_list_of_type(x: list, ty: type, msg: str = "") -> None:
    check(isinstance(x, list) and all(isinstance(i, ty) for i in x), msg)


def check_set_of_type(x: set, ty: type, msg: str = "") -> None:
    check(isinstance(x, set) and all(isinstance(i, ty) for i in x), msg)


def is_list_of_type(x: list, ty: type) -> bool:
    return isinstance(x, list) and all(isinstance(i, ty) for i in x)


def is_set_of_type(x: set, ty: type) -> bool:
    return isinstance(x, set) and all(isinstance(i, ty) for i in x)
