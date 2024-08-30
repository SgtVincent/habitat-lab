import numpy as np
import magnum as mn

def project_3d_to_2d(
    camera_info,
    point_3d
):
    viewport = camera_info['viewport']
    projection_matrix = mn.Matrix4(camera_info['projection_matrix'])
    camera_matrix = mn.Matrix4(camera_info['camera_matrix'])
    projection_size = camera_info['projection_size']

    # get the scene render camera and sensor object
    W, H = viewport[0], viewport[1]

    # use the camera and projection matrices to transform the point onto the near plane
    projected_point_3d = projection_matrix.transform_point(
        camera_matrix.transform_point(point_3d)
    )
    # convert the 3D near plane point to integer pixel space
    point_2d = mn.Vector2(projected_point_3d[0], -projected_point_3d[1])
    point_2d = point_2d / projection_size
    point_2d += mn.Vector2(0.5)
    point_2d *= viewport

    out_bound = 10
    point_2d = np.nan_to_num(point_2d, nan=W + out_bound, posinf=W + out_bound,
                             neginf=-out_bound)
    return point_2d.astype(int)


def get_coords(sampled_points, camera_info):
    sampled_points = np.array(sampled_points)
    if sampled_points.ndim <= 1:
        sampled_points = np.array([sampled_points])
    points_2d = []
    for point_3d in sampled_points:
        point_2d = project_3d_to_2d(
            camera_info,
            point_3d
        )
        points_2d.append(point_2d)
    return points_2d
