"""
几何体相交测试

为各种几何体类型进行相交测试

支持的几何体类型：
- PLANE (0): 平面
- SPHERE (2): 球体  
- CAPSULE (3): 胶囊体
- ELLIPSOID (4): 椭球体
- CYLINDER (5): 圆柱体
- BOX (6): 盒子
"""

import taichi as ti
from .utils import _transform_ray_to_local, _transform_point_to_world

@ti.func
def ray_triangle_distance(ray_start, ray_direction, v0, v1, v2):
    """返回射线与三角形的命中距离t，未命中返回-1.0
    使用 Möller-Trumbore 算法
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = ray_direction.cross(edge2)
    a = edge1.dot(h)
    t_ret = -1.0
    if ti.abs(a) >= 1e-6:  # 不平行
        f = 1.0 / a
        s = ray_start - v0
        u = f * s.dot(h)
        if 0.0 <= u <= 1.0:
            q = s.cross(edge1)
            v = f * ray_direction.dot(q)
            if v >= 0.0 and u + v <= 1.0:
                t = f * edge2.dot(q)
                if t > 1e-6:  # 正向命中
                    t_ret = t
    return t_ret

@ti.func
def ray_plane_distance(ray_start, ray_direction, center, size, rotation):
    """返回射线与平面命中距离 t；未命中返回 -1.0"""
    # 转换射线到平面的局部坐标系
    local_start, local_direction = _transform_ray_to_local(ray_start, ray_direction, center, rotation)

    # 在局部坐标系中，平面的法向量是 z 轴
    normal = ti.math.vec3(0.0, 0.0, 1.0)
    half_width = size[0]
    half_height = size[1]

    t_ret = -1.0
    denom = local_direction.dot(normal)

    # 避免除以零，检查光线是否与平面平行
    if ti.abs(denom) >= 1e-6:
        # 局部坐标系中平面在原点，所以只需要计算到原点的距离
        t = -local_start.z / denom
        # 如果 t 为正且命中在平面范围内
        if t >= 0:
            local_hit = local_start + t * local_direction
            if ti.abs(local_hit.x) <= half_width and ti.abs(local_hit.y) <= half_height:
                t_ret = t

    return t_ret

@ti.func
def ray_sphere_distance(ray_start, ray_direction, center, size, rotation):
    """返回射线与球体命中距离 t；未命中返回 -1.0"""
    # 球体旋转不会改变其形状
    radius = size[0]

    t_ret = -1.0
    oc = ray_start - center
    a = ray_direction.dot(ray_direction)
    b = 2.0 * oc.dot(ray_direction)
    c = oc.dot(oc) - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant >= 0:
        t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
        if t < 0:
            t = (-b + ti.sqrt(discriminant)) / (2.0 * a)
        if t >= 0:
            t_ret = t

    return t_ret

@ti.func
def ray_box_distance(ray_start, ray_direction, center, size, rotation):
    """返回射线与盒子命中距离 t；未命中返回 -1.0"""
    # 转换射线到盒子的局部坐标系
    local_start, local_direction = _transform_ray_to_local(ray_start, ray_direction, center, rotation)

    # 处理局部坐标系中的射线方向为零的情况
    inv_dir = ti.math.vec3(
        1.0 / (local_direction.x if ti.abs(local_direction.x) > 1e-6 else 1e10),
        1.0 / (local_direction.y if ti.abs(local_direction.y) > 1e-6 else 1e10),
        1.0 / (local_direction.z if ti.abs(local_direction.z) > 1e-6 else 1e10)
    )

    t_min = -1e10
    t_max = 1e10

    # x 轴
    t1 = (-size.x - local_start.x) * inv_dir.x
    t2 = (size.x - local_start.x) * inv_dir.x
    t_min = ti.max(t_min, ti.min(t1, t2))
    t_max = ti.min(t_max, ti.max(t1, t2))

    # y 轴
    t1 = (-size.y - local_start.y) * inv_dir.y
    t2 = (size.y - local_start.y) * inv_dir.y
    t_min = ti.max(t_min, ti.min(t1, t2))
    t_max = ti.min(t_max, ti.max(t1, t2))

    # z 轴
    t1 = (-size.z - local_start.z) * inv_dir.z
    t2 = (size.z - local_start.z) * inv_dir.z
    t_min = ti.max(t_min, ti.min(t1, t2))
    t_max = ti.min(t_max, ti.max(t1, t2))

    t_ret = -1.0
    if t_max >= t_min and t_max >= 0:
        t = t_min if t_min >= 0 else t_max
        if t >= 0:
            t_ret = t
    return t_ret

@ti.func
def ray_cylinder_distance(ray_start, ray_direction, center, size, rotation):
    """返回射线与圆柱体命中距离 t；未命中返回 -1.0
    size.x: 半径，size.z: 半高
    """
    # 转换射线到圆柱体的局部坐标系
    local_start, local_direction = _transform_ray_to_local(ray_start, ray_direction, center, rotation)

    radius = size[0]
    half_height = size[2]

    # 在局部坐标系中，圆柱体的中心轴与 z 轴平行
    # 仅考虑 xy 平面上的方向分量
    ray_dir_xy = ti.math.vec2(local_direction.x, local_direction.y)
    oc_xy = ti.math.vec2(local_start.x, local_start.y)

    # 解二次方程 at² + bt + c = 0
    a = ray_dir_xy.dot(ray_dir_xy)

    t_ret = -1.0
    # 如果 a 很小，射线几乎与 z 轴平行
    if a < 1e-6:
        # 检查射线是否在圆柱体内部
        if oc_xy.norm() <= radius:
            # 计算与顶部或底部平面的交点
            t1 = (half_height - local_start.z) / local_direction.z
            t2 = (-half_height - local_start.z) / local_direction.z

            # 选择最小的正 t 值
            t = -1.0
            if t1 >= 0 and (t2 < 0 or t1 < t2):
                t = t1
            elif t2 >= 0:
                t = t2

            if t >= 0:
                t_ret = t
    else:
        # 标准的圆柱体-射线相交测试（侧面）
        b = 2.0 * oc_xy.dot(ray_dir_xy)
        c = oc_xy.dot(oc_xy) - radius * radius

        discriminant = b * b - 4 * a * c

        if discriminant >= 0:
            sqrt_disc = ti.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)

            # 选择最小的正 t 值
            t = -1.0
            if t1 >= 0:
                t = t1
            elif t2 >= 0:
                t = t2

            # 检查交点是否在圆柱体高度范围内
            if t >= 0:
                local_hit = local_start + t * local_direction
                if ti.abs(local_hit.z) <= half_height:
                    t_ret = t
                else:
                    # 侧面交点不在高度范围内，检查端盖
                    cap_t = -1.0
                    if local_direction.z < 0 and local_start.z > half_height:
                        cap_t = (half_height - local_start.z) / local_direction.z
                    elif local_direction.z > 0 and local_start.z < -half_height:
                        cap_t = (-half_height - local_start.z) / local_direction.z

                    if cap_t >= 0:
                        local_hit = local_start + cap_t * local_direction
                        cap_xy = ti.math.vec2(local_hit.x, local_hit.y)
                        if cap_xy.norm() <= radius:
                            t_ret = cap_t

    return t_ret

@ti.func
def ray_ellipsoid_distance(ray_start, ray_direction, center, size, rotation):
    """返回射线与椭球体命中距离 t；未命中返回 -1.0"""
    # 转换射线到椭球体的局部坐标系
    local_start, local_direction = _transform_ray_to_local(ray_start, ray_direction, center, rotation)

    # 将问题转换为单位球相交，通过缩放空间
    inv_size = ti.math.vec3(1.0 / size.x, 1.0 / size.y, 1.0 / size.z)

    # 缩放局部射线（不要归一化方向）
    scaled_start = ti.math.vec3(
        local_start.x * inv_size.x,
        local_start.y * inv_size.y,
        local_start.z * inv_size.z
    )
    scaled_dir = ti.math.vec3(
        local_direction.x * inv_size.x,
        local_direction.y * inv_size.y,
        local_direction.z * inv_size.z
    )

    # 解二次方程 at² + bt + c = 0
    a = scaled_dir.dot(scaled_dir)
    b = 2.0 * scaled_start.dot(scaled_dir)
    c = scaled_start.dot(scaled_start) - 1.0

    t_ret = -1.0
    discriminant = b * b - 4 * a * c
    if discriminant >= 0:
        t1 = (-b - ti.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + ti.sqrt(discriminant)) / (2.0 * a)
        t = t1 if t1 >= 0 else t2
        if t >= 0:
            t_ret = t

    return t_ret

@ti.func
def ray_capsule_distance(ray_start, ray_direction, center, size, rotation):
    """返回射线与胶囊体命中距离 t；未命中返回 -1.0
    在 MuJoCo 中: size.x 为半径，size.z 为圆柱部分半高
    """
    # 转换射线到胶囊体的局部坐标系
    local_start, local_direction = _transform_ray_to_local(ray_start, ray_direction, center, rotation)

    radius = size[0]
    half_height = size[2]

    # 胶囊体两个半球的中心（局部坐标）
    sphere1_center = ti.math.vec3(0.0, 0.0, half_height)
    sphere2_center = ti.math.vec3(0.0, 0.0, -half_height)

    # 为圆柱部分构造 size（局部坐标，单位旋转与零平移）
    cylinder_size = ti.math.vec3(radius, radius, half_height)
    identity_mat = ti.Matrix.identity(ti.f32, 3)

    # 先测试圆柱体侧面与端盖
    cylinder_t = ray_cylinder_distance(
        local_start, local_direction,
        ti.math.vec3(0.0, 0.0, 0.0), cylinder_size, identity_mat
    )

    # 初始化最小距离
    min_t = 1e10
    if cylinder_t >= 0 and cylinder_t < min_t:
        min_t = cylinder_t

    # 再测试两个半球（完整球体相交后再用 z 方向裁剪半球）
    sphere_size = ti.math.vec3(radius, radius, radius)

    # 上半球
    t1 = ray_sphere_distance(local_start, local_direction, sphere1_center, sphere_size, identity_mat)
    if t1 >= 0 and t1 < min_t:
        local_hit = local_start + t1 * local_direction
        if (local_hit.z - sphere1_center.z) >= 0:  # 半球裁剪
            min_t = t1

    # 下半球
    t2 = ray_sphere_distance(local_start, local_direction, sphere2_center, sphere_size, identity_mat)
    if t2 >= 0 and t2 < min_t:
        local_hit = local_start + t2 * local_direction
        if (local_hit.z - sphere2_center.z) <= 0:
            min_t = t2

    return min_t if min_t < 1e10 else -1.0