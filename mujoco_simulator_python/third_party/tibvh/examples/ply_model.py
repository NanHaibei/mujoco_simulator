import numpy as np

from gaussian_model import RGB2SH

def xyz2ply(grids, output_ply):
    N = grids.shape[0]
    dtype = np.dtype([
        ('x','<f4'),('y','<f4'),('z','<f4'),
        ('density','<f4'),('red','u1'),('green','u1'),('blue','u1')
    ])
    arr = np.empty(N, dtype=dtype)
    arr['x'] = grids[:,0]; arr['y'] = grids[:,1]; arr['z'] = grids[:,2]
    arr['density'] = grids[:,3] if grids.shape[1]>3 else 0.0
    arr['red'] = 0; arr['green'] = 0; arr['blue'] = 255

    with open(output_ply, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {N}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float density\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        arr.tofile(f)
    print("Saved binary ply:", output_ply)

def xyz2supersplat(grids, output_ply, base_scale, colors=None):
    positions = grids[:, :3].astype(np.float32)
    N = int(positions.shape[0])

    if colors is None:
        _rgb = np.random.random((N, 3)).astype(np.float32)
    else:
        _rgb = np.asarray(colors, dtype=np.float32)
        if _rgb.shape[0] != N:
            raise ValueError('colors length does not match number of points')
    shs = RGB2SH(_rgb)

    opa_ = np.ones(N, dtype=np.float32)
    opacities = -np.log(1e-6 + 1.0 / np.clip(opa_, 1e-6, 1.0) - 1.0).astype(np.float32)
    log_scale = np.full(N, np.log(base_scale / 3.), dtype=np.float32)

    rot = np.zeros((N, 4), dtype=np.float32)
    rot[:, 0] = 1.0

    ply_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"), ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ])

    rec = np.empty(N, dtype=ply_dtype)
    rec["x"], rec["y"], rec["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    rec["f_dc_0"], rec["f_dc_1"], rec["f_dc_2"] = shs[:, 0], shs[:, 1], shs[:, 2]
    rec["opacity"] = opacities
    rec["scale_0"], rec["scale_1"], rec["scale_2"] = log_scale, log_scale, log_scale
    rec["rot_0"], rec["rot_1"], rec["rot_2"], rec["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]

    # 写出二进制little-endian PLY：header + 一次性二进制块
    with open(output_ply, "wb") as f:
        header = []
        header.append("ply\n")
        header.append("format binary_little_endian 1.0\n")
        header.append(f"element vertex {N}\n")
        header.append("property float x\n")
        header.append("property float y\n")
        header.append("property float z\n")
        header.append("property float f_dc_0\n")
        header.append("property float f_dc_1\n")
        header.append("property float f_dc_2\n")
        header.append("property float opacity\n")
        header.append("property float scale_0\n")
        header.append("property float scale_1\n")
        header.append("property float scale_2\n")
        header.append("property float rot_0\n")
        header.append("property float rot_1\n")
        header.append("property float rot_2\n")
        header.append("property float rot_3\n")
        header.append("end_header\n")
        f.write("".join(header).encode("ascii"))

        # 结构化数组已为小端，直接写出
        rec.tofile(f)

    print(f"Saved binary supersplat PLY to: {output_ply}")