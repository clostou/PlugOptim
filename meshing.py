"""



"""


import numpy as np
import gmsh
import nozzleMap


def _profile_to_msh(profile_points, partition_tag, lc=0.1,
                   extrude_wall=True, recombine=True, order2=False, save_path=None):
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("nozzle")
    # 创建计算域几何
    pts = []
    for point in profile_points[partition_tag['inlet'][0]: partition_tag['inlet'][1], : ]:
        pts.append(gmsh.model.geo.addPoint(*point, .0, lc))
    for point in profile_points[partition_tag['wall'][0]: partition_tag['wall'][1] + 1, : ]:
        pts.append(gmsh.model.geo.addPoint(*point, .0))
    for point in profile_points[partition_tag['outlet'][0] + 1: partition_tag['axis'][1], : ]:
        pts.append(gmsh.model.geo.addPoint(*point, .0, lc))
    pts.append(pts[0])
    curve = []
    # 添加壁面边界层
    if extrude_wall:
        for i in range(partition_tag['wall'][0], partition_tag['wall'][1]):
            curve.append(gmsh.model.geo.addLine(pts[i], pts[i + 1]))
        n = 10    # number of layers
        r = 1.2    # ratio
        d = [1.7e-3]    # thickness of first layer
        for i in range(1, n):
            d.append(d[-1] + d[0] * r**i)
        #d = np.logspace(-3, -1, 5)
        print("height of boundary layer: %.2e" % d[-1])
        gmsh.option.setNumber('Geometry.ExtrudeReturnLateralEntities', 0)
        e = gmsh.model.geo.extrudeBoundaryLayer([(1, c) for c in curve], np.linspace(1, 1, n), d, True)
        top_ent = [s for s in e if s[0] == 1]
        top_cur = [s[1] for s in top_ent]
        gmsh.model.geo.synchronize()
        bnd_ent = gmsh.model.getBoundary(top_ent)
        bnd_pts = [s[1] for s in bnd_ent]
        c_left = gmsh.model.geo.addLine(bnd_pts[0], pts[partition_tag['wall'][0]])
        c_right = gmsh.model.geo.addLine(pts[partition_tag['wall'][1]], bnd_pts[1])
        corner_pts = [pts[partition_tag['inlet'][0]],
                      *bnd_pts,
                      pts[partition_tag['outlet'][1]]]
        interior_cur = [gmsh.model.geo.addLine(corner_pts[0], corner_pts[1]),
                        *top_cur,
                        gmsh.model.geo.addLine(corner_pts[2], corner_pts[3]),
                        gmsh.model.geo.addLine(corner_pts[3], corner_pts[0])]
        cl = gmsh.model.geo.addCurveLoop(interior_cur)
        s = gmsh.model.geo.addPlaneSurface([cl])
        if recombine:
            outlet_n = int((profile_points[partition_tag['outlet'][0], 1] -
                            profile_points[partition_tag['outlet'][1], 1]) / lc)
            gmsh.model.geo.mesh.setTransfiniteCurve(interior_cur[0], outlet_n)
            gmsh.model.geo.mesh.setTransfiniteCurve(interior_cur[-2], outlet_n)
            gmsh.model.geo.mesh.setTransfiniteCurve(interior_cur[-1], len(top_cur) + 1)
            gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left", corner_pts)
            gmsh.model.geo.mesh.setRecombine(2, s)
            gmsh.option.setNumber("Mesh.Smoothing", 100)
        gmsh.model.geo.synchronize()
        # 创建物理组命名
        gmsh.model.addPhysicalGroup(1, [interior_cur[0], c_left], name='inlet')
        gmsh.model.addPhysicalGroup(1, curve, name='wall')
        gmsh.model.addPhysicalGroup(1, [c_right, interior_cur[-2]], name='outlet')
        gmsh.model.addPhysicalGroup(1, [interior_cur[-1]], name='axis')
    else:
        curve = []
        for i in range(len(pts) - 1):
            curve.append(gmsh.model.geo.addLine(pts[i], pts[i + 1]))
        cl = gmsh.model.geo.addCurveLoop(curve)
        s = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()
        # 创建物理组命名
        for name, (ind1, ind2) in partition_tag.items():
            gmsh.model.addPhysicalGroup(1, curve[ind1: ind2], name=name)
    # 网格生成
    gmsh.model.mesh.generate(2)
    # 2nd order + fast curving of the boundary layer + optimization
    if order2:    # 若二维网格导出为bdf格式并用fluent读取，则应当关闭该选项
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.optimize('HighOrderFastCurving')
        gmsh.model.mesh.optimize('HighOrder')
    if save_path:
        gmsh.option.setNumber('Mesh.Format', 31)
        gmsh.option.setNumber('Mesh.LabelType', 2)    # 导出设置Element Tag: Physical entity
        gmsh.write(save_path)
    gmsh.fltk.run()
    gmsh.finalize()


def _profile_to_msh2(profile_points, partition_tag, lc=0.1, ratio=1.05, order2=False, save_path='nozzle.bdf'):
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("nozzle")
    # 创建计算域几何
    pts = []
    for point in profile_points[partition_tag['inlet'][0]: partition_tag['inlet'][1], :]:
        pts.append(gmsh.model.geo.addPoint(*point, .0, lc))
    for point in profile_points[partition_tag['wall'][0]: partition_tag['wall'][1] + 1, :]:
        pts.append(gmsh.model.geo.addPoint(*point, .0))
    for point in profile_points[partition_tag['outlet'][0] + 1: partition_tag['axis'][1], :]:
        pts.append(gmsh.model.geo.addPoint(*point, .0, lc))
    pts.append(pts[0])
    corner_pts = [pts[partition_tag['inlet'][0]],
                  pts[partition_tag['wall'][0]],
                  pts[partition_tag['outlet'][0]],
                  pts[partition_tag['axis'][0]]]
    curve = [gmsh.model.geo.addLine(corner_pts[0], corner_pts[1])]
    for i in range(partition_tag['wall'][0], partition_tag['wall'][1]):
        curve.append(gmsh.model.geo.addLine(pts[i], pts[i + 1]))
    curve.append(gmsh.model.geo.addLine(corner_pts[2], corner_pts[3]))
    curve.append(gmsh.model.geo.addLine(corner_pts[3], corner_pts[0]))
    cl = gmsh.model.geo.addCurveLoop(curve)
    s = gmsh.model.geo.addPlaneSurface([cl])
    outlet_n = int((profile_points[partition_tag['outlet'][0], 1] -
                    profile_points[partition_tag['outlet'][1], 1]) / lc)
    gmsh.model.geo.mesh.setTransfiniteCurve(curve[0], outlet_n, "Progression", - ratio)
    gmsh.model.geo.mesh.setTransfiniteCurve(curve[-2], outlet_n, "Progression", ratio)
    gmsh.model.geo.mesh.setTransfiniteCurve(curve[-1], partition_tag['wall'][1] - partition_tag['wall'][0] + 1)
    gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left", corner_pts)
    gmsh.model.geo.mesh.setRecombine(2, s)
    gmsh.option.setNumber("Mesh.Smoothing", 10)
    #ov = gmsh.model.geo.extrude([(2, s)], 0, 0, lc, [1], recombine=True)
    gmsh.model.geo.synchronize()
    # 创建物理组命名
    gmsh.model.addPhysicalGroup(2, [s], name='nozzle')
    gmsh.model.addPhysicalGroup(1, curve[0: 1], name='inlet')
    gmsh.model.addPhysicalGroup(1, curve[1: -2], name='wall')
    gmsh.model.addPhysicalGroup(1, curve[-2: -1], name='outlet')
    gmsh.model.addPhysicalGroup(1, curve[-1: ], name='axis')
    # 网格生成
    gmsh.model.mesh.generate(2)
    # 2nd order + fast curving of the boundary layer + optimization
    if order2:
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.optimize('HighOrderFastCurving')
        gmsh.model.mesh.optimize('HighOrder')
    if save_path:
        gmsh.option.setNumber('Mesh.LabelType', 2)
        gmsh.write(save_path)
    gmsh.fltk.run()
    gmsh.finalize()


def profile_to_msh(profile_points, partition_tag, lc=0.1, ratio=1.05, order2=False, save_path='nozzle.msh'):
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("nozzle")
    # 创建计算域几何
    pts = []
    for point in profile_points[partition_tag['inlet'][0]: partition_tag['inlet'][1], :]:
        pts.append(gmsh.model.geo.addPoint(*point, .0, lc))
    for point in profile_points[partition_tag['wall'][0]: partition_tag['wall'][1] + 1, :]:
        pts.append(gmsh.model.geo.addPoint(*point, .0))
    for point in profile_points[partition_tag['outlet'][0] + 1: partition_tag['axis'][1], :]:
        pts.append(gmsh.model.geo.addPoint(*point, .0, lc))
    gmsh.model.geo.rotate([(0, p) for p in pts], 0, 0, 0, 1, 0, 0, -np.pi * 1.5 / 180)
    pts.append(pts[0])
    corner_pts = [pts[partition_tag['inlet'][0]],
                  pts[partition_tag['wall'][0]],
                  pts[partition_tag['outlet'][0]],
                  pts[partition_tag['axis'][0]]]
    curve = [gmsh.model.geo.addLine(corner_pts[0], corner_pts[1])]
    for i in range(partition_tag['wall'][0], partition_tag['wall'][1]):
        curve.append(gmsh.model.geo.addLine(pts[i], pts[i + 1]))
    curve.append(gmsh.model.geo.addLine(corner_pts[2], corner_pts[3]))
    curve.append(gmsh.model.geo.addLine(corner_pts[3], corner_pts[0]))
    cl = gmsh.model.geo.addCurveLoop(curve)
    s = gmsh.model.geo.addPlaneSurface([cl])
    outlet_n = int((profile_points[partition_tag['outlet'][0], 1] -
                    profile_points[partition_tag['outlet'][1], 1]) / lc)
    gmsh.model.geo.mesh.setTransfiniteCurve(curve[0], outlet_n, "Progression", - ratio)
    gmsh.model.geo.mesh.setTransfiniteCurve(curve[-2], outlet_n, "Progression", ratio)
    gmsh.model.geo.mesh.setTransfiniteCurve(curve[-1], partition_tag['wall'][1] - partition_tag['wall'][0] + 1)
    gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left", corner_pts)
    gmsh.model.geo.mesh.setRecombine(2, s)
    gmsh.option.setNumber("Mesh.Smoothing", 10)
    #ov = gmsh.model.geo.extrude([(2, s)], 0, 0, lc, [1], recombine=True)
    ov = gmsh.model.geo.revolve([(2, s)], 0, 0, 0, 1, 0, 0, np.pi * 3 / 180, [1], recombine=True)
    gmsh.model.geo.synchronize()
    # 创建物理组命名
    gmsh.model.addPhysicalGroup(3, [ov[1][1]], name='nozzle')
    gmsh.model.addPhysicalGroup(2, [s], name='asym1')
    gmsh.model.addPhysicalGroup(2, [ov[0][1]], name='asym2')
    gmsh.model.addPhysicalGroup(2, [ov[2][1]], name='inlet')
    gmsh.model.addPhysicalGroup(2, [s[1] for s in ov[3: -1]], name='profile')
    gmsh.model.addPhysicalGroup(2, [ov[-1][1]], name='outlet')
    # 网格生成
    gmsh.model.mesh.generate(3)
    # 2nd order + fast curving of the boundary layer + optimization
    if order2:
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.optimize('HighOrderFastCurving')
        gmsh.model.mesh.optimize('HighOrder')
    if save_path:
        gmsh.option.setNumber('Mesh.Format', 1)
        gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
        gmsh.write(save_path)
    gmsh.fltk.run()
    gmsh.finalize()


if __name__ == '__main__':
    net_output = np.array([[0.8, 0.3, 0.3],
                           [0.5, 0.2, 0.3],
                           [0.5, 0.1, 0.3]])
    points = nozzleMap.point_transform(net_output)
    profile, control, tag = nozzleMap.bezier_profile(points, sub_n=50)
    #profile_to_msh(profile, tag, lc=0.02, ratio=1.03)
    _profile_to_msh(profile, tag, lc=0.02, extrude_wall=False, recombine=True, order2=False, save_path=None)


