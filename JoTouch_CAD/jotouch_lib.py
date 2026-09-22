"""JoTouch geometry library (CadQuery). Part builders + headless renderer."""
import cadquery as cq
from cadquery import exporters
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from jotouch_params import PIN_HOLE, TESS


def cyl_Y(r, y0, h, x=0.0, z=0.0):
    return cq.Solid.makeCylinder(r, h, cq.Vector(x, y0, z), cq.Vector(0, 1, 0))


def cyl_X(r, x0, h, y=0.0, z=0.0):
    return cq.Solid.makeCylinder(r, h, cq.Vector(x0, y, z), cq.Vector(1, 0, 0))


def phalanx(L, w0, w1, t0, t1, pin=PIN_HOLE, tip=False):
    r0, r1 = w0 / 2.0, w1 / 2.0
    h0, h1 = t0 / 2.0, t1 / 2.0
    body = (cq.Workplane("YZ").ellipse(r0, h0)
            .workplane(offset=L).ellipse(r1, h1).loft(combine=True))
    res = body.union(cq.Workplane(obj=cyl_Y(r0, -t0 / 2.0, t0)))
    if tip:
        res = res.union(cq.Workplane(obj=cq.Solid.makeSphere(r1, cq.Vector(L, 0, 0))))
    else:
        res = res.union(cq.Workplane(obj=cyl_Y(r1, -t1 / 2.0, t1, x=L)))
    res = res.cut(cq.Workplane(obj=cyl_Y(pin / 2.0, -t0, 2 * t0)))
    if not tip:
        res = res.cut(cq.Workplane(obj=cyl_Y(pin / 2.0, -t1, 2 * t1, x=L)))
    return res


def link_bar(L, w=6.0, t=3.0, hole=PIN_HOLE):
    b = cq.Workplane("XY").rect(L, w).extrude(t).translate((L / 2.0, 0, 0))
    d0 = cq.Workplane("XY").circle(w / 2.0).extrude(t)
    d1 = cq.Workplane("XY").circle(w / 2.0).extrude(t).translate((L, 0, 0))
    bar = b.union(d0).union(d1)
    return bar.faces(">Z").workplane().pushPoints([(0, 0), (L, 0)]).hole(hole)


def finger(name, d):
    prox, mid, dist = d["prox"], d["mid"], d["dist"]
    w, t = d["w"], d["t"]
    gap = 2.5
    parts = {}
    x = 0.0
    parts["proximal"] = phalanx(prox, w, w * 0.86, t, t * 0.88).translate((x, 0, 0))
    x += prox + gap
    parts["middle"] = phalanx(mid, w * 0.86, w * 0.74, t * 0.88, t * 0.78).translate((x, 0, 0))
    x += mid + gap
    parts["distal"] = phalanx(dist, w * 0.74, w * 0.6, t * 0.78, t * 0.62, tip=True).translate((x, 0, 0))
    coupler = (link_bar(prox + gap, w=6.0, t=3.0)
               .translate((0, 0, t / 2.0 + 1.5)).translate((0, w / 2.0 + 3.0, 0)))
    parts["coupler"] = coupler
    return parts


def palm_chassis(P, fingers, gm):
    L, W, T, fil, wall = P["length"], P["width"], P["thick"], P["fillet"], P["wall"]
    body = (cq.Workplane("XY").box(L, W, T, centered=(False, True, True)).translate((-L, 0, 0)))
    body = body.edges("|Z").fillet(fil)
    body = body.faces(">Z").edges(">X").chamfer(6.0)
    body = body.faces("<X").shell(-wall)
    for nm, d in fingers.items():
        y = d["y"]
        body = body.union(cq.Workplane(obj=cyl_Y(d["t"] * 0.55, y - d["w"] * 0.5 - 2, d["w"] + 4, x=0, z=0)))
        body = body.cut(cq.Workplane(obj=cyl_Y(2.6, y - d["w"], 2 * d["w"], x=0, z=0)))
        body = body.cut(cq.Workplane(obj=cyl_Y(gm["dia"] / 2.0, y - gm["height"] / 2.0, gm["height"],
                                               x=-22 - (abs(y) * 0.15), z=0)))
    body = body.union(cq.Workplane(obj=cyl_X(16.0, -L - 12, 12.0)))
    body = body.cut(cq.Workplane(obj=cyl_X(9.0, -L - 12, 40.0)))
    return body


def thumb(TH):
    prox, dist, w, t = TH["prox"], TH["dist"], TH["w"], TH["t"]
    parts = {}
    parts["cmc"] = (cq.Workplane("XY").box(22, w + 4, t + 2).edges("|Z").fillet(6.0)).translate((-13, 0, 0))
    parts["proximal"] = phalanx(prox, w, w * 0.85, t, t * 0.85)
    parts["distal"] = phalanx(dist, w * 0.85, w * 0.66, t * 0.85, t * 0.66, tip=True).translate((prox + 2.5, 0, 0))
    return parts


def wrist(WR, gm, x0=-97.0):
    dia, Lx, gap = WR["dia"], WR["length"], WR["yoke_gap"]
    yoke_t = 6.0
    parts = {}
    drum = cq.Workplane(obj=cyl_X(dia / 2.0, x0 - Lx, Lx))
    drum = drum.cut(cq.Workplane(obj=cyl_X(gm["dia"] / 2.0 + 1, x0 - gm["height"] - 4, gm["height"] + 2)))
    drum = drum.cut(cq.Workplane(obj=cyl_X(7.0, x0 - Lx - 2, Lx + 4)))
    parts["pronation_drum"] = drum
    arm_top = (cq.Workplane("XY").box(26, yoke_t, dia).translate((x0 + 13, gap / 2.0 + yoke_t / 2.0, 0)))
    arm_bot = (cq.Workplane("XY").box(26, yoke_t, dia).translate((x0 + 13, -(gap / 2.0 + yoke_t / 2.0), 0)))
    web = (cq.Workplane("XY").box(8, gap + 2 * yoke_t, dia).translate((x0 + 2, 0, 0)))
    yoke = arm_top.union(arm_bot).union(web)
    yoke = yoke.cut(cq.Workplane(obj=cyl_Y(2.6, -(gap / 2.0 + yoke_t + 1), gap + 2 * yoke_t + 2, x=x0 + 22, z=0)))
    parts["flexion_yoke"] = yoke
    return parts


def to_compound(parts):
    sols = [(p.val() if isinstance(p, cq.Workplane) else p) for p in parts.values()]
    return cq.Compound.makeCompound(sols)


def export_step(obj, path):
    if isinstance(obj, dict):
        obj = to_compound(obj)
    exporters.export(obj, path)


def _mesh(solid, tol=TESS):
    v, f = solid.tessellate(tol)
    return np.array([[p.x, p.y, p.z] for p in v]), f


def render(objs, path, elev=22, azim=-60, title=None, figsize=(6.5, 5.5)):
    palette = ["#6fa8dc", "#e6a23c", "#7bc67b", "#d96f6f", "#b18cd9",
               "#5bc8c8", "#d98cc0", "#9aa0a6", "#c7b15b", "#8fce00"]
    if isinstance(objs, dict):
        items = [(p, palette[i % len(palette)]) for i, p in enumerate(objs.values())]
    elif isinstance(objs, list):
        items = objs
    else:
        items = [(objs, palette[0])]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    allV = []
    for obj, color in items:
        v = obj.val() if isinstance(obj, cq.Workplane) else obj
        V, F = _mesh(v)
        allV.append(V)
        pc = Poly3DCollection([V[list(t)] for t in F], facecolor=color, edgecolor="none", alpha=1.0)
        pc.set_linewidth(0)
        ax.add_collection3d(pc)
    A = np.vstack(allV)
    mn, mx = A.min(0), A.max(0)
    ctr = (mn + mx) / 2
    rng = (mx - mn).max() / 2 * 1.05
    for a, c in zip("xyz", ctr):
        getattr(ax, f"set_{a}lim")(c - rng, c + rng)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=115)
    plt.close(fig)
    return path
