"""JoTouch parametric dimensions (mm). X=proximal->distal, Y=ulnar->radial, Z=palmar->dorsal."""
GM3506 = dict(dia=40.0, height=20.0, bore=8.0, mount_pcd=25.0, mount_hole=3.2, weight_g=80)
AS5600_magnet = dict(dia=6.0, height=2.5)
PIN = 3.0
PIN_HOLE = 3.2
M3_HOLE = 3.2
M3_BOSS = 6.0
FINGERS = {
    "index":  dict(prox=39.8, mid=22.4, dist=15.8, w=19.0, t=17.0, y=+25.5),
    "middle": dict(prox=44.6, mid=26.3, dist=17.4, w=20.0, t=18.0, y=+8.5),
    "ring":   dict(prox=41.4, mid=25.7, dist=17.3, w=19.0, t=17.0, y=-8.5),
    "little": dict(prox=32.7, mid=18.1, dist=15.8, w=16.0, t=15.0, y=-24.0),
}
THUMB = dict(prox=31.6, dist=21.7, w=21.0, t=20.0)
PALM = dict(length=97.0, width=84.0, thick=30.0, fillet=10.0, wall=3.0)
WRIST = dict(dia=46.0, length=55.0, yoke_gap=34.0, pivot_dia=3.2)
TESS = 0.25
