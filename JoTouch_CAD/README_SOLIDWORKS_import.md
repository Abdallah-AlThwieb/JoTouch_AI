# JoTouch Prosthetic Hand + Wrist — CAD Package

Parametric solid model of the JoTouch upper-limb prosthetic, built as real
B-rep solids and exported to **STEP (AP214)** so it opens in SOLIDWORKS as a
fully editable model — not a dead mesh. Sized for an **average adult male
right hand** (50th percentile anthropometry).

## What's in this package

```
step/
  JoTouch_hand_assembly.step   full hand + wrist (22 solids) — import this first
  JoTouch_palm.step            palm chassis (motor bays, MCP bosses, wrist spigot)
  JoTouch_index.step           index finger (3 phalanges + four-bar coupler)
  JoTouch_middle.step          middle finger
  JoTouch_ring.step            ring finger
  JoTouch_little.step          little finger
  JoTouch_thumb.step           opposable thumb (CMC base + 2 phalanges)
  JoTouch_wrist.step           2-DOF wrist (pronation drum + flexion yoke)
renders/                       reference images (iso / top / palmar / part views)
jotouch_params.py              every dimension as an editable parameter
jotouch_lib.py                 the parametric build code (regenerate any size)
```

## Open in SOLIDWORKS

1. **File ▸ Open**, set file type to **STEP (*.step; *.stp)**, choose
   `JoTouch_hand_assembly.step`.
2. In the import dialog leave **"Import as a part / assembly"** — SOLIDWORKS
   asks; choose **Assembly** to get each part as its own component, or **Part**
   (multibody) for a single file. Assembly is recommended for mating motors.
3. When prompted **"Run Import Diagnostics?"** click **Yes** — the solids are
   watertight, so it should report no faults.
4. Units are **millimetres**. If SOLIDWORKS opens in inches, set
   Document Properties ▸ Units ▸ MMGS.

To open one part at a time (e.g. to redesign a single finger), open its
individual `.step` file instead.

## Coordinate frame

- **X** = proximal → distal (wrist → fingertips)
- **Y** = ulnar → radial (little-finger side → thumb side)
- **Z** = palmar → dorsal (palm → back of hand); fingers flex toward −Z
- Joint pivot axes run along **Y**. The MCP knuckle line sits at **X = 0**.

## Key dimensions (mm)

| Item | Value |
|---|---|
| Overall length (incl. wrist drum) | ~252 |
| Hand width across MCPs | ~84 |
| Palm: L × W × thick | 97 × 84 × 30 |
| Middle finger phalanges (P/M/D) | 44.6 / 26.3 / 17.4 |
| Wrist drum dia × length | 46 × 55 |
| Pivot pins | Ø3 (Ø3.2 clearance holes) |

Structural volume ≈ 378 cm³ (≈ 469 g if printed 100% solid PLA; real infill
makes it much lighter). Seven GM3506 motors add ~560 g.

## Mechanism notes

- **Fingers** use the four-bar linkage approach from the design report (not
  tendons): each has proximal/middle/distal phalanges with Ø3.2 pivot holes and
  a coupler link that couples middle-phalanx flexion to the driven proximal joint.
- **Palm** carries Ø40 cavities that are the **GM3506 motor bays** plus MCP
  pivot bosses on the distal edge and a wrist spigot on the proximal face.
  (Because a Ø40 motor is larger than the 30 mm palm depth, final packaging
  staggers some motors into the wrist/forearm module — the bays mark their drive
  positions.)
- **Wrist** gives 2 DOF: the drum rotates for pronation/supination and houses a
  GM3506; the yoke (clevis) pivots on a Y-axis pin for flexion/extension.

## Re-generating at a different size

Edit `jotouch_params.py` (e.g. change `FINGERS[...]["prox"]` or `PALM`), then
run `python jotouch_lib.py`-driven build — every part rescales parametrically,
so a custom patient hand is a parameter change, not a remodel.
