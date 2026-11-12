#!/usr/bin/env python3
# coding: utf-8
"""
weld_path_generator.py
======================
TCP weld-path generator for a multi-jig robotic welding cell.

Several jigs sit on a semicircular table around the robot. Each jig holds one
part type. The path of a part is defined once, in the local frame of its jig,
and is then placed on every jig with a homogeneous transform:

    P_global = Rot_z(alpha) . Trans(0, R_table, 0) . P_local

The script validates the weld heat input, builds the global waypoints, and
writes a teach-pendant style program listing (text + optional PNG preview).

NOTE: every dimension, angle and welding parameter in this file is an
ILLUSTRATIVE value. None of them is production data.

Usage:
    python weld_path_generator.py --preset 5jig
    python weld_path_generator.py --preset 4jig --outdir examples --no-png
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime

import numpy as np

# ============================================================================
# 1. PART GEOMETRY  (illustrative values, mm)
# ============================================================================
# --- A-frame bracket: two legs joined at the apex, with a crossbar ---
L_AF = 300.0    # leg length
C_AF = 140.0    # crossbar length
G_AF = 30.0     # apex gap
F_AF = 220.0    # distance between the two feet
f_AF = (C_AF - G_AF) / (F_AF - G_AF)   # position of the crossbar joint on a leg

BAR_RADIUS = 5.0        # radius of the round bar
STANDOFF = 2.0          # torch stand-off
FILLET_WELD_LEN = 25.0  # length of each linear fillet
WRAP_DEG = 300.0        # total wrap angle of the circular weld on the joint

# --- Ring bracket: one full circular fillet ---
R_RING = 25.0           # weld circle radius

# --- Box bracket: two parallel linear welds ---
LEN_BOX = 50.0          # weld line length
GAP_BOX = 30.0          # gap between the two lines

# --- Table layout ---
R_TABLE = 400.0         # radial offset of every jig origin from the table center

# ============================================================================
# 2. WELDING PARAMETERS  (illustrative values)
# ============================================================================
#            current [A], voltage [V], travel speed [cm/min]
PROCESS = {
    "linear":   {"I": 145.0, "V": 20.0, "v": 42.0},
    "circular": {"I": 115.0, "V": 19.5, "v": 33.0},
}

ETA = 0.80          # GMAW arc efficiency
Q_DESIGN = 0.350    # target heat input [kJ/mm]
Q_TOLERANCE = 0.10  # +/- 10 %

# ============================================================================
# 3. JIG LAYOUT PRESETS  (alpha = jig angle on the table, deg)
# ============================================================================
PRESETS = {
    "5jig": {
        "program": 1,
        "jigs": [
            {"alpha": -50.0, "type": "aframe"},
            {"alpha": -25.0, "type": "aframe"},
            {"alpha":   0.0, "type": "box"},
            {"alpha": +25.0, "type": "ring"},
            {"alpha": +50.0, "type": "ring"},
        ],
    },
    "4jig": {
        "program": 2,
        "jigs": [
            {"alpha": -60.0, "type": "aframe"},
            {"alpha": -20.0, "type": "aframe"},
            {"alpha": +20.0, "type": "ring"},
            {"alpha": +60.0, "type": "ring"},
        ],
    },
}

TYPE_LABEL = {"aframe": "A-frame", "box": "Box 2 lines", "ring": "Ring circular"}
TOOL = "A1 T1"


# ============================================================================
# 4. A-FRAME DERIVED GEOMETRY
# ============================================================================
def aframe_height(L, F):
    """Height of the apex above the feet line."""
    return math.sqrt(L * L - (F / 2) ** 2)


def aframe_joint_center(F, h, f):
    """Crossbar joint on the left leg, at fraction f from the apex."""
    apex = np.array([0.0, h, 0.0])
    foot_l = np.array([-F / 2, 0.0, 0.0])
    return apex + f * (foot_l - apex)


def unit_vector(a, b):
    v = b - a
    return v / np.linalg.norm(v)


def perpendicular(u, side):
    """In-plane normal of a leg; side is 'front' or 'back'."""
    sign = 1.0 if side == "front" else -1.0
    return np.array([-sign * u[1], sign * u[0], 0.0])


# ============================================================================
# 5. PART DEFINITIONS  (local waypoints + weld list)
# ============================================================================
# Each part returns:
#   points : list of (xyz, label) in the local jig frame
#   welds  : list of {"process", "start", "moves"}; indices refer to `points`.
#            One weld = one arc-on ... arc-off sequence.
def part_aframe():
    """14 waypoints: 4 linear fillets at the apex + 2 circular arcs on the joint."""
    h = aframe_height(L_AF, F_AF)
    apex = np.array([0.0, h, 0.0])
    feet = {"L": np.array([-F_AF / 2, 0.0, 0.0]), "R": np.array([+F_AF / 2, 0.0, 0.0])}
    points, welds = [], []

    for leg, foot in feet.items():
        u = unit_vector(apex, foot)
        for side in ("front", "back"):
            n = perpendicular(u, side)
            p_start = apex + (BAR_RADIUS + STANDOFF) * n
            p_end = p_start + FILLET_WELD_LEN * u
            i = len(points)
            tag = f"AF_LIN_{leg}_{side[0].upper()}"
            points += [(p_start, f"{tag}_S"), (p_end, f"{tag}_E")]
            welds.append({"process": "linear", "start": i, "moves": [("LIN", [i + 1])]})

    center = aframe_joint_center(F_AF, h, f_AF)
    half = WRAP_DEG / 2.0
    r = BAR_RADIUS + STANDOFF
    for k in range(2):
        i = len(points)
        for t, lbl in ((k * half, "S"), (k * half + half / 2, "M"), ((k + 1) * half, "E")):
            th = math.radians(t)
            offset = np.array([r * math.cos(th), 0.0, r * math.sin(th)])
            points.append((center + offset, f"AF_CIR_{k + 1}_{lbl}"))
        welds.append({"process": "circular", "start": i, "moves": [("CIR", [i + 1, i + 2])]})
    return points, welds


def part_ring():
    """4 waypoints: full 360 deg circle as two half arcs, one arc-on."""
    R = R_RING
    points = [
        (np.array([+R, 0.0, 0.0]), "RING_S"),    # 0 deg
        (np.array([0.0, +R, 0.0]), "RING_V1"),   # 90 deg
        (np.array([-R, 0.0, 0.0]), "RING_M"),    # 180 deg
        (np.array([0.0, -R, 0.0]), "RING_V2"),   # 270 deg
    ]
    welds = [{"process": "circular", "start": 0,
              "moves": [("CIR", [1, 2]), ("CIR", [3, 0])]}]
    return points, welds


def part_box():
    """4 waypoints: two parallel linear welds, second one in reverse direction."""
    L, G = LEN_BOX, GAP_BOX
    points = [
        (np.array([+G / 2, -L / 2, 0.0]), "BOX_L1_S"),
        (np.array([+G / 2, +L / 2, 0.0]), "BOX_L1_E"),
        (np.array([-G / 2, +L / 2, 0.0]), "BOX_L2_S"),
        (np.array([-G / 2, -L / 2, 0.0]), "BOX_L2_E"),
    ]
    welds = [
        {"process": "linear", "start": 0, "moves": [("LIN", [1])]},
        {"process": "linear", "start": 2, "moves": [("LIN", [3])]},
    ]
    return points, welds


PARTS = {"aframe": part_aframe, "ring": part_ring, "box": part_box}


# ============================================================================
# 6. JIG TRANSFORM  (4x4 homogeneous)
# ============================================================================
def jig_transform(alpha_deg, radius=R_TABLE):
    """T = Rot_z(alpha) . Trans(0, radius, 0)."""
    a = math.radians(alpha_deg)
    rot = np.array([
        [math.cos(a), -math.sin(a), 0.0, 0.0],
        [math.sin(a),  math.cos(a), 0.0, 0.0],
        [0.0,          0.0,         1.0, 0.0],
        [0.0,          0.0,         0.0, 1.0],
    ])
    trans = np.eye(4)
    trans[1, 3] = radius
    return rot @ trans


def to_global(point, T):
    """Apply a 4x4 transform to one xyz point."""
    return (T @ np.append(point, 1.0))[:3]


# ============================================================================
# 7. HEAT INPUT
# ============================================================================
def heat_input(I, V, v_cm_min, eta=ETA):
    """Q = eta * I * V / v   [kJ/mm], with v converted from cm/min to mm/s."""
    v_mm_s = v_cm_min * 10.0 / 60.0
    return (eta * I * V) / v_mm_s / 1000.0


def check_heat_input(Q, Q_des=Q_DESIGN, tol=Q_TOLERANCE):
    """Return (ok, deviation in %)."""
    dev = (Q - Q_des) / Q_des
    return abs(dev) <= tol, dev * 100.0


# ============================================================================
# 8. BUILD JIGS + STEP LIST
# ============================================================================
def build_jigs(preset):
    """Place every part of the preset on its jig. Returns a list of jig dicts."""
    jigs = []
    for idx, cfg in enumerate(preset["jigs"], start=1):
        points, welds = PARTS[cfg["type"]]()
        T = jig_transform(cfg["alpha"])
        jigs.append({
            "idx": idx,
            "alpha": cfg["alpha"],
            "type": cfg["type"],
            "waypoints": [(to_global(p, T), lbl) for p, lbl in points],
            "welds": welds,
        })
    return jigs


def generate_steps(jigs):
    """Turn the weld lists into numbered program steps."""
    out, step, base = [], 0, 0

    def add(cmd, param, comment):
        nonlocal step
        step += 1
        out.append({"kind": "step", "step": step, "cmd": cmd, "param": param,
                    "tool": TOOL, "comment": f"; {comment}"})

    def pname(i):
        return f"P{base + i + 1:03d}"

    for jig in jigs:
        out.append({"kind": "jig_header", "idx": jig["idx"],
                    "alpha": jig["alpha"], "type": jig["type"]})
        for n, weld in enumerate(jig["welds"], start=1):
            prm = PROCESS[weld["process"]]
            add("100% JOINT", pname(weld["start"]), f"Jig {jig['idx']} weld {n} approach")
            add("AS", f"{prm['I']:.0f}A {prm['V']}V", f"Arc start ({weld['process']})")
            for kind, idxs in weld["moves"]:
                add(f"{prm['v']:.0f} cm/min {kind}",
                    " ".join(pname(i) for i in idxs), f"Weld {n} {kind}")
            add("AE", "", f"Arc end weld {n}")
        base += len(jig["waypoints"])

    add("100% JOINT", "HOME", f"Return home - {len(jigs)} jig cycle done")
    return out, step


# ============================================================================
# 9. TEXT OUTPUT
# ============================================================================
def write_program_file(jigs, program, path):
    """Write a teach-pendant style listing. Illustrative format, not a
    controller import file."""
    steps, total_step = generate_steps(jigs)
    total_wp = sum(len(j["waypoints"]) for j in jigs)
    lines = [
        f"PROGRAM  = {program}",
        f"COMMENT  : 1 play-back cycle = {len(jigs)} jigs ({total_wp} global points)",
        f"GENERATED: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "; ==================== POSITION VARIABLES ====================",
    ]
    pid = 0
    for jig in jigs:
        lines.append(f"; --- Jig {jig['idx']}  (alpha = {jig['alpha']:+.1f} deg, {jig['type']}) ---")
        for pt, lbl in jig["waypoints"]:
            pid += 1
            lines.append(f"P{pid:03d}   X={pt[0]:+8.2f}  Y={pt[1]:+8.2f}  Z={pt[2]:+8.2f}   ; {lbl}")
        lines.append("")

    lines += ["; ==================== STEP INSTRUCTIONS ====================", "[START]"]
    for it in steps:
        if it["kind"] == "jig_header":
            lines += ["", f"; ===== JIG {it['idx']}  (alpha = {it['alpha']:+.1f} deg, {it['type']}) ====="]
        else:
            lines.append(f"{it['step']:04d}   {it['cmd']:<16}  {it['param']:<12}   {it['tool']}   {it['comment']}")
    lines += ["[END]", "", f"; Total global points : {pid}", f"; Total steps         : {total_step}"]

    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


# ============================================================================
# 10. PNG PREVIEW  (optional, needs Pillow)
# ============================================================================
FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),          # Linux
    ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/consolab.ttf"),     # Windows
    ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Menlo.ttc"),  # macOS
]


def _load_fonts(size):
    """Return (regular, bold). Falls back to Pillow's built-in font."""
    from PIL import ImageFont
    for regular, bold in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(regular, size), ImageFont.truetype(bold, size)
        except OSError:
            continue
    try:
        font = ImageFont.load_default(size)   # Pillow >= 10.1
    except TypeError:
        font = ImageFont.load_default()
    return font, font


def render_preview(jigs, program, path):
    """Draw the program listing as a teach-pendant style image."""
    from PIL import Image, ImageDraw

    steps, total_step = generate_steps(jigs)
    total_wp = sum(len(j["waypoints"]) for j in jigs)
    reg, bold = _load_fonts(12)
    _, big = _load_fonts(17)

    BG, PANEL, BORDER = "#F0F0F0", "#0D1117", "#30363D"
    YELLOW, RED, BLUE, GREEN = "#F0C674", "#F85149", "#79C0FF", "#7EE787"
    PURPLE, GRAY, WHITE = "#D2A8FF", "#8B949E", "#FFFFFF"
    TYPE_COLOR = {"aframe": RED, "ring": BLUE, "box": GREEN}

    row = 17
    left_rows = len(steps) + 2 * len(jigs) + 6
    right_rows = total_wp + 2 * len(jigs) + 6
    W, top = 1400, 130
    H = top + max(left_rows, right_rows) * row + 30

    img = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(img)

    dr.rectangle([(15, 15), (W - 15, 60)], fill=YELLOW, outline="#8B7735", width=2)
    dr.text((30, 27), f"Teach   Program {program}   Step 000   {datetime.now():%d/%m/%Y %H:%M}",
            font=big, fill="#1A1F2E")
    dr.rectangle([(15, 80), (W - 15, 115)], fill="#2C5288")
    dr.text((30, 88), f"100 %   JOINT   {TOOL}      1 play-back cycle = {len(jigs)} jigs",
            font=big, fill=WHITE)

    split = 800
    dr.rectangle([(15, top), (split - 5, H - 15)], fill=PANEL, outline=BORDER, width=2)
    dr.rectangle([(split + 5, top), (W - 15, H - 15)], fill=PANEL, outline=BORDER, width=2)

    # left panel: steps
    x, y = 35, top + 15
    dr.text((x, y), f"PROGRAM STEP INSTRUCTIONS  ({total_step} steps)", font=bold, fill=RED)
    y += row * 2
    for it in steps:
        if it["kind"] == "jig_header":
            y += row // 2
            dr.text((x, y), f"; === JIG {it['idx']}  (alpha = {it['alpha']:+.1f} deg) ===",
                    font=bold, fill=YELLOW)
            dr.text((x + 330, y), f"[{TYPE_LABEL[it['type']]}]", font=bold, fill=TYPE_COLOR[it["type"]])
            y += row + row // 2
        else:
            dr.text((x, y), f"{it['step']:04d}", font=reg, fill=BLUE)
            dr.text((x + 50, y), it["cmd"], font=reg, fill=YELLOW)
            dr.text((x + 200, y), it["param"], font=reg, fill=PURPLE)
            dr.text((x + 320, y), it["tool"], font=reg, fill=GREEN)
            dr.text((x + 390, y), it["comment"], font=reg, fill=GRAY)
            y += row

    # right panel: positions
    x, y = split + 25, top + 15
    dr.text((x, y), f"POSITION VARIABLES  ({total_wp} points)", font=bold, fill=RED)
    y += row
    dr.text((x, y), "P_global = Rot_z(alpha) . Trans(0, R, 0) . P_local", font=reg, fill=GRAY)
    y += row
    pid = 0
    for jig in jigs:
        y += row // 2
        dr.text((x, y), f"; === Jig {jig['idx']}  (alpha = {jig['alpha']:+.1f} deg) ===",
                font=bold, fill=YELLOW)
        dr.text((x + 300, y), f"[{TYPE_LABEL[jig['type']]}]", font=bold, fill=TYPE_COLOR[jig["type"]])
        y += row + row // 2
        for pt, lbl in jig["waypoints"]:
            pid += 1
            dr.text((x, y), f"P{pid:03d}", font=reg, fill="#C9D1D9")
            dr.text((x + 50, y), f"X={pt[0]:+8.2f}", font=reg, fill=BLUE)
            dr.text((x + 160, y), f"Y={pt[1]:+8.2f}", font=reg, fill=GREEN)
            dr.text((x + 270, y), f"Z={pt[2]:+7.2f}", font=reg, fill=YELLOW)
            dr.text((x + 370, y), f"; {lbl}", font=reg, fill=GRAY)
            y += row

    img.save(path)


# ============================================================================
# 11. MAIN
# ============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="TCP weld-path generator for a multi-jig cell")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="5jig")
    ap.add_argument("--outdir", default=".", help="output folder")
    ap.add_argument("--no-png", action="store_true", help="skip the PNG preview")
    args = ap.parse_args(argv)

    preset = PRESETS[args.preset]
    program = preset["program"]
    print("=" * 70)
    print(f"Weld path generator - preset {args.preset} (PROGRAM {program})")
    print("=" * 70)

    # 1) heat input check
    print(f"\n[Heat input]  Q_design = {Q_DESIGN} kJ/mm, tolerance +/-{Q_TOLERANCE * 100:.0f}%")
    all_ok = True
    for name, p in PROCESS.items():
        Q = heat_input(p["I"], p["V"], p["v"])
        ok, dev = check_heat_input(Q)
        all_ok = all_ok and ok
        print(f"  {name:<9} Q = {Q:.3f} kJ/mm  ({dev:+.1f}%)   {'PASS' if ok else 'FAIL'}")
    if not all_ok:
        print("  [STOP] welding parameters are outside the tolerance.")
        return 1

    # 2) place the parts on the jigs
    jigs = build_jigs(preset)
    print(f"\n[Layout]  R_table = {R_TABLE:.0f} mm")
    for j in jigs:
        print(f"  Jig {j['idx']}  alpha = {j['alpha']:+6.1f} deg  {j['type']:<7} "
              f"{len(j['waypoints']):2d} points  {len(j['welds'])} welds")
    total_wp = sum(len(j["waypoints"]) for j in jigs)
    _, total_step = generate_steps(jigs)
    print(f"\n  Total: {total_wp} global points, {total_step} steps")

    # 3) outputs
    os.makedirs(args.outdir, exist_ok=True)
    txt = os.path.join(args.outdir, f"PROGRAM_{program:02d}.txt")
    write_program_file(jigs, program, txt)
    print(f"\n[Output]  {txt}")

    if not args.no_png:
        try:
            png = os.path.join(args.outdir, f"PROGRAM_{program:02d}_preview.png")
            render_preview(jigs, program, png)
            print(f"[Output]  {png}")
        except ImportError:
            print("[Output]  PNG skipped (install Pillow to enable it)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
