"""Unit tests. Run from the repo root with:  python -m pytest"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import weld_path_generator as w


def test_heat_input_formula():
    # 0.8 * 100 A * 20 V / (60 cm/min = 10 mm/s) = 160 J/mm = 0.160 kJ/mm
    assert math.isclose(w.heat_input(100, 20, 60, eta=0.8), 0.160)


def test_default_parameters_pass_heat_input_check():
    for p in w.PROCESS.values():
        ok, _ = w.check_heat_input(w.heat_input(p["I"], p["V"], p["v"]))
        assert ok


def test_waypoint_counts_per_part():
    assert len(w.part_aframe()[0]) == 14
    assert len(w.part_ring()[0]) == 4
    assert len(w.part_box()[0]) == 4


def test_preset_totals():
    total = lambda name: sum(len(j["waypoints"]) for j in w.build_jigs(w.PRESETS[name]))
    assert total("5jig") == 40
    assert total("4jig") == 36


def test_transform_keeps_distances():
    T = w.jig_transform(37.0)
    a, b = np.array([10.0, 5.0, 2.0]), np.array([-4.0, 8.0, 0.0])
    d_local = np.linalg.norm(a - b)
    d_global = np.linalg.norm(w.to_global(a, T) - w.to_global(b, T))
    assert math.isclose(d_local, d_global)


def test_jig_origin_lies_on_table_radius():
    for alpha in (-60.0, 0.0, 25.0):
        origin = w.to_global(np.zeros(3), w.jig_transform(alpha))
        assert math.isclose(np.linalg.norm(origin), w.R_TABLE)


def test_ring_points_stay_on_the_weld_circle():
    for jig in w.build_jigs(w.PRESETS["4jig"]):
        if jig["type"] != "ring":
            continue
        center = w.to_global(np.zeros(3), w.jig_transform(jig["alpha"]))
        for pt, _ in jig["waypoints"]:
            assert math.isclose(np.linalg.norm(pt - center), w.R_RING)


def test_ring_jigs_do_not_overlap():
    rings = [j for j in w.build_jigs(w.PRESETS["4jig"]) if j["type"] == "ring"]
    c0 = w.to_global(np.zeros(3), w.jig_transform(rings[0]["alpha"]))
    c1 = w.to_global(np.zeros(3), w.jig_transform(rings[1]["alpha"]))
    assert np.linalg.norm(c0 - c1) > 2 * w.R_RING


def test_steps_reference_existing_points():
    for name, preset in w.PRESETS.items():
        jigs = w.build_jigs(preset)
        total = sum(len(j["waypoints"]) for j in jigs)
        steps, n = w.generate_steps(jigs)
        numbered = [s for s in steps if s["kind"] == "step"]
        assert [s["step"] for s in numbered] == list(range(1, n + 1))
        for s in numbered:
            for token in s["param"].split():
                if token.startswith("P") and token[1:].isdigit():
                    assert 1 <= int(token[1:]) <= total
