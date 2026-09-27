"""Tests for the build gate's combo-validation skip rules.

The gate boots ComfyUI inside the image where the model volume is not
mounted, so file-picker combos legitimately contain no staged files. Four
releases (v0.3.0..v0.4.1) failed to build because ComfyUI pads the empty
combos with built-in entries ("pixel_space", "example.png") that the skip
logic did not recognize.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify-comfy-nodes.py"
_spec = importlib.util.spec_from_file_location("verify_comfy_nodes", _SCRIPT)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def _object_info(combo: list) -> dict:
    """Minimal /object_info shape for a VAELoader with the given combo.

    Real object_info wraps the options list directly: ["combo"] is spelled
    [[opt, ...]], so the gate reads spec_in[0] as the options list.
    """
    return {"VAELoader": {"input": {"required": {"vae_name": [combo]}}}}


def _graph(vae_name: str) -> dict:
    return {"g": {"3": {"class_type": "VAELoader",
                        "inputs": {"vae_name": vae_name}}}}


class BuildGateComboTests(unittest.TestCase):
    def test_builtin_only_combo_is_skipped(self):
        """['pixel_space'] / ['example.png'] mean 'no assets staged', not invalid."""
        for combo in (["pixel_space"], ["example.png"]):
            with self.subTest(combo=combo):
                problems = gate.validate_graph_inputs(
                    _graph("MiniMaxH3/minimax_h3_video_vae_fp16.safetensors"),
                    _object_info(combo))
                self.assertEqual(problems, [])

    def test_empty_and_placeholder_combos_are_skipped(self):
        """The pre-existing skip rules keep working."""
        for combo in ([], ["(no .safetensors upscale models found in: /comfyui/models)"]):
            with self.subTest(combo=combo):
                problems = gate.validate_graph_inputs(
                    _graph("MiniMaxH3/minimax_h3_video_vae_fp16.safetensors"),
                    _object_info(combo))
                self.assertEqual(problems, [])

    def test_staged_combo_still_validates_values(self):
        """A combo that lists real files must still catch a bad value."""
        problems = gate.validate_graph_inputs(
            _graph("MiniMaxH3/minimax_h3_video_vae_fp32.safetensors"),
            _object_info(["pixel_space", "MiniMaxH3/minimax_h3_video_vae_fp16.safetensors"]))
        self.assertEqual(len(problems), 1)
        self.assertIn("not in combo", problems[0])

    def test_good_value_in_staged_combo_passes(self):
        problems = gate.validate_graph_inputs(
            _graph("MiniMaxH3/minimax_h3_video_vae_fp16.safetensors"),
            _object_info(["pixel_space", "MiniMaxH3/minimax_h3_video_vae_fp16.safetensors"]))
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
