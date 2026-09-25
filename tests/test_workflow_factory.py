"""Tests for the worker's workflow factory (native H3 graph construction)."""

from __future__ import annotations

import json
import unittest

from worker import workflow_factory as wf

PATHS = {
    "dasiwa-hybrid-v2-int8": "diffusion_models/MiniMaxH3/dasiwa_hybrid_v2_int8.safetensors",
    "dasiwa-hybrid-v2-int4": "diffusion_models/MiniMaxH3/dasiwa_hybrid_v2_int4.safetensors",
    "qwen3vl-32b-nvfp4-awq": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video-vae-fp16": "vae/MiniMaxH3/minimax_h3_video_vae_fp16.safetensors",
    "audio-vae-fp32": "vae/MiniMaxH3/minimax_h3_audio_vae_fp32.safetensors",
}


def spec(**overrides):
    base = {
        "task": "t2va",
        "quality": "quality",
        "prompt": {"integrated_multimodal_description": "[Shot 1] A lighthouse.",
                   "overall_soundscape": "Wind.",
                   "non_diegetic_music": "Pulsing synth."},
    }
    base.update(overrides)
    return base


class FrameMathTests(unittest.TestCase):
    def test_snaps_to_17k_plus_5(self):
        for seconds, expected in ((5, 124), (1, 39), (8, 192), (15, 362)):
            self.assertEqual(wf.frames_for_duration(seconds), expected, seconds)

    def test_out_of_range(self):
        with self.assertRaises(wf.SpecError):
            wf.frames_for_duration(16)
        with self.assertRaises(wf.SpecError):
            wf.frames_for_duration(0.2)


class CanvasTests(unittest.TestCase):
    def test_snaps_to_32(self):
        self.assertEqual(wf.clamp_canvas(1000, 700), (992, 704))

    def test_area_cap(self):
        w, h = wf.clamp_canvas(2560, 1440)
        self.assertEqual((w, h), (2560, 1440))
        w, h = wf.clamp_canvas(4096, 2160)
        self.assertLessEqual(w * h, 2560 * 1440)


class PromptTests(unittest.TestCase):
    def test_t2va_sections(self):
        text = wf.assemble_prompt("t2va", spec()["prompt"], 5.0)
        self.assertIn("integrated_multimodal_description: [Shot 1] A lighthouse.", text)
        self.assertIn("overall_soundscape: Wind.", text)
        self.assertIn("non_diegetic_music: Pulsing synth.", text)

    def test_blank_music_becomes_na(self):
        text = wf.assemble_prompt("t2va", {"integrated_multimodal_description": "x"}, 5.0)
        self.assertIn("non_diegetic_music: N/A", text)

    def test_i2va_alignment_line(self):
        text = wf.assemble_prompt("i2va", spec()["prompt"], 5.0)
        self.assertTrue(text.startswith("For the target video, at 0.00 seconds"))
        self.assertIn("Picture 1 (from Shot 1) is fully referenced.", text)

    def test_flf2va_alignment_line_mentions_last_frame_time(self):
        text = wf.assemble_prompt("flf2va", spec()["prompt"], 8.0)
        self.assertIn("Picture 2 (from Shot 1) aligns with the 8.00-second mark", text)

    def test_ref2va_six_sections(self):
        text = wf.assemble_prompt("ref2va", {
            "subject_definitions": "s", "summary": "s2", "retention_analysis": "r",
            "detailed_description": "d", "overall_soundscape": "o"}, 5.0)
        for key in ("subject_definitions:", "summary:", "retention_analysis:",
                    "detailed_description:", "overall_soundscape:", "non_diegetic_music: N/A"):
            self.assertIn(key, text)

    def test_missing_imd_raises(self):
        with self.assertRaises(wf.SpecError):
            wf.assemble_prompt("t2va", {}, 5.0)


class GraphTests(unittest.TestCase):
    def build(self, overrides=None):
        return wf.build_graph(spec(**(overrides or {})), paths=PATHS, gen_id="g1",
                              upload_dir="nocturne/g1")

    def test_t2va_graph_shape(self):
        graph, meta = self.build()
        self.assertEqual(graph["1"]["class_type"], "UNETLoader")
        self.assertEqual(graph["1"]["inputs"]["unet_name"],
                         "diffusion_models/MiniMaxH3/dasiwa_hybrid_v2_int8.safetensors")
        self.assertEqual(graph["2"]["inputs"]["type"], "minimax")
        self.assertEqual(graph["5"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertEqual(graph["11"]["class_type"], "SamplerCustomAdvanced")
        self.assertEqual(graph["14"]["class_type"], "CreateVideo")
        self.assertEqual(graph["15"]["class_type"], "SaveVideo")
        self.assertEqual(graph["15"]["inputs"]["filename_prefix"], "nocturne/g1")
        self.assertEqual(graph["14"]["inputs"]["fps"], 24)
        self.assertEqual(graph["14"]["inputs"]["audio"], ["13", 0])
        self.assertEqual(meta["frames"], 124)

    def test_seed_randomized(self):
        _, meta = self.build({"seed": -1})
        self.assertIsInstance(meta["seed"], int)
        self.assertGreaterEqual(meta["seed"], 0)

    def test_quality_presets(self):
        _, quality_meta = self.build()
        self.assertEqual(quality_meta["steps"], 25)
        self.assertEqual(quality_meta["sampler_name"], "res_multistep")
        self.assertEqual(quality_meta["shift_video"], 11.0)
        _, turbo_meta = self.build({"quality": "turbo"})
        self.assertEqual(turbo_meta["steps"], 8)
        self.assertEqual(turbo_meta["sampler_name"], "euler")
        self.assertEqual(turbo_meta["shift_audio"], 4.5)

    def test_i2va_wires_first_frame(self):
        graph, _ = self.build({
            "task": "i2va",
            "first_frame": "nocturne/g1/first_frame.png",
        })
        self.assertEqual(graph["5"]["inputs"]["first_frame"], ["21", 0])
        self.assertEqual(graph["21"]["class_type"], "LoadImage")

    def test_i2va_requires_first_frame(self):
        with self.assertRaises(wf.SpecError):
            self.build({"task": "i2va"})

    def test_flf2va_wires_both_frames(self):
        graph, _ = self.build({
            "task": "flf2va",
            "first_frame": "nocturne/g1/first_frame.png",
            "last_frame": "nocturne/g1/last_frame.png",
        })
        self.assertEqual(graph["5"]["inputs"]["first_frame"], ["21", 0])
        self.assertEqual(graph["5"]["inputs"]["last_frame"], ["22", 0])

    def test_ref2va_graph(self):
        graph, _ = self.build({
            "task": "ref2va",
            "prompt": {"subject_definitions": "s", "summary": "s", "retention_analysis": "r",
                       "detailed_description": "d", "overall_soundscape": "o"},
            "ref_images": ["nocturne/g1/ref_image_1.png"],
        })
        self.assertEqual(graph["5"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(graph["5"]["inputs"]["ref_images"], {"ref_image_1": ["201", 0]})
        self.assertEqual(graph["5"]["inputs"]["audio_vae"], ["4", 0])

    def test_ref2va_requires_refs(self):
        with self.assertRaises(wf.SpecError):
            self.build({"task": "ref2va",
                        "prompt": {"subject_definitions": "s", "summary": "s",
                                   "retention_analysis": "r", "detailed_description": "d",
                                   "overall_soundscape": "o"}})

    def test_lora_stack_via_dasiwa_loader(self):
        graph, _ = self.build({"loras": [{"name": "lora_a.safetensors", "strength": 0.8},
                                          {"name": "lora_b.safetensors", "strength": 1.2},
                                          {"name": "", "strength": 1.0}]})
        node = graph["16"]
        self.assertEqual(node["class_type"], "DaSiWa_LTX2LoraLoader")
        self.assertEqual(node["inputs"]["model"], ["1", 0])
        self.assertEqual(node["inputs"]["clip"], ["2", 0])
        self.assertEqual(node["inputs"]["model_type"], "Basic")
        stack = json.loads(node["inputs"]["stack_data"])
        self.assertEqual(len(stack), 2)  # blank row dropped
        self.assertEqual(stack[0], {"on": True, "lora": "lora_a.safetensors",
                                    "str": 0.8, "vs": 1, "as": 1})
        self.assertEqual(graph["6"]["inputs"]["model"], ["16", 0])
        self.assertEqual(graph["5"]["inputs"]["clip"], ["16", 1])

    def test_no_loras_keeps_direct_wiring(self):
        graph, _ = self.build()
        self.assertNotIn("16", graph)
        self.assertEqual(graph["6"]["inputs"]["model"], ["1", 0])
        self.assertEqual(graph["5"]["inputs"]["clip"], ["2", 0])

    def test_frame_interpolation_wiring(self):
        graph, meta = self.build({"frame_interpolation": True})
        self.assertEqual(graph["32"]["class_type"], "FrameInterpolationModelLoader")
        self.assertEqual(graph["33"]["class_type"], "FrameInterpolate")
        self.assertEqual(graph["33"]["inputs"]["interp_model"], ["32", 0])
        self.assertEqual(graph["33"]["inputs"]["multiplier"], 2)
        self.assertEqual(graph["14"]["inputs"]["images"], ["33", 0])
        self.assertEqual(graph["14"]["inputs"]["fps"], 48)
        self.assertEqual(meta["fps"], 48)

    def test_no_interpolation_default(self):
        graph, meta = self.build()
        self.assertNotIn("33", graph)
        self.assertEqual(graph["14"]["inputs"]["fps"], 24)
        self.assertFalse(meta["frame_interpolation"])

    def test_chunk_ffn_wiring(self):
        graph, meta = self.build({"chunk_ffn": True})
        self.assertEqual(graph["17"]["class_type"], "MiniMaxChunkFeedForward")
        self.assertEqual(graph["17"]["inputs"]["chunks"], 4)
        self.assertEqual(graph["9"]["inputs"]["model"], ["17", 0])
        self.assertEqual(graph["10"]["inputs"]["model"], ["17", 0])
        self.assertTrue(meta["chunk_ffn"])
        graph2, meta2 = self.build()
        self.assertNotIn("17", graph2)
        self.assertEqual(graph2["9"]["inputs"]["model"], ["6", 0])
        self.assertFalse(meta2["chunk_ffn"])

    def test_upscale_chain(self):
        paths = {**PATHS, "2x-animesharp": "upscale_models/2x_anime.safetensors"}
        graph, _ = wf.build_graph(spec(upscale_model="2x-animesharp"), paths=paths,
                                  gen_id="g1", upload_dir="nocturne/g1")
        self.assertEqual(graph["14"]["inputs"]["images"], ["31", 0])
        self.assertEqual(graph["31"]["class_type"], "ImageUpscaleWithModel")

    def test_unknown_checkpoint(self):
        with self.assertRaises(wf.SpecError):
            self.build({"checkpoint": "nope"})

    def test_bad_steps(self):
        with self.assertRaises(wf.SpecError):
            self.build({"steps": 0})


if __name__ == "__main__":
    unittest.main()
