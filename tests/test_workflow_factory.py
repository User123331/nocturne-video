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
    def build(self, overrides=None, paths=None):
        return wf.build_graph(spec(**(overrides or {})), paths=paths or PATHS,
                              gen_id="g1", upload_dir="nocturne/g1")

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

    def test_upscale_modes(self):
        cases = [
            ({"mode": "model", "model": "2x-animesharp"}, ["30", "31"]),
            ({"mode": "simple", "multiplier": 2, "interpolation": "Lanczos"}, ["34"]),
            ({"mode": "rtx", "scale": 2}, ["35"]),
            ({"mode": "h3_latent"}, ["40", "41", "42", "43"]),
        ]
        for upscale, node_ids in cases:
            paths = {**PATHS, "2x-animesharp": "upscale_models/2x_anime.safetensors",
                     "latent-upscaler-3d": "latent_upscaler.safetensors"}
            graph, meta = self.build({"upscale": upscale}, paths=paths)
            for nid in node_ids:
                self.assertIn(nid, graph, f"{upscale['mode']}: missing node {nid}")
            self.assertEqual(meta["upscale"]["mode"], upscale["mode"])
        # Simple mode passes the user's multiplier and interpolation through.
        paths = {**PATHS}
        graph, _ = self.build(
            {"upscale": {"mode": "simple", "multiplier": 3, "interpolation": "Bicubic"}},
            paths=paths)
        node = graph["34"]["inputs"]
        self.assertEqual(node["scale_multiplier"], 3)
        self.assertEqual(node["interpolation"], "Bicubic")

    def test_h3_latent_upscale_reroutes_decodes(self):
        paths = {**PATHS, "latent-upscaler-3d": "latent_upscaler.safetensors"}
        graph, _ = self.build({"upscale": {"mode": "h3_latent"}}, paths=paths)
        # Both the video and audio decode read the re-sampled latent.
        self.assertEqual(graph["12"]["inputs"]["samples"], ["43", 0])
        self.assertEqual(graph["13"]["inputs"]["samples"], ["43", 0])
        ultimate = graph["43"]["inputs"]
        self.assertEqual(ultimate["latent"], ["11", 0])
        self.assertEqual(ultimate["conditioning"], ["5", 0])
        self.assertEqual(ultimate["noise"], ["7", 0])
        # Target is a straight 2x of the canvas, snapped to the VAE's 16x grid.
        params = graph["40"]["inputs"]
        self.assertEqual(params["width"], 2048)
        self.assertEqual(params["height"], 1536)
        self.assertEqual(params["precision"], "fp16")

    def test_cache_node_chains_after_sigma_shift(self):
        graph, meta = self.build({"cache": {"reuse_threshold": 0.1, "max_steps": 3}})
        self.assertEqual(graph["18"]["class_type"], "MiniMaxH3Cache")
        self.assertEqual(graph["18"]["inputs"]["reuse_threshold"], 0.1)
        self.assertEqual(graph["18"]["inputs"]["max_steps"], 3)
        self.assertEqual(graph["9"]["inputs"]["model"], ["18", 0])
        self.assertTrue(meta["cache"])
        graph2, _ = self.build()
        self.assertNotIn("18", graph2)

    def test_watermark_node_uses_staged_filename(self):
        graph, meta = self.build({"watermark": {"image": "watermark.png", "scale": 0.2}})
        self.assertEqual(graph["37"]["class_type"], "DaSiWa_Watermark")
        self.assertEqual(graph["37"]["inputs"]["watermark_path"], "nocturne/g1/watermark.png")
        self.assertEqual(graph["37"]["inputs"]["scale"], 0.2)
        self.assertEqual(graph["14"]["inputs"]["images"], ["37", 0])
        self.assertTrue(meta["watermark"])

    def test_upscale_rejections(self):
        for bad in ({"mode": "nope"}, {"mode": "model", "model": "missing"},
                    {"mode": "simple", "multiplier": 99},
                    {"mode": "rtx", "scale": 8},
                    {"mode": "h3_latent", "precision": "int8"}):
            with self.assertRaises(wf.SpecError, msg=str(bad)):
                self.build({"upscale": bad})

    def test_rtx_quality_levels_match_node_enum(self):
        # DaSiWa's RTX node offers Low/Medium/High/Ultra only; "Super" does not
        # exist and the node rejects it at validation time.
        self.assertEqual(wf.RTX_QUALITY_LEVELS, ("Low", "Medium", "High", "Ultra"))
        with self.assertRaises(wf.SpecError):
            self.build({"upscale": {"mode": "rtx", "upscale_quality": "Super"}})
        graph, _ = self.build({"upscale": {"mode": "rtx", "upscale_quality": "High",
                                           "denoise": True, "deblur": True}})
        node = graph["35"]["inputs"]
        self.assertEqual(node["upscale_quality"], "High")
        self.assertEqual(node["denoise_quality"], "Ultra")
        self.assertTrue(node["denoise"])
        self.assertEqual(node["resize_type"], "Scale")
        self.assertEqual(node["resize_method"], "Center Crop (Fill)")

    def test_cache_rejections(self):
        with self.assertRaises(wf.SpecError):
            self.build({"cache": {"start_percent": 0.9, "end_percent": 0.1}})
        with self.assertRaises(wf.SpecError):
            self.build({"cache": {"max_steps": 99}})

    def test_custom_fps(self):
        # 5s at 8fps = 40 frames, snapped up to the 17k+5 grid; the combine
        # plays back at the chosen rate and the flf2va alignment line uses it.
        graph, meta = self.build({"fps": 8, "duration_seconds": 5})
        self.assertEqual(graph["5"]["inputs"]["length"], 56)  # 17*3+5
        self.assertEqual(graph["14"]["inputs"]["fps"], 8)
        self.assertEqual(meta["generation_fps"], 8)
        self.assertEqual(meta["duration_seconds"], 7.0)  # 56 frames / 8 fps
        # Interpolation doubles the output rate, not the generation rate.
        graph2, meta2 = self.build({"fps": 8, "duration_seconds": 5,
                                    "frame_interpolation": True})
        self.assertEqual(graph2["14"]["inputs"]["fps"], 16)
        self.assertEqual(meta2["fps"], 16)
        with self.assertRaises(wf.SpecError):
            self.build({"fps": 2})

    def test_custom_quality_uses_explicit_values(self):
        # The dashboard sends quality="custom" with every field filled.
        graph, meta = self.build({"quality": "custom", "sampler_name": "dpmpp_2m",
                                  "scheduler": "karras", "steps": 12,
                                  "shift_video": 9.0, "shift_audio": 4.0})
        self.assertEqual(graph["8"]["inputs"]["sampler_name"], "dpmpp_2m")
        self.assertEqual(graph["9"]["inputs"]["steps"], 12)
        self.assertEqual(graph["6"]["inputs"]["shift_video"], 9.0)
        self.assertEqual(meta["quality"], "custom")
        # Unset fields fall back to the Final preset.
        graph2, _ = self.build({"quality": "custom", "steps": 12})
        self.assertEqual(graph2["8"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(graph2["6"]["inputs"]["shift_audio"], 4.0)
        with self.assertRaises(wf.SpecError):
            self.build({"quality": "nonsense"})

    def test_frame_cap_spans_duration_and_fps(self):
        """Duration and fps can each be in range while the product is not."""
        self.assertEqual(wf.frames_for_duration(15, 24), 362)   # exactly the cap
        for duration, fps in ((15, 48), (15, 32), (10, 48), (8, 48)):
            with self.assertRaises(wf.SpecError, msg=f"{duration}s @ {fps}fps"):
                wf.frames_for_duration(duration, fps)

    def test_ref2va_reference_audio_is_wired(self):
        graph, _ = wf.build_graph(
            spec(task="ref2va", ref_images=["a.png"],
                 ref_audios=["one.mp3", "two.wav"]),
            paths=PATHS, gen_id="g1", upload_dir="nocturne/g1")
        cond = graph["5"]["inputs"]
        self.assertEqual(sorted(cond["ref_audios"]), ["ref_audio_1", "ref_audio_2"])
        self.assertEqual(cond["ref_audios"]["ref_audio_1"], ["231", 0])
        self.assertEqual(cond["ref_audios"]["ref_audio_2"], ["232", 0])
        self.assertEqual(graph["231"]["class_type"], "LoadAudio")
        self.assertEqual(graph["231"]["inputs"]["audio"], "nocturne/g1/one.mp3")
        self.assertEqual(graph["232"]["inputs"]["audio"], "nocturne/g1/two.wav")
        # No audio refs means no audio nodes at all.
        graph2, _ = wf.build_graph(spec(task="ref2va", ref_images=["a.png"]),
                                   paths=PATHS, gen_id="g1", upload_dir="nocturne/g1")
        self.assertNotIn("ref_audios", graph2["5"]["inputs"])

    def test_ref2va_audio_limit(self):
        with self.assertRaises(wf.SpecError):
            wf.build_graph(spec(task="ref2va", ref_images=["a.png"],
                                ref_audios=["1.mp3", "2.mp3", "3.mp3", "4.mp3"]),
                           paths=PATHS, gen_id="g1", upload_dir="u")

    def test_unknown_checkpoint(self):
        with self.assertRaises(wf.SpecError):
            self.build({"checkpoint": "nope"})

    def test_bad_steps(self):
        with self.assertRaises(wf.SpecError):
            self.build({"steps": 0})


if __name__ == "__main__":
    unittest.main()


class LoaderNameTests(unittest.TestCase):
    """Loader names must be relative to models/<kind>/, not prefixed with it.

    ComfyUI's loader nodes resolve within their own models/<kind>/ directory,
    so passing "diffusion_models/MiniMaxH3/x.safetensors" to UNETLoader fails
    with value_not_in_list before any node runs.
    """

    def test_graph_uses_folder_relative_loader_names(self):
        import sys
        sys.path.insert(0, ".")
        # Rebuild the slug->loader-name mapping the handler produces.
        from worker import asset_manager
        from pathlib import Path
        manifest = asset_manager.load_manifest(Path("config/assets.manifest.json"))
        by_slug = {a["slug"]: a for a in manifest}
        for slug in ("dasiwa-hybrid-v2-int8", "video-vae-fp16", "audio-vae-fp32",
                     "qwen3vl-32b-nvfp4-awq", "taeh3-preview", "rife-v4.26"):
            name = asset_manager.comfy_relative_name(by_slug[slug])
            self.assertFalse(
                name.startswith(by_slug[slug]["kind"] + "/"),
                f"{slug}: loader name {name!r} must not repeat the kind prefix")
        # The two VAE entries share a kind but differ by relative_dir.
        self.assertEqual(
            asset_manager.comfy_relative_name(by_slug["video-vae-fp16"]),
            "MiniMaxH3/minimax_h3_video_vae_fp16.safetensors")
        self.assertEqual(
            asset_manager.comfy_relative_name(by_slug["taeh3-preview"]),
            "taeh3.safetensors")

    def test_graph_passes_relative_names_to_loaders(self):
        import sys
        sys.path.insert(0, ".")
        from worker import asset_manager
        from pathlib import Path
        by_slug = {a["slug"]: a for a in asset_manager.load_manifest(Path("config/assets.manifest.json"))}
        paths = {s: asset_manager.comfy_relative_name(a) for s, a in by_slug.items()}
        graph, _ = wf.build_graph(spec(), paths=paths, gen_id="g1",
                                  upload_dir="nocturne/g1")
        self.assertEqual(graph["1"]["inputs"]["unet_name"],
                         "MiniMaxH3/dasiwa_hybrid_v2_int8.safetensors")
        self.assertEqual(graph["3"]["inputs"]["vae_name"],
                         "MiniMaxH3/minimax_h3_video_vae_fp16.safetensors")
        self.assertEqual(graph["2"]["inputs"]["clip_name"],
                         "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
