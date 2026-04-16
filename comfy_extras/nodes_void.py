import nodes
import node_helpers
import torch
import comfy
import comfy.latent_formats
import comfy.model_management
import comfy.utils
from comfy_api.latest import io, ComfyExtension
from typing_extensions import override


class VOIDQuadmaskPreprocess(io.ComfyNode):
    """Preprocess a quadmask video for VOID inpainting.

    Quantizes mask values to four semantic levels, inverts, and normalizes:
      0   -> primary object to remove
      63  -> overlap of primary + affected
      127 -> affected region (interactions)
      255 -> background (keep)

    After inversion and normalization, the output mask has values in [0, 1]
    with four discrete levels: 1.0 (remove), ~0.75, ~0.50, 0.0 (keep).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VOIDQuadmaskPreprocess",
            category="mask/video",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input("dilate_width", default=0, min=0, max=50, step=1,
                             tooltip="Dilation radius for the primary mask region (0 = no dilation)"),
            ],
            outputs=[
                io.Mask.Output(display_name="quadmask"),
            ],
        )

    @classmethod
    def execute(cls, mask, dilate_width=0) -> io.NodeOutput:
        m = mask.clone()

        if m.max() <= 1.0:
            m = m * 255.0

        if dilate_width > 0 and m.ndim >= 3:
            binary = (m < 128).float()
            kernel_size = dilate_width * 2 + 1
            if binary.ndim == 3:
                binary = binary.unsqueeze(1)
            dilated = torch.nn.functional.max_pool2d(
                binary, kernel_size=kernel_size, stride=1, padding=dilate_width
            )
            if dilated.ndim == 4:
                dilated = dilated.squeeze(1)
            m = torch.where(dilated > 0.5, torch.zeros_like(m), m)

        m = torch.where(m <= 31, torch.zeros_like(m), m)
        m = torch.where((m > 31) & (m <= 95), torch.full_like(m, 63), m)
        m = torch.where((m > 95) & (m <= 191), torch.full_like(m, 127), m)
        m = torch.where(m > 191, torch.full_like(m, 255), m)

        m = (255.0 - m) / 255.0

        return io.NodeOutput(m)


class VOIDInpaintConditioning(io.ComfyNode):
    """Build VOID inpainting conditioning for CogVideoX.

    Encodes the processed quadmask and masked source video through the VAE,
    producing a 32-channel concat conditioning (16ch mask + 16ch masked video)
    that gets concatenated with the 16ch noise latent by the model.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VOIDInpaintConditioning",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Image.Input("video", tooltip="Source video frames [T, H, W, 3]"),
                io.Mask.Input("quadmask", tooltip="Preprocessed quadmask from VOIDQuadmaskPreprocess [T, H, W]"),
                io.Int.Input("width", default=672, min=16, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("height", default=384, min=16, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("length", default=49, min=1, max=nodes.MAX_RESOLUTION, step=1,
                             tooltip="Number of pixel frames to process"),
                io.Int.Input("batch_size", default=1, min=1, max=64),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, video, quadmask,
                width, height, length, batch_size) -> io.NodeOutput:

        temporal_compression = 4
        latent_t = ((length - 1) // temporal_compression) + 1
        latent_h = height // 8
        latent_w = width // 8

        vid = video[:length]
        vid = comfy.utils.common_upscale(
            vid.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        qm = quadmask[:length]
        if qm.ndim == 3:
            qm = qm.unsqueeze(-1)
        qm = comfy.utils.common_upscale(
            qm.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        if qm.ndim == 4 and qm.shape[-1] == 1:
            qm = qm.squeeze(-1)

        mask_condition = qm
        if mask_condition.ndim == 3:
            mask_condition_3ch = mask_condition.unsqueeze(-1).expand(-1, -1, -1, 3)
        else:
            mask_condition_3ch = mask_condition

        inverted_mask_3ch = 1.0 - mask_condition_3ch
        masked_video = vid[:, :, :, :3] * (1.0 - mask_condition_3ch)

        mask_latents = vae.encode(inverted_mask_3ch)
        masked_video_latents = vae.encode(masked_video)

        def _match_temporal(lat, target_t):
            if lat.shape[2] > target_t:
                return lat[:, :, :target_t]
            elif lat.shape[2] < target_t:
                pad = target_t - lat.shape[2]
                return torch.cat([lat, lat[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)
            return lat

        mask_latents = _match_temporal(mask_latents, latent_t)
        masked_video_latents = _match_temporal(masked_video_latents, latent_t)

        inpaint_latents = torch.cat([mask_latents, masked_video_latents], dim=1)

        # CogVideoX-Fun was trained with Diffusers convention where VAE latents
        # are scaled by 0.7 (vae.config.scaling_factor). CogVideoX.concat_cond()
        # applies process_latent_in (×sf=1.153) to the stored conditioning.
        # Pre-multiply by 0.7 so the model sees the correct magnitude:
        #   stored = vae_output × 0.7  →  after process_in: (vae_output×0.7)×sf = raw×0.7
        DIFFUSERS_SCALING_FACTOR = 0.7
        inpaint_latents = inpaint_latents * DIFFUSERS_SCALING_FACTOR

        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": inpaint_latents}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": inpaint_latents}
        )

        noise_latent = torch.zeros(
            [batch_size, 16, latent_t, latent_h, latent_w],
            device=comfy.model_management.intermediate_device()
        )

        return io.NodeOutput(positive, negative, {"samples": noise_latent})


class VOIDWarpedNoise(io.ComfyNode):
    """Generate optical-flow warped noise for VOID Pass 2 refinement.

    Takes the Pass 1 output video and produces temporally-correlated noise
    by warping Gaussian noise along optical flow vectors. This noise is used
    as the initial latent for Pass 2, resulting in better temporal consistency.

    Requires: pip install rp (auto-installs Go-with-the-Flow dependencies)
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VOIDWarpedNoise",
            category="latent/video",
            inputs=[
                io.Image.Input("video", tooltip="Pass 1 output video frames [T, H, W, 3]"),
                io.Int.Input("width", default=672, min=16, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("height", default=384, min=16, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("length", default=49, min=1, max=nodes.MAX_RESOLUTION, step=1,
                             tooltip="Number of pixel frames"),
                io.Int.Input("batch_size", default=1, min=1, max=64),
            ],
            outputs=[
                io.Latent.Output(display_name="warped_noise"),
            ],
        )

    @classmethod
    def execute(cls, video, width, height, length, batch_size) -> io.NodeOutput:
        import numpy as np

        try:
            import rp
            rp.r._pip_import_autoyes = True
            rp.git_import('CommonSource')
            import rp.git.CommonSource.noise_warp as nw
        except ImportError:
            raise RuntimeError(
                "VOIDWarpedNoise requires the 'rp' package. Install with: pip install rp"
            )

        temporal_compression = 4
        latent_t = ((length - 1) // temporal_compression) + 1
        latent_h = height // 8
        latent_w = width // 8

        vid = video[:length].cpu().numpy()
        vid_uint8 = (vid * 255).clip(0, 255).astype(np.uint8)

        frames = [vid_uint8[i] for i in range(vid_uint8.shape[0])]
        frames = rp.resize_images_to_hold(frames, height=height, width=width)
        frames = rp.crop_images(frames, height=height, width=width, origin='center')
        frames = rp.as_numpy_array(frames)

        FRAME = 2**-1
        FLOW = 2**3
        LATENT_SCALE = 8

        warp_output = nw.get_noise_from_video(
            frames,
            remove_background=False,
            visualize=False,
            save_files=False,
            noise_channels=16,
            output_folder=None,
            resize_frames=FRAME,
            resize_flow=FLOW,
            downscale_factor=round(FRAME * FLOW) * LATENT_SCALE,
        )

        warped_np = warp_output.numpy_noises  # (T, H, W, C)
        if warped_np.dtype == np.float16:
            warped_np = warped_np.astype(np.float32)

        import cv2

        if warped_np.shape[0] != latent_t:
            indices = np.linspace(0, warped_np.shape[0] - 1, latent_t).astype(int)
            warped_np = warped_np[indices]

        if warped_np.shape[1] != latent_h or warped_np.shape[2] != latent_w:
            resized = []
            for t_idx in range(latent_t):
                frame = warped_np[t_idx]
                ch_resized = [
                    cv2.resize(frame[:, :, c], (latent_w, latent_h),
                               interpolation=cv2.INTER_LINEAR)
                    for c in range(frame.shape[2])
                ]
                resized.append(np.stack(ch_resized, axis=2))
            warped_np = np.stack(resized, axis=0)

        # (T, H, W, C) -> (B, C, T, H, W)
        warped_tensor = torch.from_numpy(
            warped_np.transpose(3, 0, 1, 2)
        ).float().unsqueeze(0)

        if batch_size > 1:
            warped_tensor = warped_tensor.repeat(batch_size, 1, 1, 1, 1)

        warped_tensor = warped_tensor.to(comfy.model_management.intermediate_device())

        return io.NodeOutput({"samples": warped_tensor})


class Noise_FromLatent:
    """Wraps a pre-computed LATENT tensor as a NOISE source."""
    def __init__(self, latent_dict):
        self.seed = 0
        self._samples = latent_dict["samples"]

    def generate_noise(self, input_latent):
        return self._samples.clone().cpu()


class VOIDWarpedNoiseSource(io.ComfyNode):
    """Convert a LATENT (e.g. from VOIDWarpedNoise) into a NOISE source
    for use with SamplerCustomAdvanced."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VOIDWarpedNoiseSource",
            category="sampling/custom_sampling/noise",
            inputs=[
                io.Latent.Input("warped_noise",
                    tooltip="Warped noise latent from VOIDWarpedNoise"),
            ],
            outputs=[io.Noise.Output()],
        )

    @classmethod
    def execute(cls, warped_noise) -> io.NodeOutput:
        return io.NodeOutput(Noise_FromLatent(warped_noise))


class VOID_DDIM(comfy.samplers.Sampler):
    """DDIM sampler for VOID inpainting models.

    VOID was trained with the diffusers CogVideoXDDIMScheduler which operates in
    alpha-space (input std ≈ 1). The standard KSampler applies noise_scaling that
    multiplies by sqrt(1+sigma^2) ≈ 4500x, which is incompatible with VOID's
    training. This sampler skips noise_scaling and implements the DDIM update rule
    directly using sigma-to-alpha conversion.
    """

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        x = noise.to(torch.float32)
        model_options = extra_args.get("model_options", {})
        seed = extra_args.get("seed", None)
        s_in = x.new_ones([x.shape[0]])

        for i in trange(len(sigmas) - 1, disable=disable_pbar):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            denoised = model_wrap(x, sigma * s_in, model_options=model_options, seed=seed)

            if callback is not None:
                callback(i, denoised, x, len(sigmas) - 1)

            if sigma_next == 0:
                x = denoised
            else:
                alpha_t = 1.0 / (1.0 + sigma ** 2)
                alpha_prev = 1.0 / (1.0 + sigma_next ** 2)

                pred_eps = (x - (alpha_t ** 0.5) * denoised) / (1.0 - alpha_t) ** 0.5
                x = (alpha_prev ** 0.5) * denoised + (1.0 - alpha_prev) ** 0.5 * pred_eps

        return x


class VOIDSampler(io.ComfyNode):
    """VOID DDIM sampler for use with SamplerCustom / SamplerCustomAdvanced.

    Required for VOID inpainting models. Implements the same DDIM loop that VOID
    was trained with (diffusers CogVideoXDDIMScheduler), without the noise_scaling
    that the standard KSampler applies. Use with RandomNoise or VOIDWarpedNoiseSource.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VOIDSampler",
            category="sampling/custom_sampling/samplers",
            inputs=[],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput(VOID_DDIM())

    get_sampler = execute


class VOIDExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            VOIDQuadmaskPreprocess,
            VOIDInpaintConditioning,
            VOIDWarpedNoise,
            VOIDWarpedNoiseSource,
            VOIDSampler,
        ]


async def comfy_entrypoint() -> VOIDExtension:
    return VOIDExtension()
