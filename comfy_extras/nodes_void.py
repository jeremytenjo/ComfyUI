import nodes
import node_helpers
import torch
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


class VOIDExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            VOIDQuadmaskPreprocess,
            VOIDInpaintConditioning,
        ]


async def comfy_entrypoint() -> VOIDExtension:
    return VOIDExtension()
