from genmo.mochi_preview.pipelines import (
    DecoderModelFactory,
    DitModelFactory,
    MochiSingleGPUPipeline,
    T5ModelFactory,
    linear_quadratic_schedule,
)

pipeline = MochiSingleGPUPipeline(
    text_encoder_factory=T5ModelFactory(),
    dit_factory=DitModelFactory(
        model_path=f"weights/dit.safetensors", model_dtype="bf16"
    ),
    decoder_factory=DecoderModelFactory(
        model_path=f"weights/decoder.safetensors",
    ),
    cpu_offload=True,
    decode_type="tiled_spatial",
)

video = pipeline(
    height=128,
    width=128,
    num_frames=13,
    num_inference_steps=64,
    sigma_schedule=linear_quadratic_schedule(64, 0.025),
    cfg_schedule=[6.0] * 64,
    batch_cfg=False,
    prompt="your favorite prompt here ...",
    negative_prompt="",
    seed=12345,
)