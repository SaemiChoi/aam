import torch
from aam import cross_attention, utils
from aam.pipeline import StableDiffusionPipeline

device = 'cpu'

seed = 87
prompt = ["futuristic teddy bear", 0, [1, 2]]

pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-4", torch_dtype=torch.float16, local_files_only=True)
pipe = pipe.to(device)

with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():

    # stable diffusion
    out = pipe(
        prompt[0],
        num_inference_steps=10,
        generator=utils.set_seed(seed, device),
        negative_prompt='cropped',
        nsfw_enabled=False
    )
    out.images[0].save('output_ori.jpg')

    # stable diffusion /w aam
    with cross_attention.hook(pipe):
        out = pipe(
            prompt,
            num_inference_steps=30,
            generator=utils.set_seed(seed, device),
            negative_prompt='cropped',
            nsfw_enabled=False
        )
        out.images[0].save('output_aam.jpg')