
## Usage

The worker accepts the following input parameters:

| Parameter                 | Type    | Default  | Required  | Description                                                                                                         |
| :------------------------ | :------ | :------- | :-------- | :------------------------------------------------------------------------------------------------------------------ |
| `prompt`                  | `str`   | `None`   | **Yes\*** | The main text prompt describing the desired image.                                                                  |
| `negative_prompt`         | `str`   | `None`   | No        | Text prompt specifying concepts to exclude from the image                                                           |
| `height`                  | `int`   | `1024`   | No        | The height of the generated image in pixels                                                                         |
| `width`                   | `int`   | `1024`   | No        | The width of the generated image in pixels                                                                          |
| `seed`                    | `int`   | `None`   | No        | Random seed for reproducibility. If `None`, a random seed is generated                                              |
| `num_inference_steps`     | `int`   | `25`     | No        | Number of denoising steps for the base model                                                                        |                                                                    |
| `guidance`          | `float` | `7.5`    | No        | Classifier-Free Guidance scale. Higher values lead to images closer to the prompt, lower values more creative       |

> [!NOTE]  
> `prompt` is required

### Example Request

```json
{
  "input": {
    "prompt": "A majestic steampunk dragon soaring through a cloudy sky, intricate clockwork details, golden hour lighting, highly detailed",
    "negative_prompt": "blurry, low quality, deformed, ugly, text, watermark, signature",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    "seed": 42,
  }
}
```
