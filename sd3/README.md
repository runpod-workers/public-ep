### Public-Image Gen Endpoint 

***Docker Image***
`mwiki/sd3:v1`

## 🧪 Model Testing Instructions

### 🚀 Deployment Requirements

- **GPU Requirement**: > 24GB
- **Docker Image Size**: Deployment may take some time as the Docker image is large; the model is pre-packaged ("baked in") within the image.

### 📦 API Behavior

Once the model is deployed, it exposes an API endpoint that accepts a structured input and returns an image in **base64 format**. If you're using the [RunPod SDK](https://github.com/runpod/runpod-python), the image can be decoded and rendered. 

---

### 📥 Example Input(Try it out in UI)

```json
{
  "input": {
    "prompt": "a photograph of a dog",
    "negative_prompt": "",
    "height": "",
    "width": "",
    "num_inference_steps": "",
    "guidance": ""
  }
}
```

### 🔍 Parameter Descriptions

| Parameter              | Type     | Description                                                                                       |
|------------------------|----------|---------------------------------------------------------------------------------------------------|
| `prompt`               | `string` | The main text prompt that guides the image generation (e.g., `"a sunset over mountains"`).        |
| `negative_prompt`      | `string` | Optional. A prompt used to suppress unwanted features in the image (e.g., `"blurry, low quality"`). |
| `height`               | `int`    | Desired height of the output image in pixels (e.g., `512`, `768`). Must be supported by the model. |
| `width`                | `int`    | Desired width of the output image in pixels. Must be supported by the model.                      |
| `num_inference_steps`  | `int`    | Number of denoising steps used in the generation process. Higher values yield more detailed images (common range: `20–50`). |
| `guidance` (CFG Scale) | `float`  | Classifier-Free Guidance Scale. Controls how closely the image follows the prompt (`5–15` is typical). |
