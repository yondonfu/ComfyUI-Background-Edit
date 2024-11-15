# ComfyUI-Background-Edit

ComfyUI-Background-Edit is a set of [ComfyUI](https://www.comfy.org/) nodes for editing background of images/videos with CUDA acceleration support.

Supported use cases:

- Background blurring
- Background removal
- Background swapping

The CUDA accelerated nodes can be used in real-time workflows for live video streams using [comfystream](https://github.com/yondonfu/comfystream).

--- 

- [ComfyUI-Background-Edit](#comfyui-background-edit)
- [Install](#install)
  - [Comfy Registry](#comfy-registry)
  - [Manual](#manual)
- [Example Real-Time Live Video Workflows](#example-real-time-live-video-workflows)
- [Example Image Workflows](#example-image-workflows)
- [Nodes](#nodes)

# Install

The recommended installation method is to use the Comfy Registry.

## Comfy Registry

These nodes can be installed via the [Comfy Registry](https://registry.comfy.org/nodes/comfyui-background-edit).

```
comfy node registry-install comfyui-background-edit
```

## Manual

These nodes can also be installed manually by copying them into your `custom_nodes` folder and then installing dependencies:

```
cd custom_nodes
git clone https://github.com/yondonfu/ComfyUI-Background-Edit
cd ComfyUI-Background-Edit
pip install -r requirements.txt
```

# Example Real-Time Live Video Workflows

**Prerequisites**

- Install [ComfyUI-Depth-Anything-Tensorrt](https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt).
  - The example workflow uses TensorRT to accelerate DepthAnything2 to meet real-time requirements.

The [workflow (API format)](https://github.com/yondonfu/comfystream/blob/main/workflows/depth-bg-blur-gpu-workflow.json) can be used with [comfystream](https://github.com/yondonfu/comfystream) to run real-time background blurring on a live video stream.

The [workflow](./examples/realtime-depth-bg-blur-workflow.json) can also be saved and dropped into ComfyUI to load the workflow for further modifications.

# Example Image Workflows 

**Prerequisites**:

- Install [ComfyUI-DepthAnythingV2](https://github.com/kijai/ComfyUI-DepthAnythingV2).

The following example workflows are applied to this input image:

![input](./examples/input.jpg)

The output images can be saved and dropped into ComfyUI to load the workflows that created them.

**Background Blurring**

![depth-bg-blur-output](./examples/depth-bg-blur-output.png)

**Background Removal**

![depth-bg-black-output](./examples/depth-bg-black-output.png)

**Background Swapping**

![depth-bg-blue-output](./examples/depth-bg-blue-output.png)

This output just shows the background being swapped to a solid blue color, but in theory the background could be any image of your choice!

# Nodes

| Node            | Description                                                                                                 |
| --------------- | ----------------------------------------------------------------------------------------------------------- |
| BackgroundColor | Creates black/red/green/blue images with same dimensions as input images.                                   |
| Composite       | Creates composites of input foreground images, background images and foreground masks (CPU/CUDA supported). |
| GaussianBlur    | Applies gaussian blur to input images (CPU/CUDA supported).                                                 |