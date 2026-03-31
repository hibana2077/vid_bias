# VideoMAE

## Overview

The VideoMAE model was proposed in [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://huggingface.co/papers/2203.12602) by Zhan Tong, Yibing Song, Jue Wang, Limin Wang.
VideoMAE extends masked auto encoders ([MAE](vit_mae)) to video, claiming state-of-the-art performance on several video classification benchmarks.

The abstract from the paper is the following:

*Pre-training video transformers on extra large-scale datasets is generally required to achieve premier performance on relatively small datasets. In this paper, we show that video masked autoencoders (VideoMAE) are data-efficient learners for self-supervised video pre-training (SSVP). We are inspired by the recent ImageMAE and propose customized video tube masking and reconstruction. These simple designs turn out to be effective for overcoming information leakage caused by the temporal correlation during video reconstruction. We obtain three important findings on SSVP: (1) An extremely high proportion of masking ratio (i.e., 90% to 95%) still yields favorable performance of VideoMAE. The temporally redundant video content enables higher masking ratio than that of images. (2) VideoMAE achieves impressive results on very small datasets (i.e., around 3k-4k videos) without using any extra data. This is partially ascribed to the challenging task of video reconstruction to enforce high-level structure learning. (3) VideoMAE shows that data quality is more important than data quantity for SSVP. Domain shift between pre-training and target datasets are important issues in SSVP. Notably, our VideoMAE with the vanilla ViT backbone can achieve 83.9% on Kinects-400, 75.3% on Something-Something V2, 90.8% on UCF101, and 61.1% on HMDB51 without using any extra data.*

 VideoMAE pre-training. Taken from the original paper. 

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/MCG-NJU/VideoMAE).

## Using Scaled Dot Product Attention (SDPA)

PyTorch includes a native scaled dot-product attention (SDPA) operator as part of `torch.nn.functional`. This function
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the
[official documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
or the [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
page for more information.

SDPA is used by default for `torch>=2.1.1` when an implementation is available, but you may also set
`attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.

```py
from transformers import VideoMAEForVideoClassification
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", attn_implementation="sdpa", dtype=torch.float16)
...
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

On a local benchmark (A100-40GB, PyTorch 2.3.0, OS Ubuntu 22.04) with `float32` and `MCG-NJU/videomae-base-finetuned-kinetics` model, we saw the following speedups during inference.

|   Batch size |   Average inference time (ms), eager mode |   Average inference time (ms), sdpa model |   Speed up, Sdpa / Eager (x) |
|--------------|-------------------------------------------|-------------------------------------------|------------------------------|
|            1 |                                        37 |                                        10 |                      3.7  |
|            2 |                                        24 |                                        18 |                      1.33 |
|            4 |                                        43 |                                        32 |                      1.34 |
|            8 |                                        84 |                                        60 |                      1.4  |

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with VideoMAE. If
you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll
review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

**Video classification**

- [A notebook](https://github.com/huggingface/notebooks/blob/main/examples/video_classification.ipynb) that shows how
to fine-tune a VideoMAE model on a custom dataset.
- [Video classification task guide](../tasks/video_classification)
- [A 🤗 Space](https://huggingface.co/spaces/sayakpaul/video-classification-ucf101-subset) showing how to perform inference with a video classification model.

## VideoMAEConfig[[transformers.VideoMAEConfig]]

#### transformers.VideoMAEConfig[[transformers.VideoMAEConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/videomae/configuration_videomae.py#L24)

This is the configuration class to store the configuration of a VideoMAEModel. It is used to instantiate a Videomae
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [MCG-NJU/videomae-base](https://huggingface.co/MCG-NJU/videomae-base)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import VideoMAEConfig, VideoMAEModel

>>> # Initializing a VideoMAE videomae-base style configuration
>>> configuration = VideoMAEConfig()

>>> # Randomly initializing a model from the configuration
>>> model = VideoMAEModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

image_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `224`) : The size (resolution) of each image.

patch_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `16`) : The size (resolution) of each patch.

num_channels (`int`, *optional*, defaults to `3`) : The number of input channels.

num_frames (`int`, *optional*, defaults to 16) : The number of frames in each video.

tubelet_size (`int`, *optional*, defaults to 2) : The number of tubelets.

hidden_size (`int`, *optional*, defaults to `768`) : Dimension of the hidden representations.

num_hidden_layers (`int`, *optional*, defaults to `12`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `12`) : Number of attention heads for each attention layer in the Transformer decoder.

intermediate_size (`int`, *optional*, defaults to `3072`) : Dimension of the MLP representations.

hidden_act (`str`, *optional*, defaults to `gelu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

hidden_dropout_prob (`float`, *optional*, defaults to `0.0`) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_probs_dropout_prob (`float`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to `1e-12`) : The epsilon used by the layer normalization layers.

qkv_bias (`bool`, *optional*, defaults to `True`) : Whether to add a bias to the queries, keys and values.

use_mean_pooling (`bool`, *optional*, defaults to `True`) : Whether to mean pool the final hidden states instead of using the final hidden state of the [CLS] token.

decoder_num_attention_heads (`int`, *optional*, defaults to 6) : Number of attention heads for each attention layer in the decoder.

decoder_hidden_size (`int`, *optional*, defaults to 384) : Dimensionality of the decoder.

decoder_num_hidden_layers (`int`, *optional*, defaults to 4) : Number of hidden layers in the decoder.

decoder_intermediate_size (`int`, *optional*, defaults to 1536) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the decoder.

norm_pix_loss (`bool`, *optional*, defaults to `True`) : Whether to normalize the target patch pixels.

## VideoMAEImageProcessor[[transformers.VideoMAEImageProcessor]]

#### transformers.VideoMAEImageProcessor[[transformers.VideoMAEImageProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/videomae/image_processing_videomae.py#L35)

Constructs a VideoMAEImageProcessor image processor.

preprocesstransformers.VideoMAEImageProcessor.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/videomae/image_processing_videomae.py#L53[{"name": "videos", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.processing_utils.ImagesKwargs]"}]- **videos** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`) --
  Video or batch of videos to preprocess. Expects a single video (list of frames) or a batch of videos
  (list of list of frames). Each frame can be a PIL image, numpy array, or torch tensor with pixel values
  ranging from 0 to 255. If passing in frames with pixel values between 0 and 1, set `do_rescale=False`.
- **return_tensors** (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) --
  Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.
- ****kwargs** ([ImagesKwargs](/docs/transformers/main/en/main_classes/processors#transformers.ImagesKwargs), *optional*) --
  Additional image preprocessing options. Model-specific kwargs are listed above; see the TypedDict class
  for the complete list of supported arguments.0`~image_processing_base.BatchFeature`- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

**Parameters:**

- ****kwargs** ([ImagesKwargs](/docs/transformers/main/en/main_classes/processors#transformers.ImagesKwargs), *optional*) : Additional image preprocessing options. Model-specific kwargs are listed above; see the TypedDict class for the complete list of supported arguments.

**Returns:**

``~image_processing_base.BatchFeature``

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## VideoMAEImageProcessorPil[[transformers.VideoMAEImageProcessorPil]]

#### transformers.VideoMAEImageProcessorPil[[transformers.VideoMAEImageProcessorPil]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/videomae/image_processing_pil_videomae.py#L33)

Constructs a VideoMAEImageProcessor image processor.

preprocesstransformers.VideoMAEImageProcessorPil.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/videomae/image_processing_pil_videomae.py#L51[{"name": "videos", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.processing_utils.ImagesKwargs]"}]- **videos** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`) --
  Video or batch of videos to preprocess. Expects a single video (list of frames) or a batch of videos
  (list of list of frames). Each frame can be a PIL image, numpy array, or torch tensor with pixel values
  ranging from 0 to 255. If passing in frames with pixel values between 0 and 1, set `do_rescale=False`.
- **return_tensors** (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) --
  Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.
- ****kwargs** ([ImagesKwargs](/docs/transformers/main/en/main_classes/processors#transformers.ImagesKwargs), *optional*) --
  Additional image preprocessing options. Model-specific kwargs are listed above; see the TypedDict class
  for the complete list of supported arguments.0`~image_processing_base.BatchFeature`- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

**Parameters:**

- ****kwargs** ([ImagesKwargs](/docs/transformers/main/en/main_classes/processors#transformers.ImagesKwargs), *optional*) : Additional image preprocessing options. Model-specific kwargs are listed above; see the TypedDict class for the complete list of supported arguments.

**Returns:**

``~image_processing_base.BatchFeature``

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## VideoMAEVideoProcessor[[transformers.VideoMAEVideoProcessor]]

#### transformers.VideoMAEVideoProcessor[[transformers.VideoMAEVideoProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/videomae/video_processing_videomae.py#L20)

preprocesstransformers.VideoMAEVideoProcessor.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/videomae/video_processing_videomae.py#L36[{"name": "videos", "val": ""}, {"name": "**kwargs", "val": ""}]

## VideoMAEModel[[transformers.VideoMAEModel]]

#### transformers.VideoMAEModel[[transformers.VideoMAEModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/videomae/modeling_videomae.py#L407)

The bare Videomae Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.VideoMAEModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/videomae/modeling_videomae.py#L426[{"name": "pixel_values", "val": ": FloatTensor"}, {"name": "bool_masked_pos", "val": ": torch.BoolTensor | None = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [VideoMAEImageProcessor](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEImageProcessor). See `VideoMAEImageProcessor.__call__()` for details (`processor_class` uses
  [VideoMAEImageProcessor](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEImageProcessor) for processing images).
- **bool_masked_pos** (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Each video in the
  batch must have the same number of masked patches. If `None`, then all patches are considered. Sequence
  length is `(num_frames // tubelet_size) * (image_size // patch_size) ** 2`.0[BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`A [BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VideoMAEConfig](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEConfig)) and inputs.
The [VideoMAEModel](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Examples:

```python
>>> import torch
>>> from transformers import VideoMAEVideoProcessor, VideoMAEModel
>>> from huggingface_hub import hf_hub_download

>>> # replace this with your own video file
>>> video_path = hf_hub_download(
...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
... )

>>> video_processor = VideoMAEVideoProcessor.from_pretrained("MCG-NJU/videomae-base")
>>> model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

>>> # prepare video for the model
>>> inputs = video_processor(video_path, return_tensors="pt")

>>> # forward pass
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 1568, 768]
```

**Parameters:**

config ([VideoMAEModel](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)``

A [BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VideoMAEConfig](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEConfig)) and inputs.

## VideoMAEForPreTraining[[transformers.VideoMAEForPreTraining]]

`VideoMAEForPreTraining` includes the decoder on top for self-supervised pre-training.

#### transformers.VideoMAEForPreTraining[[transformers.VideoMAEForPreTraining]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/videomae/modeling_videomae.py#L520)

The VideoMAE Model transformer with the decoder on top for self-supervised pre-training.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.VideoMAEForPreTraining.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/videomae/modeling_videomae.py#L538[{"name": "pixel_values", "val": ": FloatTensor"}, {"name": "bool_masked_pos", "val": ": BoolTensor"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [VideoMAEImageProcessor](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEImageProcessor). See `VideoMAEImageProcessor.__call__()` for details (`processor_class` uses
  [VideoMAEImageProcessor](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEImageProcessor) for processing images).
- **bool_masked_pos** (`torch.BoolTensor` of shape `(batch_size, sequence_length)`) --
  Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Each video in the
  batch must have the same number of masked patches. Sequence length is `(num_frames // tubelet_size) *
  (image_size // patch_size) ** 2`.0`VideoMAEForPreTrainingOutput` or `tuple(torch.FloatTensor)`A `VideoMAEForPreTrainingOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VideoMAEConfig](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEConfig)) and inputs.
The [VideoMAEForPreTraining](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`) -- Pixel reconstruction loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, patch_size ** 2 * num_channels)`) -- Pixel reconstruction logits.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Examples:
```python
>>> from transformers import AutoImageProcessor, VideoMAEForPreTraining
>>> import numpy as np
>>> import torch

>>> num_frames = 16
>>> video = list(np.random.randint(0, 256, (num_frames, 3, 224, 224)))

>>> image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
>>> model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")

>>> pixel_values = image_processor(video, return_tensors="pt").pixel_values

>>> num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
>>> seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
>>> bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

>>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
>>> loss = outputs.loss
```

**Parameters:**

config ([VideoMAEForPreTraining](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEForPreTraining)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``VideoMAEForPreTrainingOutput` or `tuple(torch.FloatTensor)``

A `VideoMAEForPreTrainingOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VideoMAEConfig](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEConfig)) and inputs.

## VideoMAEForVideoClassification[[transformers.VideoMAEForVideoClassification]]

#### transformers.VideoMAEForVideoClassification[[transformers.VideoMAEForVideoClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/videomae/modeling_videomae.py#L689)

VideoMAE Model transformer with a video classification head on top (a linear layer on top of the average pooled hidden
states of all tokens) e.g. for ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.VideoMAEForVideoClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/videomae/modeling_videomae.py#L703[{"name": "pixel_values", "val": ": torch.Tensor | None = None"}, {"name": "labels", "val": ": torch.Tensor | None = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [VideoMAEImageProcessor](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEImageProcessor). See `VideoMAEImageProcessor.__call__()` for details (`processor_class` uses
  [VideoMAEImageProcessor](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEImageProcessor) for processing images).
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).0[ImageClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`A [ImageClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VideoMAEConfig](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEConfig)) and inputs.
The [VideoMAEForVideoClassification](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEForVideoClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Examples:

```python
>>> import torch
>>> from transformers import VideoMAEVideoProcessor, VideoMAEForVideoClassification
>>> from huggingface_hub import hf_hub_download

>>> # replace this with your own video file
>>> video_path = hf_hub_download(
...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
... )

>>> video_processor = VideoMAEVideoProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
>>> model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

>>> inputs = video_processor(video_path, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)
...     logits = outputs.logits

>>> # model predicts one of the 400 Kinetics-400 classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
eating spaghetti
```

**Parameters:**

config ([VideoMAEForVideoClassification](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEForVideoClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[ImageClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)``

A [ImageClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VideoMAEConfig](/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEConfig)) and inputs.

