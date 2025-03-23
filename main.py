# -*- coding: utf-8 -*-
"""jax_gemma.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/jax_gemma.ipynb

# JAX Gemma on Colab TPU

Gemma is a family language models released by Google DeepMind in the paper [Gemma: Open Models Based on Gemini Research and Technology](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf). Gemma leverages the same research and technology used to create the Gemini models. It was trained using a dataset consisting of 6-trillion tokens, formed from web documents, code and mathematics. The result is a series of state-of-the-art models at the 2B and 7B scale, all of which are open-sourced and permissively licensed.

JAX is a Python library for high-performance machine-learning research. It uses the XLA (accelerator linear algebra) compiler to fuse sequences of operations together and run them at once. In the context of machine-learning, this allows JAX to execute blocks of modelling code very efficiently on modern hardware accelerators, like GPUs and TPUs, making it appealing for large-scale research and applications. Gemma itself was trained using JAX on TPU v5e cores.

Gemma has support in the 🤗 Transformers library from day-1, in both PyTorch and JAX. In this Google Colab, we'll showcase how to use the Gemma 2B model in JAX, leveraging the power of TPU v2 cores to run batched generation with a throughput of 475 tokens per second. For details on using Gemma in PyTorch on GPU, refer to the corresponding [documentation](https://huggingface.co/google/gemma-2b#usage).

## Connect to a TPU

First, we need to register our Hugging Face Hub token with our Google Colab runtime. Since the Gemma model is gated, our token will be checked when the model is downloaded to ensure we have accepted the terms-of-use.

To register your token, click the key symbol 🔑 in the left-hand pane of the screen. Name the secret `HF_TOKEN`, and copy a token from your Hugging Face Hub account: https://huggingface.co/settings/tokens. Your token should now be registered, allowing you to access the Gemma weights to this Colab session.

Next, we can connect to a TPU. Google Colab offers TPU v2's on its free tier, which we'll make use of for this notebook. If you have access to later generations of TPU, such as v3, v4 or v5e machines, you can run the same notebook and achieve significantly faster generation speeds.

To connect to a TPU v2, click on the button `Connect TPU` in the top right-hand corner of the screen. Once connected to a TPU, we need to initialise it with the `setup_tpu` method:
"""


import jax

print(jax.devices())

"""Great! We have a TPU device with 8 chips. In this notebook, we'll leverage *data parallelism* across these 8 chips, by sending 1/8 of our batch to each chip and generating the LLM outputs in parallel. TPUs shine at higher batch-sizes, where computations can easily be distributed across devices.

JAX comes pre-installed on Colab TPUs at the correct version, so there's no need to re-install it. However, we do need to updgrade Transformers to the latest version, since the Gemma model is only included in the latest library release:
"""

"""## Load the Model

The Gemma model can be loaded using the familiar [`from_pretrained`](https://huggingface.co/docs/transformers/v4.38.1/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method in Transformers. This method downloads the model weights from the Hugging Face Hub the first time it is called, and subsequently intialises the Gemma model using these weights.

We'll load the weights for the base 2B model by specifying the corresponding model-id on the Hugging Face Hub (["google/gemma-2b"](https://huggingface.co/google/gemma-2b)). You should ensure you have accepted the model terms-of-use before downloading the model. To do so, simply head to the model repository [here](https://huggingface.co/google/gemma-2b) and fill out the required fields.

We'll set the data-type (dtype) of the computation to bfloat16, which is faster than float32 while achieving similar accuracy. Finally, we'll set the flag `_do_init=False`, which improves the loading of large models in Transformers by skipping initialising a dummy set of parameters that are subsequently overriden by the pre-trained ones.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from flax.training.common_utils import shard

from transformers import FlaxGemmaForCausalLM, AutoTokenizer

print("Loading model")
model, params = FlaxGemmaForCausalLM.from_pretrained("google/gemma-2b-it", revision="flax", _do_init=False, dtype=jnp.bfloat16)

"""If you're coming from PyTorch, the only major difference in API is how the model and parameters are handled. PyTorch is a _stateful_ framework, in which the weights are stored within the model instance. In JAX, most transformations (notably `jax.jit`) require functions that are _stateless_, meaning that they have no side effects (see [Stateful Computations](https://jax.readthedocs.io/en/latest/jax-101/07-state.html) in JAX). Since Flax models are designed to work well with JAX transformations, they too are stateless. This means that the model weights are stored **outside** of the model definition, and need to be passed as an input during inference.

We see a warning that the model parameters were loaded in bfloat16 precision - this is fine since we also want to keep the parameters in bfloat16 for inference.

The corresponding tokenizer can now be loaded using a similar API:
"""
print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

"""## Define Inputs

Next, we'll define our text inputs. Since we have 8 TPU cores over which we want to perform data parallelism, we need our batch size to be a multiple of 8. This is to ensure that each TPU core receives the same amount of data (`bsz / 8` samples).

In this example, we'll define a single prompt and copy it 8 times, giving a total batch size of 8. The "per-device" batch size is then 1/8 of this, or 1. In practice, this is not very useful or realistic, since each of our parallel computations will be processing the same input. However, you're free to extend this to use more realistic prompts than the one given below. Just ensure that the resulting batch size is a multiple of 8.
"""

input_text = 8 * ["Write an article about AI"]

"""We can pre-process our input text to token ids using the tokenizer. TPUs expect inputs of static shape, so we'll define our maximum prompt length to be 64, and always pad our inputs to this sequence length:"""

max_input_length = 32
print("Tokenizing inputs")
inputs = tokenizer(
    input_text,
    padding="max_length",
    max_length=max_input_length,
    return_attention_mask=True,
    return_tensors="np",
)

"""We now need to copy the model parameters to each TPU core. Each core will hold it's own copy of the parameters, such that it can run a model generation in parallel with the others. Copying the parameters across devices is achieved simply with the [`replicate`](https://flax.readthedocs.io/en/latest/api_reference/flax.jax_utils.html#flax.jax_utils.replicate) method from Flax."""

params = jax_utils.replicate(params)

"""Similarly, we need to split (or shard) our inputs across TPU cores. Given input ids of shape `(bsz, seq_len)`, we'll shard them to `(num_devices, bsz / num_devices, seq_len)`. In doing so, each device can receive a batch of shape `(bsz / num_devices, seq_len)`, in order to leverage data parallelism across TPU cores. Sharding our inputs is achieved with the Flax helper function [`shard`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.common_utils.shard):"""

inputs = shard(inputs.data)

"""## Inference

We can now define our data-parallel method for inference. The Transformers [`generate`](https://huggingface.co/docs/transformers/v4.38.1/en/main_classes/text_generation#transformers.FlaxGenerationMixin.generate) method provides functionality for auto-regressive generation with batching, sampling, beam-search, etc. To reap the benefits of JAX, we'll compile the generate method end-to-end, such that the operations are fused into XLA-optimised kernels and executed efficiently on our hardware accelerator.

To achieve this, we'll wrap the `generate` method with the [`jax.pmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html) transformation. The `jax.pmap` transformation compiles the `generate` method with XLA, and prepares a function that can be executed in parallel across TPU devices.
"""

def generate(inputs, params, max_new_tokens):
    generated_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        params=params,
        max_new_tokens=max_new_tokens,
        do_sample=True,
    )
    return generated_ids.sequences

print("Compiling generate function")
p_generate = jax.pmap(
    generate, "inputs", in_axes=(0, 0, None,), out_axes=0, static_broadcasted_argnums=(2,)
)

"""The `in_axes` argument to `jax.pmap` defines which axis of the positional arguments to parallelise over. The `inputs` and `params` are parallised over their first axis, which we denote by `0`. The number of generated tokens is a integer, and is thus not parallelised over, which we denote by `None`.

Similarly, the `out_axes` argument defines which axis of the outputs are parallelised over, which in this case is the first axis.

The number of maximum generated tokens `max_new_tokens` undergoes control flow in `generate`. However, JIT requires concrete values to trace out the transformed function (see [Control Flow](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#control-flow) in JAX). Thus, we specify it as a static argument, meaning it is assigned as a concrete value in the control flow. This enables JIT to trace out the function for a specific value of `max_new_tokens`. The caveat is that each value of `max_new_tokens` requires it's own compilation, in-order to trace out the respective branch.

To avoid re-compiling the generate function for different values of `max_new_tokens`, we'll define it as a global variable here, and pass it to the generate function each time:
"""

max_new_tokens = 512

"""We can now compile our parallel generate function. This done automatically the first time the function is run and will take some time to complete (typically around 2-3 minutes). A good time to read the JAX [Quick Start Guide](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) if you haven't already!"""

_ = p_generate(inputs, params, max_new_tokens)

"""Now that the function is compiled, we can run it again much faster using the optimised kernels:"""

print("Running generate function")
start = time.time()
generated_ids = p_generate(inputs, params, max_new_tokens)
runtime = time.time() - start

"""The generate function returns a batch of generated token ids. To convert these to generated text, we can decode them using the tokenizer:"""

generated_ids = jax.device_get(generated_ids.reshape(-1, generated_ids.shape[-1]))
pred_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

"""So how fast was generation? Let's compute the number of tokens generated per-second (tok/s):"""

def compute_tok_per_s(input_ids, generated_ids, runtime):
    total_inputs = np.prod(input_ids.shape)
    total_outputs = np.prod(generated_ids.shape)
    tokens_generated = total_outputs - total_inputs
    tokens_per_s = tokens_generated / runtime
    return tokens_per_s

tok_per_s = compute_tok_per_s(inputs["input_ids"], generated_ids, runtime)

"""We can then print our runtime, tokens per second, and generated text to the console:"""

print(f"Runtime with pmap: {runtime}")
print(f"Tokens per second: {tok_per_s}")
print(pred_text[0])