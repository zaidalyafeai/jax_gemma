import os
import jax
import time
import keras
import keras_nlp

print(jax.devices())

# The Keras 3 distribution API is only implemented for the JAX backend for now
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

"""## Load model"""

keras.config.set_floatx("bfloat16")
device_mesh = keras.distribution.DeviceMesh(
    (1, 8),
    ["batch", "model"],
    devices=keras.distribution.list_devices())

"""`LayoutMap` from the distribution API specifies how the weights and tensors should be sharded or replicated, using the string keys, for example, `token_embedding/embeddings` below, which are treated like regex to match tensor paths. Matched tensors are sharded with model dimensions (8 TPUs); others will be fully replicated."""

model_dim = "model"

layout_map = keras.distribution.LayoutMap(device_mesh)

# Weights that match 'token_embedding/embeddings' will be sharded on 8 TPUs
layout_map["token_embedding/embeddings"] = (None, model_dim)

# Regex to match against the query, key and value matrices in the decoder
# attention layers
layout_map["decoder_block.*attention.*(query|key|value).*kernel"] = (
    None, model_dim, None)
layout_map["decoder_block.*attention_output.*kernel"] = (
    None, None, model_dim)
layout_map["decoder_block.*ffw_gating.*kernel"] = (model_dim, None)
layout_map["decoder_block.*ffw_linear.*kernel"] = (None, model_dim)

"""`ModelParallel` allows you to shard model weights or activation tensors across all devcies on the `DeviceMesh`. In this case, some of the Gemma 7B model weights are sharded across 8 TPU cores according to the `layout_map` defined above. Now load the model in the distributed way."""

model_parallel = keras.distribution.ModelParallel(
    device_mesh, layout_map, batch_dim_name="batch")

keras.distribution.set_distribution(model_parallel)

# Download the Gemma 7B model.
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_instruct_7b_en")

# Add timing and token counting

# Function to measure tokens per second
def measure_tps(prompt, max_length):
    start_time = time.time()
    output = gemma_lm.generate(prompt, max_length=max_length)
    end_time = time.time()
    
    # Calculate total tokens (input + output)
    total_tokens = len(gemma_lm.tokenizer.encode(prompt)) + len(gemma_lm.tokenizer.encode(output))
    elapsed_time = end_time - start_time
    tps = total_tokens / elapsed_time
    
    print(f"\nOutput: {output}")
    print(f"Total tokens: {total_tokens}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Tokens per second: {tps:.2f}")
    
    return output

# Test the model with TPS measurement
prompt = "Best comedy movies: "
measure_tps(prompt, max_length=64)
