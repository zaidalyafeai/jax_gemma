import os
# The Keras 3 distribution API is only implemented for the JAX backend for now
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import time
import keras
import keras_nlp

print(jax.devices())

"""## Load model"""

keras.config.set_floatx("bfloat16")
# Create a device mesh with (1, 8) shape so that the weights are sharded across
# all 8 TPUs.
device_mesh = keras.distribution.DeviceMesh(
    (1, 8),
    ["batch", "model"],
    devices=keras.distribution.list_devices())

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

model_parallel = keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="batch")

keras.distribution.set_distribution(model_parallel)

# Download the Gemma 7B model.
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_instruct_2b_en")
# Add timing and token counting

# Function to measure tokens per second
def measure_tps(prompt, max_length):
    start_time = time.time()
    # Add attention mask preprocessing
    output = gemma_lm.generate(
        prompt, 
        max_length=max_length,
        sequence_length=64,  # Explicitly set sequence length
    )
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

# Test with a shorter sequence first
prompt = "Best comedy movies: "
measure_tps(prompt, max_length=32)  # Reduced from 64 to 32 for initial testing
