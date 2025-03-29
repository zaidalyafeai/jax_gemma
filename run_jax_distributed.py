import jax
import jax.numpy as jnp
from transformers import FlaxGemmaForCausalLM, AutoTokenizer
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
import numpy as np
import os

model_name = "google/gemma-2b-it"
max_new_tokens = 4096
dtype = jnp.bfloat16
hf_token = os.environ["HF_TOKEN"]

# https://github.com/huggingface/transformers/issues/22224
model, params = FlaxGemmaForCausalLM.from_pretrained(model_name, revision="flax", _do_init=False, dtype=dtype, token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

def get_partition_rules():
    """Returns partitioning rules for the model parameters."""
    return [
        # Embeddings
        ("embed_tokens/embedding", P("mp", None)),
        # Attention layers
        ("self_attn/q_proj/kernel", P("mp", None)),
        ("self_attn/k_proj/kernel", P("mp", None)),
        ("self_attn/v_proj/kernel", P("mp", None)),
        ("self_attn/o_proj/kernel", P(None, "mp")),
        # MLP layers
        ("mlp/gate_proj/kernel", P("mp", None)),
        ("mlp/up_proj/kernel", P("mp", None)),
        ("mlp/down_proj/kernel", P(None, "mp")),
        # LM head
        ("lm_head/kernel", P(None, "mp")),
        ("input_layernorm/weight", P(None)),
        ("post_attention_layernorm/weight", P(None)),
        ("norm/weight", P(None)),
        # LM head
        ("lm_head/kernel", P(None, "mp")),
        # Default rule
        (".*", P(None)),
    ]

def create_sharding_specs(params):

    # Get available JAX devices and create mesh
    devices = jax.devices()
    device_mesh = np.array(devices).reshape(-1)  # 1D mesh
    mesh = Mesh(device_mesh, ("mp",))
    partition_rules = get_partition_rules()
    def assign_spec(path, _):
        # Create a slash-separated string from the tuple path
        path_str = "/".join(map(str, path)).replace('[','').replace(']','').replace("'/'", "/")
        # Look for a matching partition rule
        for rule_path, spec in partition_rules:
            if rule_path in path_str:
                return NamedSharding(mesh, spec)
        return None
    return jax.tree_util.tree_map_with_path(assign_spec, params)

sharding_specs = create_sharding_specs(params)

params = jax.tree_util.tree_map(
    lambda x, spec: jax.device_put(x, spec), params, sharding_specs
)

prompt = ["write an article about AI"] * 8
inputs = tokenizer(prompt, return_tensors="np")
input_ids = jnp.array(inputs["input_ids"])
print(input_ids)

import time
def generate(input_ids, params, max_new_tokens):
    # Reshape input_ids to (1, sequence_length)
    input_ids = input_ids[jnp.newaxis, :]
    generated_ids = model.generate(
        input_ids=input_ids,
        params=params,
        max_new_tokens=max_new_tokens,
        do_sample=True,
    )
    return generated_ids.sequences.squeeze(1)

p_generate = jax.pmap(
    generate,
    in_axes=(0, None, None),  # Map only the inputs, not params or max_new_tokens
    out_axes=0, # squeeze the output
    static_broadcasted_argnums=(2,) # broadcast the max_new_tokens
)

start = time.time()
generated_ids = p_generate(input_ids, params, max_new_tokens)
pred_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
runtime = time.time() - start
num_tokens = generated_ids.shape[0] * generated_ids.shape[1]
print(f"Throughput: {num_tokens / runtime:.2f} tokens/second")
