#!/usr/bin/env python3

from pathlib import Path
import numpy as np

INPUT_DIRECTORY = Path(__file__).parent.parent
OUTPUT_DIRECTORY = INPUT_DIRECTORY / 'onboard'
NUM_LAYERS = 12
NUM_TASKS = 2  # Assuming 2 tasks for MoE gates
NUM_EXPERTS = 16
FEATURE_DIM = 192
MLP_DIM = 768
EXPERT_DIM = 384
INPUT_CHANNELS = 3
PATCH_HEIGHT = 16
PATCH_WIDTH = 16
NUM_HEADS = 3

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Function to convert NumPy array to C++ header
def _array_to_cpp_initializer(arr):
    if arr.ndim == 1:
        return '{ ' + ', '.join(f'{x}f' for x in arr) + ' }'
    else:
        return '{ ' + ',\n  '.join(_array_to_cpp_initializer(a) for a in arr) + ' }'

def npy_to_cpp_header(npy_file, header_file, var_name, dtype='float'):
    data = np.load(npy_file)
    shape = data.shape
    ndim = data.ndim
    with open(header_file, 'w') as f:
        f.write('#pragma once\n\n')
        f.write('#include <cstddef>\n')
        f.write('#include <cstdint>\n\n')
        f.write(f'const size_t {var_name}_shape[{ndim}] = {{ {", ".join(map(str, shape))} }};\n')
        f.write(f'{dtype} {var_name}')
        for dim in shape:
            f.write(f'[{dim}]')
        f.write(' = ')
        f.write(_array_to_cpp_initializer(data))
        f.write(';\n')

# Process image input
np.save(
    OUTPUT_DIRECTORY / 'images_array.npy',
    np.fromfile(
        INPUT_DIRECTORY / 'image.float32.bin',
        dtype=np.float32,
    ).reshape((1, INPUT_CHANNELS, 128, 256))  # Adjust image dimensions as per your data
)

# Process embedding weights and biases
np.save(
    OUTPUT_DIRECTORY / 'patch_embed_weights_array.npy',
    np.fromfile(
        INPUT_DIRECTORY / 'patch_embed_weight.float32.bin',
        dtype=np.float32,
    ).reshape((FEATURE_DIM, INPUT_CHANNELS, PATCH_HEIGHT, PATCH_WIDTH))
)
np.save(
    OUTPUT_DIRECTORY / 'patch_embed_bias_array.npy',
    np.fromfile(
        INPUT_DIRECTORY / 'patch_embed_bias.float32.bin',
        dtype=np.float32,
    ).reshape((FEATURE_DIM,))
)
np.save(
    OUTPUT_DIRECTORY / 'pos_embed_array.npy',
    np.fromfile(
        INPUT_DIRECTORY / 'pos_embed.float32.bin',
        dtype=np.float32,
    ).reshape((-1, FEATURE_DIM))
)

# Process per-layer weights and biases
norm_weights = []
norm_bias = []
attn_weights = []
attn_bias = []
vit_weights_l1 = []
vit_bias_l1 = []
vit_weights_l2 = []
vit_bias_l2 = []
moe_weights_l1 = []
moe_bias_l1 = []
moe_weights_l2 = []
moe_bias_l2 = []
moe_w_gate = []

for l in range(NUM_LAYERS):
    # Norm weights and biases
    norm_weights.append([
        np.fromfile(
            INPUT_DIRECTORY / f'l{l}_norm1_weight.float32.bin',
            dtype=np.float32,
        ).reshape((FEATURE_DIM,)),
        np.fromfile(
            INPUT_DIRECTORY / f'l{l}_norm2_weight.float32.bin',
            dtype=np.float32,
        ).reshape((FEATURE_DIM,))
    ])
    norm_bias.append([
        np.fromfile(
            INPUT_DIRECTORY / f'l{l}_norm1_bias.float32.bin',
            dtype=np.float32,
        ).reshape((FEATURE_DIM,)),
        np.fromfile(
            INPUT_DIRECTORY / f'l{l}_norm2_bias.float32.bin',
            dtype=np.float32,
        ).reshape((FEATURE_DIM,))
    ])

    # Attention weights and biases
    qkv_weight = np.fromfile(
        INPUT_DIRECTORY / f'l{l}_qkv_weight.float32.bin',
        dtype=np.float32,
    ).reshape((NUM_HEADS * FEATURE_DIM, FEATURE_DIM))
    qkv_weight_split = np.split(qkv_weight, NUM_HEADS)
    attn_proj_weight = np.fromfile(
        INPUT_DIRECTORY / f'l{l}_attn_proj_weight.float32.bin',
        dtype=np.float32,
    ).reshape((FEATURE_DIM, FEATURE_DIM))
    attn_weights.append(np.array(qkv_weight_split + [attn_proj_weight]))

    qkv_bias = np.fromfile(
        INPUT_DIRECTORY / f'l{l}_qkv_bias.float32.bin',
        dtype=np.float32,
    ).reshape((NUM_HEADS * FEATURE_DIM,))
    qkv_bias_split = np.split(qkv_bias, NUM_HEADS)
    attn_proj_bias = np.fromfile(
        INPUT_DIRECTORY / f'l{l}_attn_proj_bias.float32.bin',
        dtype=np.float32,
    ).reshape((FEATURE_DIM,))
    attn_bias.append(np.array(qkv_bias_split + [attn_proj_bias]))

    if l % 2 == 0:
        # ViT layers (even layers)
        vit_weights_l1.append(
            np.fromfile(
                INPUT_DIRECTORY / f'l{l}_fc1_weight.float32.bin',
                dtype=np.float32,
            ).reshape((MLP_DIM, FEATURE_DIM))
        )
        vit_bias_l1.append(
            np.fromfile(
                INPUT_DIRECTORY / f'l{l}_fc1_bias.float32.bin',
                dtype=np.float32,
            ).reshape((MLP_DIM,))
        )
        vit_weights_l2.append(
            np.fromfile(
                INPUT_DIRECTORY / f'l{l}_fc2_weight.float32.bin',
                dtype=np.float32,
            ).reshape((FEATURE_DIM, MLP_DIM))
        )
        vit_bias_l2.append(
            np.fromfile(
                INPUT_DIRECTORY / f'l{l}_fc2_bias.float32.bin',
                dtype=np.float32,
            ).reshape((FEATURE_DIM,))
        )
    else:
        # MoE layers (odd layers)
        moe_weights_l1.append(
            np.fromfile(
                INPUT_DIRECTORY / f'l{l}_htoh4_weight.float32.bin',
                dtype=np.float32,
            ).reshape((NUM_EXPERTS, EXPERT_DIM, FEATURE_DIM))
        )
        moe_bias_l1.append(
            np.fromfile(
                INPUT_DIRECTORY / f'l{l}_htoh4_bias.float32.bin',
                dtype=np.float32,
            ).reshape((NUM_EXPERTS, EXPERT_DIM))
        )
        moe_weights_l2.append(
            np.fromfile(
                INPUT_DIRECTORY / f'l{l}_h4toh_weight.float32.bin',
                dtype=np.float32,
            ).reshape((NUM_EXPERTS, FEATURE_DIM, EXPERT_DIM))
        )
        moe_bias_l2.append(
            np.fromfile(
                INPUT_DIRECTORY / f'l{l}_h4toh_bias.float32.bin',
                dtype=np.float32,
            ).reshape((NUM_EXPERTS, FEATURE_DIM))
        )
        # MoE gates per task
        gates_per_task = []
        for task in range(NUM_TASKS):
            gates_per_task.append(
                np.fromfile(
                    INPUT_DIRECTORY / f'l{l}_w_gate_T_task{task}.float32.bin',
                    dtype=np.float32,
                ).reshape((NUM_EXPERTS, FEATURE_DIM))
            )
        moe_w_gate.append(gates_per_task)

# Convert lists to arrays and save
np.save(OUTPUT_DIRECTORY / 'norm_weights_array.npy', np.array(norm_weights))
np.save(OUTPUT_DIRECTORY / 'norm_bias_array.npy', np.array(norm_bias))
np.save(OUTPUT_DIRECTORY / 'attn_weights_array.npy', np.array(attn_weights))
np.save(OUTPUT_DIRECTORY / 'attn_bias_array.npy', np.array(attn_bias))
np.save(OUTPUT_DIRECTORY / 'vit_weights_l1_array.npy', np.array(vit_weights_l1))
np.save(OUTPUT_DIRECTORY / 'vit_bias_l1_array.npy', np.array(vit_bias_l1))
np.save(OUTPUT_DIRECTORY / 'vit_weights_l2_array.npy', np.array(vit_weights_l2))
np.save(OUTPUT_DIRECTORY / 'vit_bias_l2_array.npy', np.array(vit_bias_l2))
np.save(OUTPUT_DIRECTORY / 'moe_weights_l1_array.npy', np.array(moe_weights_l1))
np.save(OUTPUT_DIRECTORY / 'moe_bias_l1_array.npy', np.array(moe_bias_l1))
np.save(OUTPUT_DIRECTORY / 'moe_weights_l2_array.npy', np.array(moe_weights_l2))
np.save(OUTPUT_DIRECTORY / 'moe_bias_l2_array.npy', np.array(moe_bias_l2))
np.save(OUTPUT_DIRECTORY / 'moe_w_gate_per_task_array.npy', np.array(moe_w_gate))

# Process reference output if available
if (INPUT_DIRECTORY / f'l{NUM_LAYERS - 1}_x_post_moe.float32.bin').exists():
    np.save(
        OUTPUT_DIRECTORY / 'reference_x_array.npy',
        np.fromfile(
            INPUT_DIRECTORY / f'l{NUM_LAYERS - 1}_x_post_moe.float32.bin',
            dtype=np.float32,
        ).reshape((-1, FEATURE_DIM))
    )
elif (INPUT_DIRECTORY / f'l{NUM_LAYERS - 1}_x_post_mlp.float32.bin').exists():
    np.save(
        OUTPUT_DIRECTORY / 'reference_x_array.npy',
        np.fromfile(
            INPUT_DIRECTORY / f'l{NUM_LAYERS - 1}_x_post_mlp.float32.bin',
            dtype=np.float32,
        ).reshape((-1, FEATURE_DIM))
    )

# Generate C++ headers
files_to_convert = [
    ('images_array.npy', 'image.h', 'images_array'),
    ('patch_embed_weights_array.npy', 'patch_embed_weights.h', 'patch_embed_weights_array'),
    ('patch_embed_bias_array.npy', 'patch_embed_bias.h', 'patch_embed_bias_array'),
    ('pos_embed_array.npy', 'pos_embed.h', 'pos_embed_array'),
    ('norm_weights_array.npy', 'norm_weights.h', 'norm_weights_array'),
    ('norm_bias_array.npy', 'norm_bias.h', 'norm_bias_array'),
    ('attn_weights_array.npy', 'attn_weights.h', 'attn_weights_array'),
    ('attn_bias_array.npy', 'attn_bias.h', 'attn_bias_array'),
    ('vit_weights_l1_array.npy', 'vit_weights_l1.h', 'vit_weights_l1_array'),
    ('vit_bias_l1_array.npy', 'vit_bias_l1.h', 'vit_bias_l1_array'),
    ('vit_weights_l2_array.npy', 'vit_weights_l2.h', 'vit_weights_l2_array'),
    ('vit_bias_l2_array.npy', 'vit_bias_l2.h', 'vit_bias_l2_array'),
    ('moe_weights_l1_array.npy', 'moe_weights_l1.h', 'moe_weights_l1_array'),
    ('moe_bias_l1_array.npy', 'moe_bias_l1.h', 'moe_bias_l1_array'),
    ('moe_weights_l2_array.npy', 'moe_weights_l2.h', 'moe_weights_l2_array'),
    ('moe_bias_l2_array.npy', 'moe_bias_l2.h', 'moe_bias_l2_array'),
    ('moe_w_gate_per_task_array.npy', 'moe_w_gate_per_task.h', 'moe_w_gate_per_task_array'),
    ('reference_x_array.npy', 'reference_x.h', 'reference_x_array'),
]

for npy_name, header_name, var_name in files_to_convert:
    npy_file = OUTPUT_DIRECTORY / npy_name
    header_file = OUTPUT_DIRECTORY / header_name
    print(f'Converting {npy_file} to {header_file}')
    npy_to_cpp_header(npy_file, header_file, var_name, dtype='float')
