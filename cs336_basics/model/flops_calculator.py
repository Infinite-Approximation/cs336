def calculate_flops(L, d, d_ff, V, N):
    # One TransformerBlock
    attn_proj_flops = 8 * L * (d**2)  # QKV (6Ld^2) + Output (2Ld^2)
    attn_score_sum_flops = 4 * (L**2) * d  # QK^T (2L^2d) + AV (2L^2d)
    mlp_flops = 6 * L * d * d_ff  # Gate (2Ld*d_ff) + Up (2Ld*d_ff) + Down (2Ld_ff*d)

    layer_total = attn_proj_flops + attn_score_sum_flops + mlp_flops
    all_layers_total = N * layer_total

    # LM Head
    lm_head_flops = 2 * L * d * V

    total_flops = all_layers_total + lm_head_flops

    # Breakdown percentages
    attn_total = N * (attn_proj_flops + attn_score_sum_flops)
    mlp_total = N * mlp_flops

    return {
        "Total": total_flops,
        "Attention": attn_total,
        "MLP": mlp_total,
        "LM Head": lm_head_flops,
        "Percentages": {
            "Attention": (attn_total / total_flops) * 100,
            "MLP": (mlp_total / total_flops) * 100,
            "LM Head": (lm_head_flops / total_flops) * 100,
        },
    }


configs = [
    {"name": "GPT-2 Small", "L": 1024, "d": 768, "d_ff": 3072, "V": 50257, "N": 12},
    {"name": "GPT-2 Medium", "L": 1024, "d": 1024, "d_ff": 4096, "V": 50257, "N": 24},
    {"name": "GPT-2 Large", "L": 1024, "d": 1280, "d_ff": 5120, "V": 50257, "N": 36},
    {"name": "GPT-2 XL", "L": 1024, "d": 1600, "d_ff": 6400, "V": 50257, "N": 48},
    {
        "name": "GPT-2 XL (Context 16k)",
        "L": 16384,
        "d": 1600,
        "d_ff": 6400,
        "V": 50257,
        "N": 48,
    },
]

print(
    f"{'Model':<20} | {'Total FLOPs':<15} | {'Attn %':<10} | {'MLP %':<10} | {'Head %':<10}"
)
print("-" * 75)
for cfg in configs:
    res = calculate_flops(cfg["L"], cfg["d"], cfg["d_ff"], cfg["V"], cfg["N"])
    print(
        f"{cfg['name']:<20} | {res['Total']:>15.2e} | {res['Percentages']['Attention']:>9.2f}% | {res['Percentages']['MLP']:>9.2f}% | {res['Percentages']['LM Head']:>9.2f}%"
    )
