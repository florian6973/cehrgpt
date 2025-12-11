from tqdm import tqdm


def get_param_size(n_layers, d_model, time_size, vocab_size, position_size):
    main_params =12 * n_layers * d_model ** 2 + d_model ** 2 + d_model + d_model * time_size / 3
    bias_norm_params = 2*d_model + 13 * n_layers * d_model + d_model + 2
    embedding_params = d_model * (position_size + vocab_size)
    main_params = int(main_params)
    bias_norm_params = int(bias_norm_params)
    embedding_params = int(embedding_params)
    # total_params = int(total_params)
    total_params = main_params + bias_norm_params + embedding_params
    return main_params, bias_norm_params, embedding_params, total_params

import pandas as pd
results=[]
# df = pd.DataFrame(columns=["d_model", "n_layers", "main_params", "bias_norm_params", "embedding_params", "total_params"])
for d_model in tqdm([96, 192, 388, 320, 512, 640, 768, 896, 1024, 1280, 1536, 2298, 3534]):
    for n_layers in [4, 6, 8, 12, 14, 16, 18, 20, 24, 28, 32]:
        if d_model % 6 != 0:
            print(f"d_model {d_model} is not divisible by 6 - no n_heads possible")
            continue
        main_params, bias_norm_params, embedding_params, total_params = get_param_size(n_layers, d_model, 78, 6516, 2048)
        # df = df.append({"d_model": d_model, "n_layers": n_layers, "main_params": main_params, "bias_norm_params": bias_norm_params, "embedding_params": embedding_params, "total_params": total_params}, ignore_index=True)
        total_params_mb = total_params / 1e6
        results.append({"d_model": d_model, "n_layers": n_layers, "main_params": main_params, "bias_norm_params": bias_norm_params, "embedding_params": embedding_params, "total_params": total_params, "total_params_mb": total_params_mb})
df = pd.DataFrame(results)
df.to_csv("param_size.csv", index=False)
df = df[["d_model", "n_layers", "total_params_mb"]]
df = df.round(3)
df.columns = ["Hidden dim", "Layers", "Params (millions)"]
df.to_latex(buf="param_size.tex", index=False, longtable=True, caption="Parameter counts for CEHR-GPT models", label="tab:param_size_cehrgpt", float_format="%.3f")


# print(get_param_size(12, 768, 78, 6516, 2048))