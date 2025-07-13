import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# model_sizes_path = 'sizes_formatted.csv'
model_sizes_path = 'sizes.csv'
vram_path = 'vram_stats.csv'

if __name__ == "__main__":
    model_sizes = pd.read_csv(model_sizes_path)
    # model_sizes['layers'] = modle
    vram_stats = pd.read_csv(vram_path)

    print(model_sizes)
    print(vram_stats)
    
    # join the two dataframes on the num_hidden_layers and hidden_size_multiplier columns
    joined_df = pd.merge(model_sizes, vram_stats, on=['layers', 'hidden_size_multiplier'], how='outer')
    print(joined_df)

    final_columns = ['layers', 'hidden_size_multiplier', 'parameters', 'max_used_mb']
    joined_df = joined_df[final_columns]
    # joined_df = joined_df[joined_df['hidden_size_multiplier'] == 1]
    print(joined_df)

    # save the joined dataframe to a csv file
    joined_df.to_csv('joined_df.csv', index=False)

    # plot parameters vs vram usanage
    plt.scatter(joined_df['parameters'], joined_df['max_used_mb'])
    plt.xlabel('Parameters')
    plt.ylabel('VRAM Usage (MB)')
    plt.title('Parameters vs VRAM Usage')
    plt.savefig('parameters_vs_vram_usage.png')

    # find relationship between parameters and vram usage
    
