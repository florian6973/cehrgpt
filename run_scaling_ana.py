import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# model_sizes_path = 'sizes_formatted.csv'
model_sizes_path = 'sizes.csv'
vram_path = 'vram_stats.csv'

if __name__ == "__main__":
    model_sizes = pd.read_csv(model_sizes_path)
    # # model_sizes['layers'] = modle
    # vram_stats = pd.read_csv(vram_path)

    # print(model_sizes)
    # print(vram_stats)
    
    # # join the two dataframes on the num_hidden_layers and hidden_size_multiplier columns
    # joined_df = pd.merge(model_sizes, vram_stats, on=['layers', 'hidden_size_multiplier'], how='outer')
    # print(joined_df)

    joined_df = pd.read_csv('data_size_stats.csv')
    joined_df['parameters'] = joined_df['n_parameters']

    final_columns = ['layers', 'hidden_size_multiplier', 'parameters', 'max_used_mb', 'memory_fraction_low', 'memory_fraction_high']
    joined_df = joined_df[final_columns]
    # joined_df = joined_df[joined_df['hidden_size_multiplier'] == 1]
    print(joined_df)

    # save the joined dataframe to a csv file
    joined_df.to_csv('joined_df.csv', index=False)

    joined_df['max_used_mb'] = (joined_df['memory_fraction_low'] + joined_df['memory_fraction_high']) / 2 * 95830
    # nvidia smi not appropriate for this, does bug

    # plot parameters vs vram usanage
    plt.scatter(joined_df['parameters'], joined_df['max_used_mb'])
    plt.xlabel('Parameters')
    plt.ylabel('VRAM Usage (MB)')
    plt.title('Parameters vs VRAM Usage')
    plt.savefig('parameters_vs_vram_usage.png')

    # find relationship between parameters and vram usage
    # fit a linear regression model
    joined_df.dropna(inplace=True)
    X = joined_df['parameters'].values.reshape(-1, 1)
    y = joined_df['max_used_mb'].values
    model = LinearRegression()
    model.fit(X, y)
    print(model.coef_)
    print(model.intercept_)
    print(model.score(X, y))
    # extrapolate for 5 billion parameters
    print(model.predict([[5000000000]]))

    # plot the data and the model
    plt.scatter(X, y)
    plt.plot(X, model.predict(X), color='red')
    plt.xlabel('Parameters')
    plt.ylabel('VRAM Usage (MB)')
    plt.title('Parameters vs VRAM Usage')
    plt.savefig('parameters_vs_vram_usage_model.png')

    # compute n_parameters where intersect 95803
    # solve equation y = mx + b for x
    m = model.coef_[0]
    b = model.intercept_
    x = (95803 - b) / m
    print(x)





    # 544 GB