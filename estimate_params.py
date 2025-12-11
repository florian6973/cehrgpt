import numpy as np

# def count(depth, width):
#     return np.power(width/768, 1.975) * (7088000*depth + 7190000)

def count(depth, width):
    return 12*(depth*width**2)+928*width+4417


# for (nh, hsm) in [(4, 2), (8, 2), (12, 2), (14, 1), (14, 2), (32, 4), (16, 2), (20, 2), (24, 2), (28, 2)]:
#     print(f"Running for {nh} layers and {hsm} hidden size multiplier")
#     print(f"Estimated parameters: {count(nh, hsm)/1e6}")


# for (nh, hsm) in [(28, 2), (36, 2), (44, 2), (52, 2), (60, 2), (68, 2), (76, 2), (84, 2), (92, 2), (100, 2)]:
#     print(f"Running for {nh} layers and {hsm} hidden size multiplier")

# find all integers that give a number of parameters +/- 10% of 100 million
for target_params in [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]:
    closest_per_width = {}
    # for width_multiplier in range(0, 20):
        # width = (2**width_multiplier) * 3
    for width_multiplier in range(1, 200):
        width = width_multiplier * 3 * 8
        for depth in range(1, 250):
            params = count(depth, width)
            if params > target_params * 0.9 and params < target_params * 1.1:
                # print(f"Depth: {depth}, Width: {width}, Parameters: {params/1e6}")
                if width not in closest_per_width:
                    closest_per_width[width] = (depth, params)
                else:
                    if abs(params - target_params) < abs(closest_per_width[width][1] - target_params):
                        closest_per_width[width] = (depth, params)

    for width, (depth, params) in closest_per_width.items():
        print(f"Target params: {target_params}, Width: {width}, Depth: {depth}, Parameters: {params/1e6}")