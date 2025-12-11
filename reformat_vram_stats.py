import csv
import pandas as pd

def reformat_vram_stats(input_file, output_file):
    """
    Reformat VRAM stats CSV to organize by layers, hidden_size_multiplier, and max_used_mb
    """
    # Read the CSV file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse the data
    layers = lines[0].strip().split(',')
    hidden_size_multipliers = lines[1].strip().split(',')
    max_used_mb = lines[2].strip().split(',')
    
    # Create organized data
    organized_data = []
    
    for i in range(len(layers)):
        layer = layers[i]
        multiplier = hidden_size_multipliers[i]
        max_used = max_used_mb[i]
        
        # Only include entries where max_used_mb is not empty/NaN
        if max_used and max_used.strip() and max_used.strip() != '':
            organized_data.append({
                'layers': int(layer),
                'hidden_size_multiplier': int(multiplier),
                'max_used_mb': int(max_used)
            })
    
    # Sort by layers, then by hidden_size_multiplier
    organized_data.sort(key=lambda x: (x['layers'], x['hidden_size_multiplier']))
    
    # Write to new CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['layers', 'hidden_size_multiplier', 'max_used_mb'])
        writer.writeheader()
        writer.writerows(organized_data)
    
    # Also create a summary
    print("Reformatted VRAM Stats:")
    print("=" * 50)
    print(f"{'Layers':<8} {'Multiplier':<12} {'Max Used (MB)':<15}")
    print("-" * 50)
    
    for row in organized_data:
        print(f"{row['layers']:<8} {row['hidden_size_multiplier']:<12} {row['max_used_mb']:<15}")
    
    print(f"\nTotal entries: {len(organized_data)}")
    
    return organized_data

if __name__ == "__main__":
    input_file = "vram_stats copy.csv"
    output_file = "vram_stats_reformatted.csv"
    
    data = reformat_vram_stats(input_file, output_file)
    print(f"\nReformatted data saved to: {output_file}") 