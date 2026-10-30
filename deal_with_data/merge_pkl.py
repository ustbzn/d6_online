# #!/usr/bin/env python3
# """
# Script to merge two pickle files
# """

# import pickle
# import os
# from pathlib import Path

# def load_pkl(file_path):
#     """Load pickle file"""
#     print(f"Loading {file_path}...")
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     print(f"Loaded {file_path}")
#     print(f"  Type: {type(data)}")
#     if isinstance(data, (list, tuple)):
#         print(f"  Length: {len(data)}")
#     elif isinstance(data, dict):
#         print(f"  Keys: {list(data.keys())}")
#     return data

# def merge_pkl_files(file1, file2, output_file):
#     """
#     Merge two pickle files
    
#     Args:
#         file1: Path to first pickle file
#         file2: Path to second pickle file
#         output_file: Path to output merged pickle file
#     """
#     # Load both pickle files
#     data1 = load_pkl(file1)
#     data2 = load_pkl(file2)
    
#     # Merge based on data type
#     if isinstance(data1, list) and isinstance(data2, list):
#         print("\nMerging lists...")
#         merged_data = data1 + data2
#         print(f"Merged list length: {len(merged_data)}")
#     elif isinstance(data1, dict) and isinstance(data2, dict):
#         print("\nMerging dictionaries...")
#         merged_data = {}
#         # all_keys = set(data1.keys()) | set(data2.keys())
#         all_keys = ['infos']
        
        
#         for key in all_keys:
#             if key in data1 and key in data2:
#                 # Both dicts have this key, try to merge intelligently
#                 val1, val2 = data1[key], data2[key]
#                 if isinstance(val1, list) and isinstance(val2, list):
#                     print(f"  Merging '{key}' lists: {len(val1)} + {len(val2)} = {len(val1) + len(val2)}")
#                     merged_data[key] = val1 + val2
#                 elif isinstance(val1, dict) and isinstance(val2, dict):
#                     print(f"  Merging '{key}' dicts (second overwrites first for duplicate keys)")
#                     merged_data[key] = {**val1, **val2}
#                 else:
#                     print(f"  Warning: '{key}' exists in both but types don't match or not mergeable")
#                     print(f"    File1 type: {type(val1)}, File2 type: {type(val2)}")
#                     print(f"    Using value from File2")
#                     merged_data[key] = val2
#             elif key in data1:
#                 merged_data[key] = data1[key]
#             else:
#                 merged_data[key] = data2[key]
        
#         print(f"Merged dict keys: {list(merged_data.keys())}")
#     elif isinstance(data1, tuple) and isinstance(data2, tuple):
#         print("\nMerging tuples...")
#         merged_data = data1 + data2
#         print(f"Merged tuple length: {len(merged_data)}")
#     else:
#         print(f"\nWarning: Data types don't match or are not mergeable!")
#         print(f"  File1 type: {type(data1)}")
#         print(f"  File2 type: {type(data2)}")
#         print("Creating a list containing both datasets...")
#         merged_data = [data1, data2]
    
#     # Save merged data
#     print(f"\nSaving merged data to {output_file}...")
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
#     with open(output_file, 'wb') as f:
#         pickle.dump(merged_data, f)
#     print(f"Successfully saved merged data!")
#     print(f"Output file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

# if __name__ == "__main__":
#     # File paths
#     # file1 = "/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/train/d6_bj_3d_d6_S1001779_20251129_train_batch_10k_filter_invalid.pkl"
#     # file2 = "/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/train/d6_bj_3d_d6_S1001779_20251203_train_batch_6k.pkl"
    
#     # # Output file path (in the same directory as this script)
#     # output_file = "/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/train/d6_bj_3d_d6_S1001779_20251203_train_all_16k.pkl"
    
#     file1 = "/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/val/d6_bj_3d_d6_S1001779_20251129_val_batch_0.5k_filter_invalid.pkl"
#     file2 = "/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/val/d6_bj_3d_d6_S1001779_20251203_val_batch_0.7k.pkl"
    
#     # Output file path (in the same directory as this script)
#     output_file = "/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/val/d6_bj_3d_d6_S1001779_20251203_val_all_1k.pkl"
    
#     print("=" * 60)
#     print("PKL File Merger")
#     print("=" * 60)
#     print(f"File 1: {file1}")
#     print(f"File 2: {file2}")
#     print(f"Output: {output_file}")
#     print("=" * 60)
    
#     # Check if files exist
#     if not os.path.exists(file1):
#         print(f"Error: {file1} does not exist!")
#         exit(1)
#     if not os.path.exists(file2):
#         print(f"Error: {file2} does not exist!")
#         exit(1)
    
#     # Merge files
#     merge_pkl_files(file1, file2, output_file)
    
#     print("\n" + "=" * 60)
#     print("Merge completed successfully!")
#     print("=" * 60)

#!/usr/bin/env python3
"""
Script to merge multiple pickle files
"""

import pickle
import os
from pathlib import Path

def load_pkl(file_path):
    """Load pickle file"""
    print(f"Loading {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {file_path}")
    print(f"  Type: {type(data)}")
    if isinstance(data, (list, tuple)):
        print(f"  Length: {len(data)}")
    elif isinstance(data, dict):
        print(f"  Keys: {list(data.keys())}")
    return data

def merge_pkl_files(file_paths, output_file):
    """
    Merge multiple pickle files
    
    Args:
        file_paths: List of paths to pickle files
        output_file: Path to output merged pickle file
    """
    merged_data = None
    
    for file_path in file_paths:
        # Load pickle file
        # breakpoint()
        data = load_pkl(file_path)
        
        # Initialize merged_data with the first file's data
        if merged_data is None:
            merged_data = data
        else:
            # Merge based on data type
            if isinstance(merged_data, list) and isinstance(data, list):
                print("\nMerging lists...")
                merged_data += data  # Append data from the new file
                print(f"Merged list length: {len(merged_data)}")
            elif isinstance(merged_data, dict) and isinstance(data, dict):
                print("\nMerging dictionaries...")
                for key in data:
                    if key in merged_data:
                        val1, val2 = merged_data[key], data[key]
                        if isinstance(val1, list) and isinstance(val2, list):
                            print(f"  Merging '{key}' lists: {len(val1)} + {len(val2)} = {len(val1) + len(val2)}")
                            merged_data[key] += val2
                        elif isinstance(val1, dict) and isinstance(val2, dict):
                            print(f"  Merging '{key}' dicts (second overwrites first for duplicate keys)")
                            merged_data[key] = {**val1, **val2}
                        else:
                            print(f"  Warning: '{key}' exists in both but types don't match or not mergeable")
                            merged_data[key] = val2
                    else:
                        merged_data[key] = data[key]
                print(f"Merged dict keys: {list(merged_data.keys())}")
            else:
                print(f"\nWarning: Data types don't match or are not mergeable!")
                print(f"  Merged data type: {type(merged_data)}")
                print(f"  New data type: {type(data)}")
                print("Creating a list containing both datasets...")
                merged_data = [merged_data, data]
    
    # Save merged data
    print(f"\nSaving merged data to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(merged_data, f)
    print(f"Successfully saved merged data!")
    print(f"Output file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    # List of pickle file paths
    file_paths = [
        # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_3K_train_bj_3d_d6_s1001778_20251225_val_visibility.pkl',
        # # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_6K_train_hy_3d_d6_20123_20251225_train_visibility.pkl',
        # # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_bj_3d_d6_S1001779_20251203_train_all_16k_zn_visibility.pkl',
        # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_bj_3d_d6_s1001779_20251209_val_batch_3k_zn_visibility.pkl',
        # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_bj_3d_d6_s1001779_20251210_val_batch_1k_zn_visibility.pkl',
        # # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_hy_3d_d6_20123_20251210_train_batch_3k_zn_visibility.pkl',
        # # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/train_16k_hy_3d_d6_20123_20251229_train.pkl',
        # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/train_bj_3d_d6_s1001779_20251229_val_visibility.pkl',
        # # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/train_hy_3d_d6_20123_20260104_train_visibility.pkl',
        # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/train_bj_3d_d6_s1001779_20260114_val.pkl', 
        
        
        # # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_20frame_train_bj_3d_d6_s1001779_20251126_badcase_visibility.pkl',
        # # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/train_bj_3d_d6_s1001779_20260105_badcase_visibility.pkl'
        
        # # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_3K_train_bj_3d_d6_s1001778_20251225_val_visibility.pkl',
        # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_6K_train_hy_3d_d6_20123_20251225_val_visibility.pkl',
        # # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_bj_3d_d6_S1001779_20251203_val_all_16k_zn_visibility.pkl',
        # # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_bj_3d_d6_s1001779_20251209_val_batch_3k_zn_visibility.pkl',
        # # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_bj_3d_d6_s1001779_20251210_val_batch_1k_zn_visibility.pkl',
        # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_hy_3d_d6_20123_20251210_val_batch_3k_zn_visibility.pkl',
        # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/train_16k_hy_3d_d6_20123_20251229_val.pkl',
        # # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/train_bj_3d_d6_s1001779_20251229_val_visibility.pkl',
        # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/train_hy_3d_d6_20123_20260104_val_visibility.pkl',
        
        # # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_6K_train_hy_3d_d6_20123_20251225_val_visibility.pkl',
        # # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/d6_hy_3d_d6_20123_20251210_val_batch_3k_zn_visibility.pkl',
        # # Add more pickle files as needed
       
        # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/train_bj_3d_d6_20123_20260104_val_all_26k.pkl',
        # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/train_bj_hy_3d_d6_20123_20260104_val_all_66k_v2.pkl'
        
        # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/train_hy_3d_d6_20123_20260114_train.pkl',
        # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/train_hy_3d_d6_20123_20260114_val.pkl',
        # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/batch/train_hy_3d_d6_20123_20260116_13k_val.pkl',
        
        # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/train_hy_3d_d6_20123_20260104_val_all_40k_v2.pkl',
        # # '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/train_hy_3d_d6_20123_20260104_train_all_40k.pkl',
        
        '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/train_hy_3d_d6_20123_20260119_val_all_58k.pkl',
        '/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/train_bj_3d_d6_20123_20260119_val_all_33k.pkl',
    ]
    
    # Output file path (in the same directory as this script)
    output_file = "/pcpt/pcpt/project/atlb/model_online/multimodal_bevfusion/det_data_pkl/trainval_add_visibility/train_bj_hy_3d_d6_20123_20260119_val_all_90k.pkl"
    
    print("=" * 60)
    print("PKL File Merger")
    print("=" * 60)
    print(f"Input Files: {file_paths}")
    print(f"Output: {output_file}")
    print("=" * 60)
    
    # Check if all files exist
    for file in file_paths:
        if not os.path.exists(file):
            print(f"Error: {file} does not exist!")
            exit(1)
    
    # Merge files
    merge_pkl_files(file_paths, output_file)
    
    print("\n" + "=" * 60)
    print("Merge completed successfully!")
    print("=" * 60)
