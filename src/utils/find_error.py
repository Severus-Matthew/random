# import os

# def find_files_with_error(folder_path, output_file="error_files.txt"):
#     matching_files = []

#     for root, dirs, files in os.walk(folder_path):
#         for filename in files:
#             filepath = os.path.join(root, filename)
#             try:
#                 with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
#                     if "ERROR" in f.read():
#                         matching_files.append(filepath)
#             except Exception as e:
#                 print(f"Could not read {filepath}: {e}")

#     with open(output_file, "w") as out:
#         out.write("\n".join(matching_files))

#     print(f"Found {len(matching_files)} file(s) containing 'ERROR'.")
#     print(f"Results saved to: {output_file}")

# # --- Usage ---
# folder_path = "/fsx/jmanvi/Internship_project/ASD/logs/full_study"  # Change this
# find_files_with_error(folder_path)
import pandas as pd
c=pd.read_csv('/fsx/jmanvi/Internship_project/ASD/results/staleness/staleness_cells.csv')
print(c[c.family=='qwen_interp'][['param','rel_weight_dist']].drop_duplicates().sort_values('param'))
# c=pd.read_csv('results/staleness/staleness_cells.csv')
s=c[(c.study=='S1_core_drift_curve')&(c.family=='qwen_interp')]
print(s.groupby(['method','param'])['acceptance_rate'].mean().round(3))