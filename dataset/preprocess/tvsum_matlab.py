import os
import h5py
import pandas as pd

_data_root = os.path.join(os.getenv('DATA_ROOT', './data'), 'tvsum')
mat_path = os.path.join(_data_root, 'matlab/ydata-tvsum50.mat')
tsv_path = os.path.join(_data_root, 'data/ydata-tvsum50-info.tsv')
output_csv = os.path.join(_data_root, 'tvsum_metadata.csv')

def ref_to_str(dataset, ref):
    chars = dataset[ref[0]][()]
    return ''.join(map(chr, chars.flatten().astype(int)))

def ref_to_scalar(dataset, ref):
    return int(dataset[ref[0]][()][0][0])

with h5py.File(mat_path, 'r') as f:
    video_refs = f['tvsum50/video'][:]
    category_refs = f['tvsum50/category'][:]
    title_refs = f['tvsum50/title'][:]
    nframe_refs = f['tvsum50/nframes'][:]

    video_ids = [ref_to_str(f, ref) for ref in video_refs]
    categories = [ref_to_str(f, ref) for ref in category_refs]
    titles = [ref_to_str(f, ref) for ref in title_refs]
    nframes = [ref_to_scalar(f, ref) for ref in nframe_refs]

df_mat = pd.DataFrame({
    "video_id": video_ids,
    "category": categories,
    "title": titles,
    "nframes": nframes
})

df_tsv = pd.read_csv(tsv_path, sep="\t", header=None)
df_tsv.columns = ["category", "video_id", "title", "url", "length"]

df = pd.merge(df_mat, df_tsv[["video_id", "url", "length"]], on="video_id", how="left")
df.to_csv(output_csv, index=False)
print(f"Metadata saved to: {output_csv}")
