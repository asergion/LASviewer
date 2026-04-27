from src.las_indexer import index_las_file

metadata = index_las_file(
    las_path=r"G:\Shared drives\Mudlogging\LAS\Offshore\3-ET-2200-RN-TEMP.las",
    output_dir="data/indexed",
)

print("Parquet:", metadata["parquet_file"])
print("Curvas válidas:", metadata["valid_curves"])
