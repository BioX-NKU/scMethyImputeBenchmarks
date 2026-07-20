#Take MLP as an example

# Define the required folder path
dirs=(
    "GSE87197/MLP"
    "GSE87197/MLP"
    "GSE87197/MLPout"
)

# Create folders
for dir in "${dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created directory: $dir"
    fi
done

# Get all files in the directory whose names contain MLP
count=0
all_files=(tsv/*MLP*.txt.gz)

for cell in "${all_files[@]}"
do
    count=$((count+1))
    python data/EncodeFromTsv.py "$cell" X_hg38.npz GSE87197/MLP/y_"$count".npz GSE87197/MLP/pos_"$count".npz --prepend_chr --chroms 10 11 12 13 14 15 16 17 18 19 20 21 22 1 2 3 4 5 6 7 8 9 --continuous
done

# Merged processed y and pos files
CUDA_VISIBLE_DEVICES=0 python data/CombineEncodedLabels.py --y_files GSE87197/MLP/y_* --pos_files GSE87197/MLP/pos_* --y_outFile GSE87197/MLP/y_MLP.npz --pos_outFile GSE87197/MLP/pos_MLP.npz

# Train CpG Transformer
CUDA_VISIBLE_DEVICES=0 python train_cpg_transformer.py X_hg38.npz GSE87197/MLP/y_MLP.npz GSE87197/MLP/pos_MLP.npz --gpus 1 --checkpoint_name GSE87197/MLP/cpgtransformer.ckpt

# Impute
CUDA_VISIBLE_DEVICES=0 python impute_genome.py cpg_transformer X_hg38.npz GSE87197/MLP/y_MLP.npz GSE87197/MLP/pos_MLP.npz GSE87197/MLPout/output_MLP.npz --model_checkpoint GSE87197/MLP/cpgtransformer.ckpt