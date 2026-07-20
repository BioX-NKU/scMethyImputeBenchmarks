#Take HCC as an example
# From 1 File to All Files for Dataset HCC
for i in $(seq 1 27)
do
    CUDA_VISIBLE_DEVICES=0 python model/trainmodel.py X_hg38.npz HCC/y_batch_$i.npz HCC/pos_batch_$i.npz --max_epochs 10 --csv_dir result/HCC
done

#For detailed operation instructions, refer to the readme of MambaCpG.