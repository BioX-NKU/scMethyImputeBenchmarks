import numpy as np
from argparse import ArgumentParser
import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import time

# Record start time
start_time = time.perf_counter()

def boolean(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter):
    pass

parser = ArgumentParser(description='Training script for CpG Transformer (supports NPZ format, calculates metrics per column)',
                        formatter_class=CustomFormatter)
parser.add_argument('X', type=str, metavar='X', help='NumPy file (.npy or .npz) containing encoded genomic data.')
parser.add_argument('y', type=str, metavar='y', help='NumPy file (.npy or .npz) containing methylation matrix (shape: [sites, columns/cells]).')
parser.add_argument('pos', type=str, metavar='pos', help='NumPy file (.npy or .npz) containing CpG site position information.')

# DataModule parameters
dm_parse = parser.add_argument_group('DataModule', 'Data module parameters (ensure y matrix is [sites, columns])')
dm_parse.add_argument('--segment_size', type=int, default=1024,
                      help='Number of CpG sites per batch (row dimension), reduce if GPU memory is insufficient.')
dm_parse.add_argument('--fracs', type=float, nargs='+', default=[1,0,0],
                      help='Chromosome ratios for train/validation/test sets (excluding chromosomes specified by --val_keys/--test_keys).')
dm_parse.add_argument('--mask_p', type=float, default=0.25,
                      help='Proportion of masked sites per batch (calculated based on column dimension).')
dm_parse.add_argument('--mask_random_p', type=float, default=0.20,
                      help='Proportion of randomly replaced masked sites.')
dm_parse.add_argument('--resample_cells', type=int, default=None,
                      help='Number of columns (cells) resampled per batch during training, set if GPU memory is insufficient (must ≤ total columns).')
dm_parse.add_argument('--resample_cells_val', type=int, default=None, 
                      help='Number of columns (cells) resampled per batch during validation, should follow the same logic as training.')
dm_parse.add_argument('--val_keys', type=str, nargs='+', default=['chr10'],
                      help='Chromosome names for validation set.')
dm_parse.add_argument('--test_keys', type=str, nargs='+', default=['chr5'], 
                      help='Chromosome names for test set.')
dm_parse.add_argument('--batch_size', type=int, default=1,
                      help='Batch size (divided by chromosome segments, ensure single batch column count does not exceed GPU memory).')
dm_parse.add_argument('--n_workers', type=int, default=4,
                      help='Number of threads for data loading, increase if CPU is the bottleneck.')

# Model parameters
model_parse = parser.add_argument_group('Model', 'CpG Transformer hyperparameters (n_cells corresponds to column count)')
model_parse.add_argument('--transfer_checkpoint', type=str, default=None,
                         help='Pretrained weight file (.ckpt or .pt). .ckpt ignores subsequent model parameters (except --lr); .pt requires matching model structure (column count can be adjusted).')
model_parse.add_argument('--RF', type=int, default=1001,
                         help='CNN receptive field size (must be odd, matching genomic segment length).')
model_parse.add_argument('--n_conv_layers', type=int, default=2,
                         help='Number of CNN layers (only 2 or 3 supported).')
model_parse.add_argument('--DNA_embed_size', type=int, default=32,
                         help='Genomic embedding dimension from CNN output.')
model_parse.add_argument('--cell_embed_size', type=int, default=32,
                         help='Column (cell) embedding dimension.')
model_parse.add_argument('--CpG_embed_size', type=int, default=32,
                         help='CpG site embedding dimension.')
model_parse.add_argument('--n_transformers', type=int, default=4,
                         help='Number of Transformer modules.')
model_parse.add_argument('--act', type=str, default='relu',
                         help='Activation function for Transformer feed-forward layer (relu/gelu).')
model_parse.add_argument('--mode', type=str, choices=['2D', 'axial', 'intercell', 'intracell', 'none'], default='axial',
                         help='Attention mode (intercell/axial more suitable for column dimension correlation modeling).')
model_parse.add_argument('--transf_hsz', type=int, default=64,
                         help='Transformer hidden layer dimension.')
model_parse.add_argument('--n_heads', type=int, default=8,
                         help='Number of self-attention heads (must satisfy transf_hsz = n_heads * head_dim).')
model_parse.add_argument('--head_dim', type=int, default=8,
                         help='Dimension of each attention head.')
model_parse.add_argument('--window', type=int, default=21,
                         help='Sliding window size for row dimension (must be odd, balancing local and global information).')
model_parse.add_argument('--layernorm', type=boolean, default=True,
                         help='Whether to use LayerNorm in Transformer.')
model_parse.add_argument('--CNN_do', type=float, default=.0,
                         help='Dropout probability for CNN.')
model_parse.add_argument('--transf_do', type=float, default=.2,
                         help='Dropout probability for Transformer attention matrix.')
model_parse.add_argument('--lr', type=float, default=5e-4,
                         help='Initial learning rate (suggested to reduce to 1e-5~1e-4 for transfer learning).')
model_parse.add_argument('--lr_decay_factor', type=float, default=.90,
                         help='Learning rate decay factor per epoch.')
model_parse.add_argument('--warmup_steps', type=int, default=1000,
                         help='Number of steps for linear learning rate warmup.')

# Logging and results parameters
log_parse = parser.add_argument_group('Logging', 'Logging and result saving parameters')
log_parse.add_argument('--tensorboard', type=boolean, default=True,
                       help='Whether to use TensorBoard to record training process (including per-column metrics).')
log_parse.add_argument('--log_folder', type=str, default='logfolder',
                       help='Directory for TensorBoard logs and model checkpoints.')
log_parse.add_argument('--experiment_name', type=str, default='experiment',
                       help='Experiment name (used to distinguish different training tasks).')
log_parse.add_argument('--earlystop', type=boolean, default=True,
                       help='Whether to use early stopping based on average validation loss (to avoid overfitting).')
log_parse.add_argument('--patience', type=int, default=10,
                       help='Number of patience epochs for early stopping (stop if validation loss does not decrease for N consecutive epochs).')
log_parse.add_argument('--checkpoint_name', type=str, default='cpgtransformer_col_best.ckpt',
                       help='Filename for the best model checkpoint (suffix must be .ckpt).')
log_parse.add_argument('--csv_dir', type=str, default='./validation_results_col_hui',
                       help='Directory to save validation/test metrics calculated per column (automatically created).')
log_parse.add_argument('--save_test_col', type=boolean, default=True,
                       help='Whether to save metrics calculated per column during testing (recommended to enable).')

# Add default Trainer arguments
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()

# -------------------------- Data loading and dimension validation (supports NPZ format) --------------------------
def load_numpy_file(file_path):
    """Load .npy or .npz file, return dictionary-like object"""
    if file_path.endswith('.npz'):
        return np.load(file_path, allow_pickle=True)  # Returns NpzFile object (dictionary-like)
    elif file_path.endswith('.npy'):
        data = np.load(file_path, allow_pickle=True)
        # If .npy file contains a dictionary, return directly; otherwise wrap as dictionary
        if isinstance(data, dict):
            return data
        else:
            # Assume single array corresponds to "whole_genome" key
            return {'whole_genome': data}
    else:
        raise ValueError(f"Unsupported file format: {file_path}, only .npy and .npz are supported")

# Load data (supports .npy and .npz formats)
print(f"Loading data: X={args.X}, y={args.y}, pos={args.pos}")
X = load_numpy_file(args.X)
y = load_numpy_file(args.y)
pos = load_numpy_file(args.pos)

# Validate y format and dimensions (supports NpzFile and dictionary)
valid_types = (dict, np.lib.npyio.NpzFile)
if not isinstance(y, valid_types):
    raise ValueError(f"y file must be dictionary or NPZ format (key=chromosome), current type is {type(y)}")

# Get example chromosome and validate dimensions
chr_example = next(iter(y.keys()))  # Get first chromosome key
y_array = y[chr_example]  # Get array from NPZ or dictionary

# Validate array dimensions
if len(y_array.shape) != 2:
    raise ValueError(f"Methylation matrix for each chromosome must be 2D [number of sites, number of columns], current shape of {chr_example} is {y_array.shape}")
n_cells = y_array.shape[1]  # Number of columns = number of cells (core: calculate column-wise metrics later)
print(f"Data validation passed: chromosome {chr_example} shape={y_array.shape}, total columns (cells)={n_cells}")

# Determine data mode (binary/continuous)
if np.all(np.mod(y_array, 1) == 0):
    data_mode = 'binary'
else:
    data_mode = 'continuous'
print(f"Data mode: {data_mode} (binary=discrete labels, continuous=continuous values)")

# -------------------------- Import model and data module --------------------------
from cpg_transformer.cpgtransformer import CpGTransformer  # Model with column-wise metrics calculation
from cpg_transformer.datamodules import CpGTransformerDataModule  # Data module

# -------------------------- Model initialization --------------------------
print(f"\nInitializing model: total columns={n_cells}, data mode={data_mode}")
if args.transfer_checkpoint:
    assert args.transfer_checkpoint.endswith('.ckpt') or args.transfer_checkpoint.endswith('.pt'), \
        "Pretrained files only support .ckpt (complete model) or .pt (weight dictionary)"
    
    if args.transfer_checkpoint.endswith('.ckpt'):
        # Load .ckpt file
        print(f"Loading pretrained .ckpt model: {args.transfer_checkpoint}")
        model = CpGTransformer.load_from_checkpoint(
            args.transfer_checkpoint,
            lr=args.lr,
            n_cells=n_cells,
            data_mode=data_mode,
            epochmax=args.max_epochs,
            y_filename=args.y,
            csv_dir=args.csv_dir
        )
        # Update column embedding layer
        model.cell_embed = torch.nn.Embedding(n_cells, model.hparams.cell_embed_size)
        model.hparams.n_cells = n_cells
        print(f".ckpt model loaded: column embedding layer dimension updated to [{n_cells}, {model.hparams.cell_embed_size}]")
    
    else:
        # Load .pt weight file
        print(f"Loading .pt weight file: {args.transfer_checkpoint}")
        pretrained_state = torch.load(args.transfer_checkpoint, map_location='cpu')
        
        # Create model
        model = CpGTransformer(
            n_cells=n_cells,
            RF=args.RF,
            n_conv_layers=args.n_conv_layers,
            CNN_do=args.CNN_do,
            DNA_embed_size=args.DNA_embed_size,
            cell_embed_size=args.cell_embed_size,
            CpG_embed_size=args.CpG_embed_size,
            transf_hsz=args.transf_hsz,
            transf_do=args.transf_do,
            act=args.act,
            n_transformers=args.n_transformers,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            window=args.window,
            layernorm=args.layernorm,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            warmup_steps=args.warmup_steps,
            mode=args.mode,
            data_mode=data_mode,
            epochmax=args.max_epochs,
            y_filename=args.y,
            csv_dir=args.csv_dir
        )
        # Load pretrained weights
        model.load_state_dict(pretrained_state, strict=False)
        print(f".pt weights loaded: only keep weights with matching structure, column embedding layer reinitialized")

else:
    # Create new model
    model = CpGTransformer(
        n_cells=n_cells,
        RF=args.RF,
        n_conv_layers=args.n_conv_layers,
        CNN_do=args.CNN_do,
        DNA_embed_size=args.DNA_embed_size,
        cell_embed_size=args.cell_embed_size,
        CpG_embed_size=args.CpG_embed_size,
        transf_hsz=args.transf_hsz,
        transf_do=args.transf_do,
        act=args.act,
        n_transformers=args.n_transformers,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        window=args.window,
        layernorm=args.layernorm,
        lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        warmup_steps=args.warmup_steps,
        mode=args.mode,
        data_mode=data_mode,
        epochmax=args.max_epochs,
        y_filename=args.y,
        csv_dir=args.csv_dir
    )

# Print key model parameters
print(f"\nModel configuration summary:")
print(f"- Total columns (cells): {model.hparams.n_cells}")
print(f"- Data mode: {model.hparams.data_mode}")
print(f"- Transformer mode: {model.hparams.mode}")
print(f"- Column-wise metrics save directory: {model.csv_dir}")
print(f"- Total training epochs: {model.hparams.epochmax}")

# -------------------------- Data module initialization --------------------------
datamodule = CpGTransformerDataModule(
    X=X,
    y=y,
    pos=pos,
    segment_size=args.segment_size,
    fracs=args.fracs,
    RF=model.RF,
    mask_perc=args.mask_p,
    mask_random_perc=args.mask_random_p,
    resample_cells=args.resample_cells,
    resample_cells_val=args.resample_cells_val,
    val_keys=args.val_keys,
    test_keys=args.test_keys,
    batch_size=args.batch_size,
    n_workers=args.n_workers
)

# Preload validation set samples to validate output shape
print(f"\nValidating data module output shapes...")
datamodule.setup('validate')
val_dataloader = datamodule.val_dataloader()
val_batch = next(iter(val_dataloader))
x_batch, y_orig_batch, y_masked_batch, pos_batch, ind_train_batch, cell_indices_batch = val_batch
print(f"Validation set batch shapes:")
print(f"- Genomic input x: {x_batch.shape}")
print(f"- Methylation matrix y_orig: {y_orig_batch.shape} (should be [batch, sites, columns])")
print(f"- Masked y_masked: {y_masked_batch.shape} (should be [batch, sites, columns])")
if y_orig_batch.shape[-1] != n_cells and args.resample_cells_val is None:
    raise ValueError(f"Number of columns in y_orig ({y_orig_batch.shape[-1]}) does not match total columns ({n_cells}), please check data module")

# -------------------------- Callback configuration --------------------------
callbacks = []

# Model checkpoint saving
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_last=True,
    filename=args.checkpoint_name.replace('.ckpt', '_epoch{epoch:02d}_valLoss{val_loss:.4f}'),
    dirpath=os.path.join(args.log_folder, args.experiment_name),
    verbose=True
)
callbacks.append(checkpoint_callback)

# Learning rate monitoring
if args.tensorboard:
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

# Early stopping
if args.earlystop:
    earlystopper = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=args.patience,
        verbose=True,
        check_finite=True
    )
    callbacks.append(earlystopper)

# Create result directories in advance
os.makedirs(args.csv_dir, exist_ok=True)
os.makedirs(os.path.join(args.log_folder, args.experiment_name), exist_ok=True)
print(f"\nResult directories created:")
print(f"- Model checkpoint directory: {os.path.join(args.log_folder, args.experiment_name)}")
print(f"- Column-wise metrics save directory: {args.csv_dir}")

# -------------------------- TensorBoard logging configuration --------------------------
if args.tensorboard:
    logger = TensorBoardLogger(
        save_dir=args.log_folder,
        name=args.experiment_name,
        default_hp_metric=False,
        log_graph=True
    )
    print(f"TensorBoard logs: {logger.log_dir} (start command: tensorboard --logdir {args.log_folder})")
else:
    logger = None

# -------------------------- Trainer initialization --------------------------
print(f"\nInitializing Trainer: max_epochs={args.max_epochs}, device={args.gpus if hasattr(args, 'gpus') else 'CPU'}")
trainer = Trainer.from_argparse_args(
    args,
    logger=logger,
    callbacks=callbacks,
    check_val_every_n_epoch=1,
    num_sanity_val_steps=2,
    deterministic=True,
    benchmark=False
)

# -------------------------- Start training --------------------------
print(f"\n=== Starting training (column-wise metrics calculation) ===")
print(f"Training epochs: {args.max_epochs}, early stopping patience: {args.patience} epochs")
print(f"Best model will be saved as: {checkpoint_callback.dirpath}/{args.checkpoint_name}")
trainer.fit(model, datamodule=datamodule)

# -------------------------- Post-training processing --------------------------
# Calculate training time
end_time = time.perf_counter()
execution_time = end_time - start_time
time_log_path = os.path.join(args.csv_dir, 'training_time.log')
with open(time_log_path, 'w', encoding='utf-8') as f:
    f.write(f"=== Training Time Log ===\n")
    f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
    f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
    f.write(f"Total time: {execution_time:.2f} seconds ({execution_time/3600:.2f} hours)\n")
    f.write(f"Total training epochs: {trainer.current_epoch}\n")
    f.write(f"Data mode: {data_mode}\n")
    f.write(f"Total columns (cells): {n_cells}\n")
    f.write(f"Best validation loss: {checkpoint_callback.best_model_score.item():.4f}\n")
print(f"\nTraining time logged to: {time_log_path}")

# Save best model
best_ckpt_path = checkpoint_callback.best_model_path
if best_ckpt_path:
    final_best_ckpt = os.path.join(args.csv_dir, args.checkpoint_name)
    torch.save(torch.load(best_ckpt_path), final_best_ckpt)
    print(f"Best model copied to: {final_best_ckpt}")
else:
    print("Warning: No best model checkpoint found (training may not have completed one epoch)")

# -------------------------- Testing phase --------------------------
if args.save_test_col and trainer.current_epoch > 0:
    print(f"\n=== Starting testing (column-wise metrics calculation) ===")
    test_model = CpGTransformer.load_from_checkpoint(
        final_best_ckpt if os.path.exists(final_best_ckpt) else best_ckpt_path,
        csv_dir=args.csv_dir,
        save_test_col=args.save_test_col
    )
    test_results = trainer.test(test_model, datamodule=datamodule, ckpt_path=None, verbose=True)
    test_col_csv = os.path.join(args.csv_dir, f"test_column_metrics_{os.path.basename(args.y)}.csv")
    print(f"Test set column-wise metrics saved to: {test_col_csv}")
else:
    print(f"\nSkipping testing phase (--save_test_col={args.save_test_col} or training not completed)")

print(f"\n=== Training script execution completed ===")