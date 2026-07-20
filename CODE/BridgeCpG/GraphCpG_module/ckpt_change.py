import torch

def process_masks(input_path, output_path):
    masks = torch.load(input_path)
    converted_tensors = []
    for mask_idx, mask in enumerate(masks):
        converted_mask = mask.clone()
        converted_mask[:, 0] = 1024 * mask_idx + converted_mask[:, 0]
        converted_tensors.append(converted_mask)
    merged_mask = torch.cat(converted_tensors, dim=0)
    torch.save(merged_mask, output_path)

if __name__ == "__main__":
    input_file = "CpGTransformer.ckpt"
    output_file = "GraphCpG.ckpt"
    process_masks(input_file, output_file)