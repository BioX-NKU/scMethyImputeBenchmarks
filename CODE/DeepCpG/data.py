#Take 2i as an example
import os
import subprocess
import glob

def main():
    data_root_dir = '2i/'
    output_base_dir = '2i/'
    pattern = '*.cov.txt.gz'
    
    dcpg_script = 'anaconda3/envs/deepcpg/bin/dcpg_data.py'
    python_executable = 'anaconda3/envs/deepcpg/bin/python'

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'

    chromosomes = [str(i) for i in range(1, 20)]

    subdirs = [d for d in os.listdir(data_root_dir) if os.path.isdir(os.path.join(data_root_dir, d))]

    for batch_name in subdirs:
        batch_dir = os.path.join(data_root_dir, batch_name)
        print(f"Processing batch: {batch_name}...")

        file_list = sorted(glob.glob(os.path.join(batch_dir, pattern)))

        if not file_list:
            print(f"No files found in {batch_dir}. Skipping batch.")
            continue

        batch_output_dir = os.path.join(output_base_dir, batch_name)
        os.makedirs(batch_output_dir, exist_ok=True)

        batch_files = [f"{file}" for file in file_list]

        cmd = [
            python_executable,
            dcpg_script,
            '--cpg_profiles'
        ]
        cmd.extend(batch_files)
        cmd.extend([
            '--dna_files', 'genome/mm9/',
            '--out_dir', batch_output_dir,
            '--dna_wlen', '1001',
            '--cpg_wlen', '50',
            '--chromos'
        ])
        cmd.extend(chromosomes)

        print(f"Processing batch {batch_name}, containing {len(batch_files)} files.")

        try:
            result = subprocess.run(cmd, env=env, check=True)
            print(f"Successfully processed batch {batch_name}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to process batch {batch_name}, error: {e}")

    print(f"Processing completed. All output files are stored in '{output_base_dir}'.")

if __name__ == '__main__':
    main()