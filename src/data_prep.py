import os
import zipfile

def prepare_lol_dataset(out_dir='data'):
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, 'lol_dataset.zip')

    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Couldn't find {zip_path}. Please place 'lol_dataset.zip' inside the '{out_dir}' folder."
        )

    print('Extracting LOL dataset...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

    # Handle both possible extraction structures
    base1 = os.path.join(out_dir, 'LOLdataset')
    base2 = out_dir

    if os.path.exists(os.path.join(base1, 'our485')):
        base = base1
    elif os.path.exists(os.path.join(base2, 'our485')):
        base = base2
    else:
        raise RuntimeError("Could not find 'our485' folder after extraction.")

    train_low = os.path.join(base, 'our485/low')
    train_high = os.path.join(base, 'our485/high')
    val_low = os.path.join(base, 'eval15/low')
    val_high = os.path.join(base, 'eval15/high')

    # Create destination folders
    os.makedirs(os.path.join(out_dir, 'train/low'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'train/high'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'val/low'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'val/high'), exist_ok=True)

    def copy_all(src, dst):
        import shutil, glob
        for file in glob.glob(os.path.join(src, '*')):
            shutil.copy(file, dst)

    # Copy images
    copy_all(train_low, os.path.join(out_dir, 'train/low'))
    copy_all(train_high, os.path.join(out_dir, 'train/high'))
    copy_all(val_low, os.path.join(out_dir, 'val/low'))
    copy_all(val_high, os.path.join(out_dir, 'val/high'))

    print('✅ LOL dataset prepared in data/train and data/val.')

if __name__ == '__main__':
    prepare_lol_dataset()
