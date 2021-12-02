import os
import shutil


def initialize_dirs(dirs: list) -> None:
    for init_dir in dirs:
        if not os.path.exists(init_dir):
            print(f"Mkdir {init_dir}")
            os.makedirs(init_dir)


def empty_dirs(to_empty: list) -> None:
    for folder in to_empty:
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
