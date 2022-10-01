import os

print('File Utils 1.0.0')



def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def list_dirs(dir_name):
    return sorted(os.listdir(dir_name))

def list_files(dir_name, filter=None):
    """
      Get all files in directory "dir_name"
    """
    file_list = os.listdir(dir_name)
    files = []
    for f in file_list:
        abs_path = os.path.join(dir_name, f)
        if filter is not None and filter not in f:
            continue
        if os.path.isdir(abs_path):
            files = files + list_files(abs_path)[0]
        else:
            files.append(abs_path)
    return sorted(files), len(files)


def get_file_name(img_path, with_type=False):
    name = os.path.basename(img_path)

    if not with_type:
        name = name.split(".")[0]
    return name