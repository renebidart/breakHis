import os
import sys
import glob
from shutil import copyfile

def main(data_loc):
    all_files = glob.glob(data_loc+'/**/*.png', recursive=True)
    all_files = [loc for loc in all_files if loc.split('.', 1)[-1] == 'png']

    for file in all_files:
        # fold = file.rsplit('/', 1)[0].rsplit('/', 1)[0].rsplit('/', 1)[0].rsplit('/', 1)[0].rsplit('/', 1)[1]
        base_loc = file.rsplit('/', 1)[0].rsplit('/', 1)[0].rsplit('/', 1)[0].rsplit('/', 1)[0]
        base_loc = base_loc.replace('mkfold_keras_8', 'mkfold_keras_2')
        new_class = file.rsplit('/', 1)[1].split('_', 1)[1].split('-', 1)[0][0]
        magnification = file.rsplit('/', 1)[0].rsplit('/', 1)[0].rsplit('/', 1)[0].rsplit('/', 1)[1]
        ttv=file.rsplit('/', 1)[0].rsplit('/', 1)[0].rsplit('/', 1)[1]
        new_loc = os.path.join(str(base_loc), str(magnification), ttv, new_class)

        if not os.path.exists(new_loc):
            os.makedirs(new_loc)

        name = file.rsplit('/', 1)[1]
        copyfile(file, os.path.join(new_loc, name))

if __name__ == "__main__":
    data_loc = sys.argv[1]
    main(data_loc)