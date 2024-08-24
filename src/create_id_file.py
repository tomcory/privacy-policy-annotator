import os
import sys

def create_id_file(orig_path: str):
    with open(os.path.join(os.path.dirname(os.path.dirname(orig_path)), 'id_file.txt'), 'w') as f:
        for file in os.listdir(orig_path):
            if file.endswith('.html'):
                f.write(os.path.splitext(file)[0] + '\n')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please provide the path to the directory containing the original files.')
        sys.exit(1)

    originals_directory_path = sys.argv[1]
    create_id_file(originals_directory_path)