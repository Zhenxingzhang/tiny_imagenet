"""

Download Tiny ImageNet data (http://cs231n.stanford.edu/project.html) if needed

"""
from __future__ import print_function
from six.moves import urllib
from tensorflow.python.platform import gfile
import tempfile
import os
import zipfile
import src.common.paths as paths

if __name__ == "__main__":
    file_dir = os.path.abspath(paths.DATA_PATH)
    with tempfile.NamedTemporaryFile() as tmpfile:
        temp_file_name = tmpfile.name
        print("Downloading tiny imagenet dataset (237 MB) ")
        a = urllib.request.urlretrieve("http://cs231n.stanford.edu/tiny-imagenet-200.zip", temp_file_name)

        file_path = os.path.join(file_dir, "tiny_imagenet-200.zip")
        gfile.Copy(temp_file_name, file_path)
        print("Extracting zip archive (this may take a while)")
        with zipfile.ZipFile(file_path, 'r') as f:
            f.extractall(file_dir)
