"""
Copy from https://github.com/TalwalkarLab/leaf/blob/master/data/femnist/preprocess/get_file_dirs.py

BSD 2-Clause License

Copyright (c) 2018, TalwalkarLab All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 following conditions are met:
    - Redistributions of source code must retain the above copyright notice, this list of conditions and the following
     disclaimer.
    - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
     the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Creates .pkl files for:
1. list of directories of every image in 'by_class'
2. list of directories of every image in 'by_write'
the hierarchal structure of the data is as follows:
- by_class -> classes -> folders containing images -> images
- by_write -> folders containing writers -> writer -> types of images -> images
the directories written into the files are of the form 'raw_data/...'
"""
import os
import pickle


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


class_files = []  # (class, file directory)
write_files = []  # (writer, file directory)

class_dir = os.path.join('raw_data', 'by_class')
classes = os.listdir(class_dir)
classes = [c for c in classes if len(c) == 2]

for cl in classes:
    cldir = os.path.join(class_dir, cl)
    rel_cldir = os.path.join(class_dir, cl)
    subcls = os.listdir(cldir)

    subcls = [s for s in subcls if (('hsf' in s) and ('mit' not in s))]

    for subcl in subcls:
        subcldir = os.path.join(cldir, subcl)
        rel_subcldir = os.path.join(rel_cldir, subcl)
        images = os.listdir(subcldir)
        image_dirs = [os.path.join(rel_subcldir, i) for i in images]

        for image_dir in image_dirs:
            class_files.append((cl, image_dir))


write_dir = os.path.join('raw_data', 'by_write')
write_parts = os.listdir(write_dir)

for write_part in write_parts:
    writers_dir = os.path.join(write_dir, write_part)
    rel_writers_dir = os.path.join(write_dir, write_part)
    writers = os.listdir(writers_dir)

    for writer in writers:
        writer_dir = os.path.join(writers_dir, writer)
        rel_writer_dir = os.path.join(rel_writers_dir, writer)
        wtypes = os.listdir(writer_dir)

        for wtype in wtypes:
            type_dir = os.path.join(writer_dir, wtype)
            rel_type_dir = os.path.join(rel_writer_dir, wtype)
            images = os.listdir(type_dir)
            image_dirs = [os.path.join(rel_type_dir, i) for i in images]

            for image_dir in image_dirs:
                write_files.append((writer, image_dir))

save_obj(
    class_files,
    os.path.join('intermediate', 'class_file_dirs'))
save_obj(
    write_files,
    os.path.join('intermediate', 'write_file_dirs'))
