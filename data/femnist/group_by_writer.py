"""
Copy from https://github.com/TalwalkarLab/leaf/blob/master/data/femnist/preprocess/group_by_writer.py

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

"""
import os
import pickle


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

wwcd = os.path.join('intermediate', 'write_with_class')
write_class = load_obj(wwcd)

writers = []  # each entry is a (writer, [list of (file, class)]) tuple
cimages = []
(cw, _, _) = write_class[0]
for (w, f, c) in write_class:
    if w != cw:
        writers.append((cw, cimages))
        cw = w
        cimages = [(f, c)]
    cimages.append((f, c))
writers.append((cw, cimages))

ibwd = os.path.join('intermediate', 'images_by_writer')
save_obj(writers, ibwd)
