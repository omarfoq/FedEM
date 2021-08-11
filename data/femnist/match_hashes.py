"""
Copy from https://github.com/TalwalkarLab/leaf/blob/master/data/femnist/preprocess/match_hashes.py

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


cfhd = os.path.join('intermediate', 'class_file_hashes')
wfhd = os.path.join('intermediate', 'write_file_hashes')
class_file_hashes = load_obj(cfhd)  # each elem is (class, file dir, hash)
write_file_hashes = load_obj(wfhd)  # each elem is (writer, file dir, hash)

class_hash_dict = {}
for i in range(len(class_file_hashes)):
    (c, f, h) = class_file_hashes[len(class_file_hashes)-i-1]
    class_hash_dict[h] = (c, f)

write_classes = []
for tup in write_file_hashes:
    (w, f, h) = tup
    write_classes.append((w, f, class_hash_dict[h][0]))

wwcd = os.path.join('intermediate', 'write_with_class')
save_obj(write_classes, wwcd)
