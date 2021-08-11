if [ ! -d "raw_data" ]; then
  echo "------------------------------"
  echo "downloading data"
  mkdir raw_data
  cd raw_data || exit
  wget https://s3.amazonaws.com/nist-srd/SD19/by_class.zip
  wget https://s3.amazonaws.com/nist-srd/SD19/by_write.zip
  unzip by_class.zip
  rm by_class.zip
  unzip by_write.zip
  rm by_write.zip
  cd ../
  echo "finished downloading data"
fi

if [ ! -d "intermediate" ]; then # stores .pkl files during preprocessing
  mkdir intermediate
fi

if [ ! -f ntermediate/class_file_dirs.pkl ]; then
  echo "------------------------------"
  echo "extracting file directories of images"
  python3 get_file_dirs.py
  echo "finished extracting file directories of images"
fi

if [ ! -f intermediate/class_file_hashes.pkl ]; then
  echo "------------------------------"
  echo "calculating image hashes"
  python3 get_hashes.py
  echo "finished calculating image hashes"
fi

if [ ! -f intermediate/write_with_class.pkl ]; then
  echo "------------------------------"
  echo "assigning class labels to write images"
  python3 match_hashes.py
  echo "finished assigning class labels to write images"
fi

if [ ! -f intermediate/images_by_writer.pkl ]; then
  echo "------------------------------"
  echo "grouping images by writer"
  python3 group_by_writer.py
  echo "finished grouping images by writer"
fi

if [ ! -f test/test.json ]; then
    echo "------------------------------"
    echo "converting data to tensors"
    python3 data_to_tensor.py
    echo "finished converting data to tensors"
fi
