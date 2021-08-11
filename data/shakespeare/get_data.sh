if [ ! -d "all_data" ] || [ ! "$(ls -A all_data)" ]; then
    if [ ! -d "raw_data" ]; then
        mkdir raw_data
    fi

    if [ ! -f raw_data/raw_data.txt ]; then
        echo "------------------------------"
        echo "retrieving raw data"
        cd raw_data || exit

        wget http://www.gutenberg.org/files/100/old/1994-01-100.zip
        unzip 1994-01-100.zip
        rm 1994-01-100.zip
        mv 100.txt raw_data.txt

        cd ../
    fi
fi

if [ ! -d "raw_data/by_play_and_character" ]; then
   echo "dividing txt data between users"
   python preprocess_shakespeare.py raw_data/raw_data.txt raw_data/
fi