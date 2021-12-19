mkdir data
cd data || exit
if ! [[ -d "LJSpeech-1.1" ]]; then
    if ! [[ -f "LJSpeech-1.1.tar.bz2" ]]; then
        wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    fi
    echo "Unpacking archive."
    tar -xjf LJSpeech-1.1.tar.bz2
fi