# Overview #

This is a repository containing some character-wise n-gram calculating
code I wrote because all the stuff I found was either doing word-level
n-grams or couldn't be used for my purposes because it looked at
character frequencies within words rather than within running text.

I have copied it from a private project since it will be useful to
others that need to calculate character n-grams.

# Setup

Standard Python setup (I'm using python3.11 - but I'm not using many odd things except for some libraries, so the code should last awhile.):
```bash
python3 -m venv venv
source venv/bin/activate # on Linux  
#  .\venv\Scripts\activate.ps1  # on Windows
pip install -r requirements.txt
pip install -e .
pre-commit install # Set up portable git hooks
```

You should set up black integration with your editor. (The project installs `blackd` by default)

# Preparing data

For my optimization, I use the common crawl WET files. There is much more data than I need, so
I just used two files from
[the March 2018 Common crawl](https://commoncrawl.org/2018/03/march-2018-crawl-archive-now-available/).
* https://data.commoncrawl.org/crawl-data/CC-MAIN-2018-13/segments/1521257644271.19/wet/CC-MAIN-20180317035630-20180317055630-00000.warc.wet.gz
* https://data.commoncrawl.org/crawl-data/CC-MAIN-2018-13/segments/1521257644271.19/wet/CC-MAIN-20180317035630-20180317055630-00001.warc.wet.gz

Once you have downloaded and unzipped them, you need to split into language chunks. If your data is in
`D:\CommonCrawl\2018-03\`, you can run the following command to split the files into language chunks. It expects the
WET files to end in ".wet". It will append to files named language_XX.txt, where XX is the two letter language code.
There will also be two special files. `language_uncertain.txt` contains text where the language probability was 0.9
or less. And `language_no-languages-detected.txt` contains sections where `langdetect` could not detect a language.
It looks like this is mostly because they contain languages that `langdetect` doesn't know about.
```powershell
python .\src\main.py --split-wet-dir D:\CommonCrawl\2018-03\
```

# Character n-grams
Once you have some language files, you can generate n-grams from them with the `--create-ngrams` option. It expects
sections separated by 4 consecutive newlines (which is what the `--split-wet-dir` option produces). It will use
the `--input-file` as the source of the text, and will write JSON n-grams to the `--output-file`. It reads all the text
into memory, so it will not work on very large files. It will also not work on files that are not UTF-8 encoded.

You can replace rare characters with a special symbol using the `--rare-character-threshold` option.

One way to choose that threshold is with `rarest_char.py`. You can choose a class of characters to consider.
In my sample, the rarest character that is on a US keyboard is "`". However, that is rare enough that many
Chinese characters are more common in the English sample. A more conservative approach would be to use the
rarest American alphanumeric character. That is much more common. I used it to create my bigrams-no-rare.json

## Example producing bigrams from the English language file:
```powershell
python .\src\main.py --create-ngrams 2 --input-file D:\CommonCrawl\2018-03\language_en.txt --output-file bigrams.json
```

# Typed character n-grams

The `--type-characters` option pre-processes the input for an imaginary keyboard layout with
sticky modifier keys. This is only useful for my particular application. However, I leave it
here because others could modify it for similar purposes.


# Contribution guidelines

I'm not looking for contributions. This repo is just here to give back to the community. You are welcome to fork it,
but I will not have much time to respond to issues or pull requests.

# Who do I talk to?

compu.moyer@gmail.com
