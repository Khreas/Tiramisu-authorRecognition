# Crawler

## Overview

The crawler is split between three files, `crawl_gutenberg.py`, `clean_text.py`, `findtext.py`:

- `crawl_gutenberg.py` contains the functions [`crawler()`](crawler.md#crawl_gutenbergpy) and [`compressFiles()`](crawler.md#crawl_gutenbergpy). 

- `clean_text.py` contains the functions [`concatenateFiles()`](crawler.md#clean_textpy) and [`cleanText()`](crawler.md#clean_textpy).

- `findtext.py` contains the functions [`getNameText()`](crawler.md#findtextpy) and [`findAllIncorrectTexts()`](crawler.md#findtextpy).

## Requirements
The required libraries / frameworks for the project are the following:

* httplib2
	* `pip3 install httplib2` should do the trick
* requests
	* `pip3 install requests` should do the trick
* [Beautiful Soup >= 4.0](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
	* `pip3 install bs4` should do the trick

### Crawl_gutenberg.py

#### `crawler(args)`

**params:** **args**, *list of strings*, a parsed version of the entries given in the command line. For more information, follow [this link](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args).

**return:** None

<br>

#### **Behaviour**

This function will effectively crawl the website gutenberg.org to find the books and then download them. It will first access the [French page](https://www.gutenberg.org/wiki/Category:FR_Genre), crawl it and extract all the `<href>` tags on it. If the `crawl_type` argument is `'styles'`, it will look for specific styles, such as *Théâtre* or *Nouvelles*. Otherwise, it will look for any french book.

Once the books identified, the crawler will access each one of them one by one, download it and store it in the specified folder *(default: './'. Note: a subdir gutenberg will be created.)*.

The function will stop once the number of files downloaded has reached the argument `nb_file`.

---

#### `compressFiles(args)`

**params:** args, a parsed version of the entries given in the command line. For more information, follow [this link](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args).

**return:** None

<br>

#### **Behaviour**

This function will zip the files together using [zipfile](https://docs.python.org/3/library/zipfile.html). If there is no file in the out directory, the program will exit. Otherwise, it will add all the files in the directory to a zip file named <out_dir\>.zip, located in the current directory.	 

## Clean_text.py

**LOGGER INCLUDED !**

#### `concatenateFiles(inputDirectory, outputFile, args)`

**params:**

- inputDirectory, the directory where the files to 			 clean are stored

- outputFile, the path to the file to be created

- args, a parsed version of the entries given in the command line. For more information, follow [this link](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args).

**return:** None

<br>

#### **Behaviour**

This function will concatenate the files contained in the directory *inputDirectory* and store the result in a single file named *outputFile* (*Note: outputFile is in fact a path*).

If no files are in the directory *inputDirectory*, the function will look for a zip file in the current directory. In a zipfile is present, it will extract the files and then concatenate them. Otherwise, the program will terminate.

---

#### `cleanText(inputDirectory, outputFile, args)`

**params:** 

- inputDirectory, the directory where the files to 			 clean are stored

- outputFile, the path to the file to be created

- args, a parsed version of the entries given in the command line. For more information, follow [this link](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args).

**return:** None

<br>

#### **Behaviour**

This function will remove many undesired characters or pattern from a given text file by scanning it character by character. The list of the removed pattern can be found [here](crawler_guide.md#cleaning-the-texts).

---

## Findtext.py

#### `getNameText(string, cmd)`

**params:** 

- string, a string that we want to find in one of the text we have, downloaded by the crawler. Default is ''.

- editor, a string containing the command to launch a text editor, e.g. "gedit" for Gedit or "subl" for Sublime Text. Default is 'gedit'.

**return:** None

<br>

#### **Behaviour**

This function will look for the input string in the text files that we have in the directory of the downloaded files. It will then print the name of the files containing it, and open them in the editor specified. (*Note: for the moment, the directory can only be 'gutenberg'. Adjustements should be done soon*)

---

#### `findAllIncorrectTexts(inputDirectory)`

**params:** inputDirectory, the directory where the files to analyze are stored.

**return:** a list containing the incorrect files names.

<br>

#### **Behaviour**

This function will check if the files of the specified directory don't have any unclosed parenthesis or bracket. Those files MUST be excluded from the cleaning process.