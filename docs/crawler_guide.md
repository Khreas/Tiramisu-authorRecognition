# Getting started with the Gutenberg crawler

# WARNING !

The robot crawling of their website is prohibited by gutenberg for anyone outside of the ESIEA. This crawler was done with the organization approval because the mirroring system of the Gutenberg website was down at the moment of the project. Before using it, please check if their mirroring system works : [Gutenberg mirrors](http://www.gutenberg.org/wiki/Gutenberg:Information_About_Robot_Access_to_our_Pages)

## What can it do ?

The present [crawler](crawler_guide.md#downloading-the-texts) has been developped to download texts made available by the [Gutenberg Organization](https://www.gutenberg.org). Currently, you can either select to download texts by style or texts of a given language. *Note: only french is supported at the moment.*

A [cleaner](crawler_guide.md#cleaning-the-texts) has also been created; the texts downloaded from the Gutenberg Organization are watermarked. In order for our network to work, those watermarks need to be removed.

Finally, in order to navigate smoothly between the downloaded files, a [finder](crawler_guide.md#finding-a-text) has been implemented. Simply enter a sentence and it will tell you which file contains it.

## Downloading the texts

The crawler developed for the Tiramisu project can be used as follow:

`python crawl_gutenberg.py --nb_files NB_FILES --out_dir OUT_DIR --crawl_type CRAWL_TYPE`

- __NB_FILES__: *integer* indicating the number of files that will be downloaded by the crawler. Default is 100.
- __OUT_DIR__: *string* indicating the path to the directory in which the files will be stored. Default is './'
- __CRAWL_TYPE__: *string* indicating the type of crawling to be done. Currently, the supported crawling types are 'authors' and 'styles'. Default is 'authors'.
- __--help__, __-h__ : print this help to the terminal.

---

The crawler currently only supports french language. More languages will be added soon.

## Cleaning the texts

If you look into the downloaded texts, you will notice that they have a particular structure, such as:

>This eBook is for the use of anyone anywhere at no cost and with
>almost no restrictions whatsoever.  You may copy it, give it away or
>re-use it under the terms of the Project Gutenberg License included
>with this eBook or online at www.gutenberg.net


>Title: Histoire comique

>Author: Anatole France

>Release Date: December 18, 2005 [EBook #17345]

>Language: French

>Character set encoding: UTF-8

>\*\*\* START OF THIS PROJECT GUTENBERG EBOOK HISTOIRE COMIQUE \*\*\*

> 						. . .

>    BALTHASAR                                                1 vol.

>    LE CRIME DE SYLVESTRE BONNARD (_Ouvrage couronné

>    par l'Académie française_)                               1 --

>    L'ÉTUI DE NACRE                                          1 --

>    LE JARDIN D'ÉPICURE                                      1 --

> 						. . .

>I

>

>

>C'était dans une loge d'actrice, à l'Odéon. Sous la lampe
>électrique, Félicie Nanteuil, la tête poudrée, du bleu aux
>paupières, du rouge aux joues et aux oreilles, du blanc au cou et

Obviously, those information aren't relevant for the analysis of the text itself. It will corrupt our data; its removal is necessary.

---

The list of the removed pattern are the following:

- **START OF** ... **END OF**
- **Project Gutenberg** ... **subscribe.**
- **< tag >** ... **< /tag >**
- **[**...**]**
- **(**...**)**
- _

The list of the converted pattern are the following:

- **\-\-** to **—**
- **<<** to **«**
- **\>\>** to **»** 

---

The cleaner developed for the Tiramisu project can be used as follow:


`python clean_text.py --nb_files NB_FILES --in_dir IN_DIR out_dir OUT_DIR --name NAME`

- __NB_FILES__: *integer* indicating the number of files cleaned by the cleaner. Default is 100.
- __OUT_DIR__: *string* indicating the path in which the output file will be stored. Default is 'data/french/input.txt'
- __NAME__: *string* indicating the name of the file to be created. Default is 'input.txt'


**NOTE**

Please ensure that your text don't contain any unclosed parenthesis or bracket. A [function](crawler.md#findtextpy) has been developed to help you do that.



## Finding a text

A finder has been implemented. If you need to look which files contains a specific sentence, just type:

`python findtext.py 'sentence'`

The program will then output you a list of files in which the sentence has been found. If *Gedit* is installed, the concerned file will be directly opened in it.

---

Need a more concrete / detailed example ? [Follow me !](example_crawler.md)