Diary
-----

# 2016 Oct 16

## Code Repository Setup
- Setup gitlab account
    - setup new project : kaggle-all-state
    - checked in datasets as first commit

## Basic data exploration:

- Check sizes
    `
    $ ls input
    -rw-r--r-- 1 swati swati 44M Sep 26 12:43 test.csv
    -rw-r--r-- 1 swati swati 67M Sep 26 12:42 train.csv
    `

- Number of lines
`$ wc -l input/*.txt`
188319 input/train.csv
125547 input/test.csv


- Number of columns
`
$ head -n 1 input/train.csv  | tr "," " " | awk '{ print NF}'
132
$ head -n 1 input/test.csv   | tr "," " " | awk '{ print NF}'
131
`


- Column Names
`
$ head -n 1 input/train.csv  | tr "," " " 
id cat1 ... cat116 cont1 ... cont14 loss
$ head -n 1 input/test.csv   | tr "," " "
id cat1 ... cat116 cont1 ... cont14
`

# 2016 Oct 17

- Continued data exploration
- Code is in code/basic-explore/explre.ipynb
- Jupyter notebook cheatsheet : https://www.cheatography.com/weidadeyue/cheat-sheets/jupyter-notebook/
- Installed yhat/ggplot : conda install -c bokeh ggplot
- Adding Git filters for ipynb : http://pascalbugnion.net/blog/ipython-notebooks-and-git.html


