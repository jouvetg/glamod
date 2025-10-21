# git add, commit, push ...

# clean
rm -r 02_notebook/outputs
rm -r 02_notebook/data/RGI2000-v7.0-G-11-02596
rm -r 02_notebook/data/input.nc
rm -r 02_notebook/__pycache__

# ziip ex
sh zip-exercices.sh

# rm the _build folder
rm -rf _build

# make the build
jupyter-book build .

# check the results in a browser
open _build/html/index.html

# publish the page
ghp-import -n -p -f _build/html
