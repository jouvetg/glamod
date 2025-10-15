# git add, commit, push ...

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
