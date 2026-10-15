# remove IGM outputs, downloaded data and build artefacts (keeps the tracked tutorial data)
rm -rf _build
find . -path ./.git -prune -o -type d \( -name outputs -o -name __pycache__ -o -name .ipynb_checkpoints \) -print -exec rm -rf {} +
find . -path ./.git -prune -o -type d -path "*/data/RGI*" -print -exec rm -rf {} +
find . -path ./.git -prune -o -type f -name "input_RGI*.nc" -print -exec rm -f {} +
rm -f project/my-glacier/experiment/params_future_*.yaml
