
# what is this

This is a WORK IN PROGRESS repo aim to recover varibale semantic information from striped binary.

We use the Trex code base: https://github.com/CUMLSec/trex


# installation

Just follow the Trex installation instructions

# how to use

1. prepare dataset

download 300k samples and unpack：https://drive.google.com/file/d/1nxLUK5cswQ44IFmZma3qA3jpgTZYJ32C/view?usp=sharing

preprocess dataset: 
```
python3 commands/ailabel/preprocess_pretrain.py
```

2. train

```
bash commands/ailabel/pretrain.sh
```
