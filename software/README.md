How to run:
```bash
cd software
wget -O model/stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget -O model/tokenizer.bin https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
make run
./run model/stories15M.bin -z model/tokenizer.bin
```


To run the smaller version:
```bash
wget -O model/stories260K.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin
wget -O model/tok512.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin
./run model/stories260K.bin -z model/tok512.bin
```

To Quantize the smaller version:
```bash
# we don't have out own repo yet so we have to:
git clone https://github.com/karpathy/llama2.c.git
cd llama2.c
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.pt
python export.py stories260K-q8.bin   --checkpoint stories260K.pt   --version 2
```