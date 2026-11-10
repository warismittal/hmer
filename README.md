# Handwritten Math → LaTeX (CROHME)

DenseNet-B encoder + 2D positional encoding + 3-layer Transformer decoder, with beam search.

```
vocab.py    token vocab            data.py     CROHME loader, preprocessing, padding/masks
model.py    encoder/decoder/beam   train.py    training (AMP, warmup+cosine, resume)
metrics.py  ExpRate, ≤1, ≤2        evaluate.py beam-search eval on 2014/16/19
predict.py  single image CLI       app.py      Gradio demo (draw or upload)
```

## Run on Kaggle (free GPU)
1. New Notebook → Accelerator **GPU T4**, Internet **On**.
2. Upload this folder as a Kaggle Dataset (or `git clone` your repo).
3. Get CROHME (the BTTR/CoMER preprocessed zip):
```bash
!git clone --depth 1 https://github.com/Green-Wood/CoMER /kaggle/working/CoMER
!cd /kaggle/working && unzip -q CoMER/data.zip      # -> data/train, data/2014, data/2016, data/2019
```
4. Train (saves `last.pt` every epoch, stops before the 12h limit):
```bash
!cd /kaggle/working/hmer && python train.py --data ../data --out /kaggle/working/runs/base --epochs 200
```
   New session? Re-add the previous output as input, copy `runs/` back, add `--resume`.
5. Evaluate:
```bash
!python evaluate.py --ckpt /kaggle/working/runs/base/best.pt --data ../data
```

## Sanity check first
Overfit ~100 samples: `--epochs 300` on a tiny `train` split. Loss should approach ~1.0 (label smoothing floor) and ExpRate on the same samples should approach 100%. If not, there's a bug.

## Demo (free)
Copy `best.pt`, `app.py`, `predict.py`, `model.py`, `data.py`, `vocab.py` and `requirements.txt` to a Hugging Face Space (Gradio SDK, CPU basic). Or run it locally with `python app.py`.

## Expected
Baseline ≈ 45–52% ExpRate on CROHME 2014 after about 200 epochs. Next steps: bidirectional training (BTTR) or coverage attention (CoMER), plus an ensemble.
