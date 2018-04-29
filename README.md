# binary-image-classification

Train a model with Keras (Tensorflow backend) for binary classification of images.

Begin with a directory containing 2 subdirectories. Each subdirectory should be named as the label for the images which it contains. E.g.

```
labeled-dataset/
├── bad
│   ├── bad1.png
│   ├── bad10.png
│   ├── bad2.png
│   ├── bad3.png
│   ├── bad4.png
│   ├── bad5.png
│   ├── bad6.png
│   ├── bad7.png
│   ├── bad8.png
│   └── bad9.png
└── good
    ├── good1.png
    ├── good10.png
    ├── good2.png
    ├── good3.png
    ├── good4.png
    ├── good5.png
    ├── good6.png
    ├── good7.png
    ├── good8.png
    └── good9.png
```

Use `split-corpus.py` to split the full corpus of labeled images into train, validate, and testing sets.

These subsets will be randomly selected and stored in subdirectories of the use supplied target directory e.g.:

```
split-dataset/
├── test
│   ├── bad
│   │   ├── bad3.png
│   │   └── bad7.png
│   └── good
│       ├── good3.png
│       └── good4.png
├── train
│   ├── bad
│   │   ├── bad1.png
│   │   ├── bad10.png
│   │   ├── bad2.png
│   │   ├── bad4.png
│   │   ├── bad6.png
│   │   └── bad9.png
│   └── good
│       ├── good1.png
│       ├── good10.png
│       ├── good2.png
│       ├── good6.png
│       ├── good7.png
│       └── good8.png
└── validate
    ├── bad
    │   ├── bad5.png
    │   └── bad8.png
    └── good
        ├── good5.png
        └── good9.png
```
Run `vgg16-with-dropout.py` to train, save, and test the model.

Note that there are hardcoded paths within these scripts, and the image type is hardcoded as .tiff. Be sure to modify these variables for your environment.
