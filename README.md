# fastText

fastText is a library for efficient learning of word representations and sentence classification.

## Running fastText in FloydHub
[FloydHub](https://www.floydhub.com) is a PaaS for training and deploying your deep learning models in the cloud. Here are the steps to run fastText in Floyd.

### Signup on FloydHub
Click [here](https://www.floydhub.com) to signup for an account. Follow the steps to install the `floyd-cli` and login on your terminal

### Create a Training Dataset
For example
```
$ mkdir traindata
$ cd traindata
$ wget -O mytrainingdata.txt https://raw.githubusercontent.com/dennybritz/cnn-text-classification-tf/master/data/rt-polaritydata/rt-polarity.pos
```

The first step is to upload your data to Floyd's servers. See [docs](http://docs.floydhub.com/home/using_datasets/)
```
$ floyd data init ftTrainData
Data source "ftTrainData" initialized in current directory

$ floyd data upload

Creating data source. Total upload size: 611.7KiB
Uploading files ...
Upload finished
DATA ID                 NAME                    VERSION
----------------------  --------------------  ---------
L7jmn5HVYKSfJbANsYKMmD  floyd_demo/ftTrainData:1          1
```
Your data is now uploaded to the cloud and available to use in your jobs. Make note of the **DATA_ID** (in this example, `L7jmn5HVYKSfJbANsYKMmD`). You can view your uploaded data in your browser using `floyd data output <DATA_ID>`

*P.S*: You can continue to change your data locally and run `floyd data upload`. This will create new versions of your `ftTrainData` dataset.

### Training your fastText Word Representation
We will clone the fastText Github repo and initialize a local project
```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ floyd init fastText
Project "fastText" initialized in current directory
```

Let us run our first training. 
```
$ floyd run --data <DATA_ID> "make && ./fasttext skipgram -input /input/mytrainingdata.txt -output /output/model"

Creating project run. Total upload size: 421.0B
Syncing code ...
RUN ID                  NAME                VERSION
----------------------  ----------------  ---------
pDmQjgKW2hLBHKZQHhQvaa  floyd_demo/fastText:1          1

To view logs enter:
    floyd logs pDmQjgKW2hLBHKZQHhQvaa
```
What happens behind the scenes:
- Your code is uploaded to the server
- A new machine is provisioned and initialized with [this](https://hub.docker.com/r/floydhub/tensorflow/) Docker image.
- The command provided (`make && ./fastText ...`) is executed inside this container.
- The dataset you created in step 1 is mounted into your container by using the `--data <DATA_ID>` flag. Any mounted data is available under `/input`. For example, in our example, the code has access to the training data at `/input/mytrainingdata.txt`
- Any files written to `/output` directory will be saved after your job completes. In this example, we will have `/output/model.bin` and `/output/model.vec` in our output

Make a note of your **RUN_ID** (in this example, pDmQjgKW2hLBHKZQHhQvaa)

You can view the logs of your job run using
```
floyd logs <RUN_ID> -t

2017-03-02 12:09:17,446 INFO - Read 0M words
2017-03-02 12:09:17,518 INFO - Number of words:  2529
2017-03-02 12:09:17,622 INFO - Number of labels: 0
2017-03-02 12:09:27,870 INFO - Progress: 2.6% 
```

Once the training is complete, you can view the generated output (model and word vector) in your browser using
```
floyd output <RUN_ID>
```

*P.S*: You can continue to make changes to your code locally, and each `floyd run` command will sync your latest changes to the server and run it.


## Requirements

**fastText** builds on modern Mac OS and Linux distributions.
Since it uses C++11 features, it requires a compiler with good C++11 support.
These include :

* (gcc-4.6.3 or newer) or (clang-3.3 or newer)

Compilation is carried out using a Makefile, so you will need to have a working **make**.
For the word-similarity evaluation script you will need:

* python 2.6 or newer
* numpy & scipy

## Building fastText

In order to build `fastText`, use the following:

```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
```

This will produce object files for all the classes as well as the main binary `fasttext`.
If you do not plan on using the default system-wide compiler, update the two macros defined at the beginning of the Makefile (CC and INCLUDES).

## Example use cases

This library has two main use cases: word representation learning and text classification.
These were described in the two papers [1](#enriching-word-vectors-with-subword-information) and [2](#bag-of-tricks-for-efficient-text-classification).

### Word representation learning

In order to learn word vectors, as described in [1](#enriching-word-vectors-with-subword-information), do:

```
$ ./fasttext skipgram -input data.txt -output model
```

where `data.txt` is a training file containing `utf-8` encoded text.
By default the word vectors will take into account character n-grams from 3 to 6 characters.
At the end of optimization the program will save two files: `model.bin` and `model.vec`.
`model.vec` is a text file containing the word vectors, one per line.
`model.bin` is a binary file containing the parameters of the model along with the dictionary and all hyper parameters.
The binary file can be used later to compute word vectors or to restart the optimization.

### Obtaining word vectors for out-of-vocabulary words

The previously trained model can be used to compute word vectors for out-of-vocabulary words.
Provided you have a text file `queries.txt` containing words for which you want to compute vectors, use the following command:

```
$ ./fasttext print-vectors model.bin < queries.txt
```

This will output word vectors to the standard output, one vector per line.
This can also be used with pipes:

```
$ cat queries.txt | ./fasttext print-vectors model.bin
```

See the provided scripts for an example. For instance, running:

```
$ ./word-vector-example.sh
```

will compile the code, download data, compute word vectors and evaluate them on the rare words similarity dataset RW [Thang et al. 2013].

### Text classification

This library can also be used to train supervised text classifiers, for instance for sentiment analysis.
In order to train a text classifier using the method described in [2](#bag-of-tricks-for-efficient-text-classification), use:

```
$ ./fasttext supervised -input train.txt -output model
```

where `train.txt` is a text file containing a training sentence per line along with the labels.
By default, we assume that labels are words that are prefixed by the string `__label__`.
This will output two files: `model.bin` and `model.vec`.
Once the model was trained, you can evaluate it by computing the precision and recall at k (P@k and R@k) on a test set using:

```
$ ./fasttext test model.bin test.txt k
```

The argument `k` is optional, and is equal to `1` by default.

In order to obtain the k most likely labels for a piece of text, use:

```
$ ./fasttext predict model.bin test.txt k
```

where `test.txt` contains a piece of text to classify per line.
Doing so will print to the standard output the k most likely labels for each line.
The argument `k` is optional, and equal to `1` by default.
See `classification-example.sh` for an example use case.
In order to reproduce results from the paper [2](#bag-of-tricks-for-efficient-text-classification), run `classification-results.sh`, this will download all the datasets and reproduce the results from Table 1.

If you want to compute vector representations of sentences or paragraphs, please use:

```
$ ./fasttext print-vectors model.bin < text.txt
```

This assumes that the `text.txt` file contains the paragraphs that you want to get vectors for.
The program will output one vector representation per line in the file.

## Full documentation

Invoke a command without arguments to list available arguments and their default values:

```
$ ./fasttext supervised
Empty input or output path.

The following arguments are mandatory:
  -input              training file path
  -output             output file path

The following arguments are optional:
  -lr                 learning rate [0.1]
  -lrUpdateRate       change the rate of updates for the learning rate [100]
  -dim                size of word vectors [100]
  -ws                 size of the context window [5]
  -epoch              number of epochs [5]
  -minCount           minimal number of word occurences [1]
  -minCountLabel      minimal number of label occurences [0]
  -neg                number of negatives sampled [5]
  -wordNgrams         max length of word ngram [1]
  -loss               loss function {ns, hs, softmax} [ns]
  -bucket             number of buckets [2000000]
  -minn               min length of char ngram [0]
  -maxn               max length of char ngram [0]
  -thread             number of threads [12]
  -t                  sampling threshold [0.0001]
  -label              labels prefix [__label__]
  -verbose            verbosity level [2]
  -pretrainedVectors  pretrained word vectors for supervised learning []
```

Defaults may vary by mode. (Word-representation modes `skipgram` and `cbow` use a default `-minCount` of 5.)

## References

Please cite [1](#enriching-word-vectors-with-subword-information) if using this code for learning word representations or [2](#bag-of-tricks-for-efficient-text-classification) if using for text classification.

### Enriching Word Vectors with Subword Information

[1] P. Bojanowski\*, E. Grave\*, A. Joulin, T. Mikolov, [*Enriching Word Vectors with Subword Information*](https://arxiv.org/abs/1607.04606)

```
@article{bojanowski2016enriching,
  title={Enriching Word Vectors with Subword Information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.04606},
  year={2016}
}
```

### Bag of Tricks for Efficient Text Classification

[2] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [*Bag of Tricks for Efficient Text Classification*](https://arxiv.org/abs/1607.01759)

```
@article{joulin2016bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.01759},
  year={2016}
}
```

(\* These authors contributed equally.)

## Resources

You can find the preprocessed YFCC100M data used in [2] at https://research.facebook.com/research/fasttext/

Pre-trained word vectors for 90 languages are available [*here*](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md).

## Join the fastText community

* Facebook page: https://www.facebook.com/groups/1174547215919768
* Google group: https://groups.google.com/forum/#!forum/fasttext-library
* Contact: [egrave@fb.com](mailto:egrave@fb.com), [bojanowski@fb.com](mailto:bojanowski@fb.com), [ajoulin@fb.com](mailto:ajoulin@fb.com), [tmikolov@fb.com](mailto:tmikolov@fb.com)

See the CONTRIBUTING file for information about how to help out.

## License

fastText is BSD-licensed. We also provide an additional patent grant.
