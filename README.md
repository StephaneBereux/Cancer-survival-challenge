# Cancer-survival-challenge

Authors : Stéphane Béreux, Gabriel Faivre, Alexandre Marquis, Michael Resplandy, Hugo Stubler

<div style="text-align: center">
<img src="https://raw.githubusercontent.com/StephaneBereux/Cancer-survival-challenge/master/img/symbol.jpg" width="250px" />
</div>

## About

Breast cancer is one of the most common cancers and the second leading cause of cancer death among women.

The Cancer Survability Challenge aim to propose a data challenge about one of the leads to tackle this subject : predict the survivability of a patient depending on its transcriptome.


## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

(Note that three mandatory libraries : `mygene`, `xenaPython` and `scikit-survival` are only available through `pip`, and not with `conda`, that is why they are not listed in the `environment.yml`.)

### Challenge description

Get started on this RAMP with the
[dedicated notebook](starting_kit.ipynb).

### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
