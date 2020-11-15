# Experiments with LIME 

This repository contains code for experiments on how the choice of hyperparameters affects explanations of 
[LIME](https://github.com/marcotcr/lime).

# Motivation 

Despite being a popular framework for ad-hoc machine learning interpretation, LIME has several major disadvantages: 

1. Explanations of similar examples might be totally different;

2. LIME explanations fidelity is low;

3. Explanation depends on the choice of LIME hyperparameters.

You may find details on these problems in my blog post on LIME framework problems: 
[What’s Wrong with LIME](https://towardsdatascience.com/whats-wrong-with-lime-86b335f34612) 

or in the following articles, which I used for writing the blog post: 

1. [On the Robustness of Interpretability Methods](https://arxiv.org/abs/1806.08049) 

2. [A study of data and label shift in the LIME framework](https://arxiv.org/abs/1910.14421)

# Reproducibility of Results

Run 

```python
python -m virtualenv lime_experiments
source lime_experiments/bin/activate
pip install -r requirements.txt
```

Then run `lime_experiments.ipynb` notebook. 
