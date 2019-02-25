
# Input
```python
data = [
    [1,2,3],
    [4,5,6],
    [7,8,9]
]
```

Tensorflow's RNN functions expect a tensor of shape \[B, T, ...] as input,
where;
B is the batch size and
T is the lenght in time of feach input (e.g. the number of words in a sentence)
The last dimensionS depend on your data.


### Batch Size
Input will be batches of x dimensional vectors with y features

### Sequence Length
How many vectors to use when learning what to predict
aka Training pattern

**If we want to learn the alfabeth;**

we use an sequence length of 1:
Starting at the beginning of the raw input data,
we can read off the first letter “A” and the next letter as the prediction “B”.
We move along one character and repeat until we reach a prediction of “Z”.

sequence length of 3:
ABC -> D
BCD -> E
CDE -> F


### Features