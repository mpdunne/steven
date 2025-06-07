# Steven

**Steven (Sample Things EVENly)** helps you sample your data in nice easy ways, evenly across the range of the data!

Steven is available on PyPI: `pip install steven`.

## How to use Steven

The main method of `steven` is `sample_data_evenly`. This takes as input a sequence-liked object such as a `list`, `tuple`, `np.ndarray` or `pd.Series`, and samples it in such a way that the items returned represent a balanced distribution across the data range.

This is useful for balancing both continuous and discrete data for machine learning applications, among other things!

Let's set up an example and plot the distribution.
```
import numpy as np
import matplotlib.pyplot as plt

from steven.sampling import sample_data_evenly

# Seed for reproducibility
seed = 8675309
np.random.seed(seed)

# Create some data...
data = np.exp(np.random.rand(100_000))
plt.hist(data, bins=50, range=[data.min(), data.max()], label='All data')

# Now sample the data...
data_sampled = sample_data_evenly(data, n_bins=50, sample_size=20_000, random_state=seed)
plt.hist(data_sampled, bins=50, range=[data.min(), data.max()], label='Sampled')

plt.title('Sample data evenly, example')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

The result should look like this:

<img width="377" alt="image" src="https://github.com/user-attachments/assets/051b34ff-66eb-4f8b-953e-aef1289832b3" />

## Keeping track of sampled indices

Optionally, `subset_data_evenly` accepts a `return_ixs` argument, which allows us to keep track of which indexes have been sampled from the inputted data. Continuing with the above example, we can do:

```
sampled_data, ixs = subset_data_evenly(data, n_bins=50, sample_size=10, random_state=seed, return_ixs=True)
```

This will return the sampled data and the ixs as a tuple:

```
>>> sampled_data, ixs 
(array([2.29744662, 1.56124329, 1.75257412, 1.39012692, 1.04761057,
        1.32016874, 1.98088368, 1.84552982, 2.6627304 , 1.5303134 ]),
 [49023, 44730, 83142, 98395, 37441, 81177, 9769, 38017, 3088, 59028])
```


