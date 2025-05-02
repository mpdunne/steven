# Steven

**Steven (Sample Things EVENly)** helps you sample your data in nice easy ways, evenly across the range of the data!

## How to use Steven

The main method of `steven` is `sample_data_evenly`. This takes as input a sequence-liked object such as a `list`, `tuple`, `np.ndarray` or `pd.Series`, and samples it in such a way that the items returned represent a balanced distribution across the data range.

This is useful for balancing both continuous and discrete data for machine learning applications, among other things!

Here is an example that you can run:
```
import numpy as np
import matplotlib.pyplot as plt

from steven.sampling import sample_data_evenly

# Seed for reproducibility
seed = 8675309

# Create some data...
data = np.exp(np.random.rand(100_000))
plt.hist(data, bins=50, range=[data.min(), data.max()], label='All data')

# Now sample the data...
data_sampled = sample_data_evenly(data, n_bins=50, sample_size=20_000, random_state=seed)
plt.hist(data_sampled, bins=50, range=[data.min(), data.max()], label='Sampled')

plt.legend()
plt.show()
```
