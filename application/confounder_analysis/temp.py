import matplotlib.pyplot as plt
import numpy as np

# put your data in a list like this:
# data = [alpha_g159, alpha_g108, alpha_g141, alpha_g110, alpha_g115, alpha_g132, alpha_g105, alpha_g126]
# as I do not have your data I created some test data
data = [sorted(np.random.normal(0, std, 100)) for std in range(1, 9)]
for ids in range(len(data)):
    plt.violinplot(data[ids],[ids],vert=False)

# labels = ["alpha_g159", "alpha_g108", "alpha_g141", "alpha_g110", "alpha_g115", "alpha_g132", "alpha_g105", "alpha_g126"]
# # add the labels (rotated by 45 degrees so that they do not overlap)
# plt.xticks(range(1, 9), labels, rotation=45)

# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.3)

plt.show()