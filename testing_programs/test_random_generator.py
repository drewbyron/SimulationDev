import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np

from spec_tools.formatting.reformatter import config_tick_reformatter
from spec_tools.formatting.reformatter import sci_reformatter
from spec_tools.spec_fitters.he6_random_beta import He6_random_beta

#plotting histogram
fig2, ax2  = plt.subplots(figsize=(9,6))

#build random beta generator
beta_generator = He6_random_beta()
events = int(1e6)

#generating random gaussian data
xhist = []
for k in range(events):

    xhist.append(beta_generator.generate_random_beta()[0])
    print("Generated Betas: {}/{}".format(k,events))
    
num_bins = 100

tick_reformatter = config_tick_reformatter(2)

n, bins, patches = ax2.hist(xhist,num_bins,facecolor = "blue", alpha = 0.9)

ax2.grid(axis='y',alpha=0.75)
ax2.set_xlabel("Energy")
ax2.xaxis.set_major_formatter(tick.FuncFormatter(tick_reformatter))
ax2.set_ylabel("#Counts")
plt.title("Random Beta Generator Test")
plt.subplots_adjust(left=0.12,bottom=0.12)
fig2.show()

