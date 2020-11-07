#%%
import neptune
project = neptune.init('gebob19/672')

# %%
exps = project.get_experiments()
best_exps = [e for e in exps if 'best' in e.get_tags()]

# %%
def get_values(experiment):
    channel_names = list(experiment.get_logs().keys())
    data = experiment.get_numeric_channels_values(*channel_names[:-1])
    return data 

#%%
best_exps_values = [get_values(e) for e in best_exps]

#%%
# remove dropout for better visualizations 
dropout_index = [e.name for e in best_exps].index('DropoutReg')
del best_exps[dropout_index]
del best_exps_values[dropout_index]

#%%
import pathlib 
chart_dir = pathlib.Path()/'charts'
chart_dir.mkdir(exist_ok=True, parents=True)

# %%
import matplotlib.pyplot as plt 

for col in best_exps_values[0].columns[1:]:
    for data, exp in zip(best_exps_values, best_exps): 
        # x = data[data.columns[0]]
        y = data[col]
        plt.plot(y, label=exp.name)
    plt.xlabel('Epoch')
    plt.ylabel(col)
    plt.legend()
    plt.title('{} vs Epoch'.format(col))
    plt.savefig(chart_dir/'{}.png'.format(col))
    plt.clf()

# %%
