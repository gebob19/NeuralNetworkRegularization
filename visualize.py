#%%
import neptune
import matplotlib.pyplot as plt 
project = neptune.init('gebob19/672-cifar')

# %%
exps = project.get_experiments()
best_exps = [e for e in exps if 'best' in e.get_tags()]

# %%
def get_values(experiment):
    channel_names = list(experiment.get_logs().keys())
    data = experiment.get_numeric_channels_values(*channel_names)
    return data 

#%%
best_exps_values = [get_values(e) for e in best_exps]

# #%%
# # remove dropout for better visualizations 
# exp_names = [e.name for e in best_exps]
# dropout_index = exp_names.index('DropoutReg')
# del best_exps[dropout_index]
# del best_exps_values[dropout_index]

#%%
for data, exp in zip(best_exps_values, best_exps):
    name = exp.name
    params = exp.get_parameters()
    l = params['reg_constant']
    if name == 'DropoutReg':
        l = params['dropout_constant']
    if name == 'Baseline':
        l = 'n/a'
    test_acc = data['test_acc'].values[-1]
    train_acc = data['train_acc'].values[-2]
    kreg, dreg = params['kernel_regularization'], params['dense_regularization']

    # loss = data['loss'].values[-2]
    # largest_singular_value = data['sum_singular_value'].values[-2]
    # w_norm = data['w_norm'].values[-2]

    # print('{}: {:.2f} {:.2f} {} {:.2f} {:.2f}'.format(name, test_acc, train_acc, l, largest_singular_value, w_norm)) 

    print('{}: {:.3f} {:.3f} {} {} {}'.format(name, test_acc, train_acc, l, kreg, dreg)) 

#%%
import pathlib 
chart_dir = pathlib.Path()/'charts/charts-cifar/'
chart_dir.mkdir(exist_ok=True, parents=True)

# %%
for col in best_exps_values[0].columns[1:]:
    for data, exp in zip(best_exps_values, best_exps): 
        y = data[col]
        plt.plot(y, label=exp.name)
    plt.xlabel('Epoch')
    plt.ylabel(col)
    plt.legend()
    plt.title('{} vs Epoch'.format(col))
    plt.savefig(chart_dir/'{}.png'.format(col))
    plt.clf()

# %%
plt.title('{} vs Epoch'.format('Accuracy'))

from matplotlib.lines import Line2D
line = Line2D([0,1],[0,1],linestyle='-', color='black')
line2 = Line2D([0,1],[0,1],linestyle='--', color='black')
colors = ['red', 'green', 'blue', 'orange', 'purple']
for data, exp, c in zip(best_exps_values, best_exps, colors): 
    plt.plot(data['train_acc'], color=c, label=exp.name)
    plt.plot(data['val_acc'], '--', color=c)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
l1 = plt.legend(loc=4)
plt.legend([line, line2], ['train', 'val'], loc=2)
plt.gca().add_artist(l1)
plt.savefig(chart_dir/'accuracy-per-epoch.png')

# %%
