''' function to create loss and accuracy graphics '''
sns.set_style("darkgrid")
sns.set(rc={'figure.figsize':(15, 6)})

def acc_loss_graph(accuracies, losses, net_list, save_file_name='plot.png', download=False):
    ''' net_list - list of the networks to explore. If one net given - use list anyway '''
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for experiment_id in net_list:
        ax1.plot(accuracies[experiment_id], label=experiment_id)
    ax1.legend()
    ax1.set_title('Validation Accuracy')
    fig.tight_layout()

    for experiment_id in net_list:
        ax2.plot(losses[experiment_id], label=experiment_id)
    ax2.legend()
    ax2.set_title('Validation Loss')

    fig.tight_layout()

    if download:
        fig.savefig(save_file_name)