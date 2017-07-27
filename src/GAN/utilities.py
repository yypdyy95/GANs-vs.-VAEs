import numpy as np
import matplotlib.pyplot as plt
'''
plotting weight matrices of convolutional layer

Inputs:
- layer: Conv layer of which weights should be visualized
'''

def visualize_weights(layer, max_filters = 16):

    weights = layer.get_weights()
    weights = np.array(weights[0])

    number_of_filters = weights.shape[3]
    if max_filters < number_of_filters:
        number_of_filters = max_filters
    number_of_planes = weights.shape[2]

    fig, ax = plt.subplots( number_of_filters,number_of_planes)
    fig.set_size_inches(12,8)
    fig.suptitle('Visualisation of Filters', fontsize=20)
    left  = 0.05    # the left side of the subplots of the figure
    right = 0.98    # the right side of the subplots of the figure
    bottom = 0.02   # the bottom of the subplots of the figure
    top = 0.85      # the top of the subplots of the figure
    wspace = 0.05   # the amount of width reserved for blank space between subplots
    hspace = 0.05   # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    print(ax.shape)
    for i in range(number_of_planes):
        for j in range(number_of_filters):

            ax[j,i].axis('off')
            im = ax[j,i].imshow(weights[:,:,i,j], cmap = 'plasma')
    fig.colorbar(im, ax=ax.ravel().tolist())
    #plt.show()
'''
get_weight_distributions:
    visualize distribution of weights of Conv Layers network
arguments:
    - network: network to get distribution from
    - bins: bins to calculate histogram

returns:
    - numpy histogram: tuple
'''


def get_weight_distributions(network, bins = 500):

    #hist = np.zeros(bins)
    all_weights = []
    hists = []
    for layer in network.layers:
        if "conv" in layer.get_config()['name']: # don't take
            weights = layer.get_weights()
            if weights != []:
                weights = np.reshape(weights[0], (-1))
            hist = np.array(np.histogram(weights, bins = bins, normed = True))
            out_shape = layer.output_shape
            hists.append(hist)

            #all_weights= np.concatenate([all_weights, weights])#.append(weights)

    #print(all_weights.shape)

    return np.array(hists)#np.histogram(all_weights, bins = bins, normed = True)
    
    
    
def get_model_name(discriminator, filters, filtersize, dropout_rate, batch_norm = False, deconv =False):
    if discriminator:
        name = "dis"
        if batch_norm:
            name += "_bn"
    else:
        name = "gen"
        if deconv:
            name += "_de"
        else:
            name += "_up"
    name += "_f" + str(filters)
    name += "_fs" + str(filtersize)
    name += "_d" + str(dropout_rate)

    name += ".h5"
    return name

    
    