import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl


def get_shape(x):
    try:
        return x.shape
    except:
        try:
            return (len(x), len(x[0]))
        except:
            pass

    assert False, f"cannot get shape of {x}"
    
    
def get_scaler(x):
    return x.item()
    

def visualize_matrix(data, annotate=True, xlabels=None, ylabels=None, path='plot_matrix.png', dpi=1000, **kwargs):
    figsize = kwargs.get('figsize', (16, 16))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data)
    
    data_shape = get_shape(data)
    assert len(data_shape) == 2, f"data must be 2-D, but found {len(data_shape)}-D"
    ylen, xlen = data_shape

    # Show all ticks and label them with the respective list entries
    if xlabels:
        assert len(xlabels) == xlen, f"xlabels must have length {xlen}, but found {len(xlabels)}"
        ax.set_xticks(np.arange(xlen), labels=xlabels)
    if ylabels:
        assert len(ylabels) == ylen, f"ylabels must have length {ylen}, but found {len(ylabels)}"
        ax.set_yticks(np.arange(ylen), labels=ylabels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if annotate:
        for i in range(ylen):
            for j in range(xlen):
                text = ax.text(j, i, get_scaler(data[i, j]),
                            ha="center", va="center", color="w", fontsize=kwargs.get('fontsize', 'small'))  # float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}

    title = kwargs.get('title', None)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    plt.show()
    plt.savefig(path, dpi=dpi)
    
    
def test_visualize_matrix():
    vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
    farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
    harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                            [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                            [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                            [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                            [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                            [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                            [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
    visualize_matrix(harvest, xlabels=vegetables, ylabels=farmers, path='plot_matrix.png', dpi=1000, title='Harvest of local farmers (in tons/year)')
    

if __name__ == '__main__':
    test_visualize_matrix()