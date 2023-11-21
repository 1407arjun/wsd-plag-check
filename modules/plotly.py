import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter


def plot_heatmap(scores, testing_data, n):
    # set width and height
    width = 8
    height = np.ceil(len(testing_data)/width).astype("int32")
    print("Width, Height:", width, ",", height)

    # copy scores to rectangular blank array
    a = np.zeros(width*height)
    a[:len(scores)] = scores
    diff = len(a) - len(scores)

    # apply gaussian smoothing for aesthetics
    a = gaussian_filter(a, sigma=1.0)

    # reshape to fit rectangle
    a = a.reshape(-1, width)

    # format labels
    labels = [" ".join(testing_data[i:i+width])
              for i in range(n-1, len(testing_data), width)]
    labels_individual = [x.split() for x in labels]
    labels_individual[-1] += [""]*diff
    labels = [f"{x:60.60}" for x in labels]

    # create heatmap
    fig = go.Figure(data=go.Heatmap(
                    z=a, x0=0, dx=1,
                    y=labels, zmin=0, zmax=1,
                    customdata=labels_individual,
                    hovertemplate='%{customdata} <br><b>Score:%{z:.3f}<extra></extra>',
                    colorscale="blues"))
    fig.update_layout({"height": height*28, "width": 1000,
                      "font": {"family": "Courier New"}})
    fig['layout']['yaxis']['autorange'] = "reversed"

    return fig
