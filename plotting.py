import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns


def share_to_colour(share, intensity=None):
    red = max(1 - 2*share, 0)
    blue = max(2*share - 1, 0)
    green = 1 - 2 * abs(share-0.5)
    value_list = [red, green, blue] if intensity is None else [red, green, blue, intensity]
    return np.array(value_list)

def scale_fn(df):
    return np.sin(np.sin(df * np.pi / 2) * np.pi / 2)


def plot_incl_intensity(effort_matrix):
    # Prepare data
    x = effort_matrix['share'].index
    y = effort_matrix['share'].columns
    matrix = np.stack([effort_matrix['share'], effort_matrix['intensity']], axis=-1)
    z = np.array([
        [
            share_to_colour(share, intensity)
            for share, intensity in row
        ] 
        for row in matrix
    ]).transpose(1,0,2)

    # Plot data
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(x, y, z)
    ax.axes.set_aspect('equal')
    return fig


def plot_heatmap(matrix, intensity_matrix=None, colourmap='coolwarm', origin='upper', title='Heatmap', xlabel='Quarter hour', ylabel='Weekday'):
    # https://stackoverflow.com/questions/65634008/convert-quadmesh-generated-by-pcolormesh-to-an-rgba-array
    # Prepare data
    y = matrix.index
    x = matrix.columns
    z = matrix
    intensity = np.ones_like(z) if intensity_matrix is None else scale_fn(intensity_matrix)
    # Make plot
    # fig, ax = plt.subplots()
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    _, ax1 = plt.subplots()
    mesh = ax1.pcolormesh(x, y, z, cmap=colourmap)
    # ax1.axes.set_aspect('equal')
    # ax1.set_title("without transparency")
    # Retrieve rgba values of the quadmesh object
    rgbas = mesh.to_rgba(z, alpha=intensity)
    # Plot back with imshow 
    fig, ax2 = plt.subplots()
    mesh_extent = [x.min(), x.max(), y.max(), y.min()]
    mesh2 = ax2.imshow(rgbas, origin=origin, extent=mesh_extent)
    ax2.set_title(title)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    # plt.tight_layout()
    # Show plot
    # fig.show()
    return fig


def plot_heatmap_seaborn(matrix, intensity_matrix=None, colourmap='coolwarm', origin='upper', title='Heatmap', xlabel='Quarter hour', scale_fn=scale_fn, ylabel='Weekday', **kwargs):
    fig, ax = plt.subplots()
    heatmap = sns.heatmap(
        matrix,
        cmap=colourmap,
        annot=False,  # Annotate with values if needed
        # linewidths=0.5,
        square=True,
        alpha=None if intensity_matrix is None else scale_fn(intensity_matrix),  # Set opacity using intensity values
        # cbar_kws={"label": "Work Share"},
        ax=ax,
        **kwargs
    )
    ax.set_title(title)
    return fig

def plot_heatmap_plotly(matrix, intensity_matrix=None, colourmap='coolwarm'):
    cmap = plt.cm.viridis
    # Convert the colormap to a Plotly-compatible colorscale
    colourscale_coolwarm = [
        [i / 255, f"rgba({int(cmap(i)[0] * 255)}, {int(cmap(i)[1] * 255)}, {int(cmap(i)[2] * 255)}, {cmap(i)[3]})"]
        for i in range(256)
    ]
    # Create the heatmap
    heatmap_share = go.Heatmap(
        z=matrix.values,  # Heatmap values (Work Share)
        x=matrix.columns,  # Weekdays
        y=matrix.index,  # Calendar weeks
        colorscale=colourscale_coolwarm,  # Custom colormap
        zmin=0, zmax=1,  # Range of work share (0 to 1)
        # opacity=pivot_intensity.values,  # Set opacity based on intensity
        colorbar=dict(title="Intensity"),  # Colorbar title
        hovertemplate="<b>Weekday:</b> %{x}<br>" +
                    "<b>Calendar Week:</b> %{y}<br>" +
                    "<b>Intensity:</b> %{z:.2f}<extra></extra>",
    )
    data = [heatmap_share]
    if not intensity_matrix is None:
        # Create an overlay for intensity (opacity grid)
        # Use white or black with varying transparency to show intensity
        colourscale_transparency = [
            (0.0, "hsla(0, 0, 1.0, 1.0)"),  # Fully opaque
            (1.0, "hsla(0, 0, 1.0, 0.0)"),  # Fully transparent
        ]
        overlay_intensity = go.Heatmap(
            z=intensity_matrix.values,  # Use uniform values for overlay
            x=intensity_matrix.columns,
            y=intensity_matrix.index,
            colorscale=colourscale_transparency,  # White fades based on intensity
            zmin=0, zmax=1,
            showscale=False,  # Hide colorbar for overlay
            # opacity=0.5,  # Apply intensity as transparency
            # colorbar=dict(title="Work Share"),  # Colorbar title
            hoverinfo="skip",  # Skip hover info for overlay,
        )
        data.append(overlay_intensity)
    # Combine both layers in the figure
    fig = go.Figure(data=data)
    # Add title and labels
    fig = fig.update_layout(
        title="Interactive Heatmap: Intensity (Colour)",
        xaxis=dict(title="Quarter hour of day"),
        yaxis=dict(title="Weekday", autorange="reversed"),  # Reversed y-axis
        width=800,
        height=800,
    )
    # fig.show()
    return fig