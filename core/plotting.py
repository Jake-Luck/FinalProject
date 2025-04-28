"""
Provides Plotting class to display coordinates, clusters and routes. Uses plotly
to get a world map which plots are drawn onto
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import plotly.graph_objects as go

# Colours to use when drawing clusters or route.
colour_set = [
    "#ff0000", "#00ff00", "#0000ff", "#00ffff", "#ff3fff", "#3f7f7f",
    "#ff7fff", "#7f7f00", "#7f007f", "#000000", "#999999", "#7f0000",
    "#3f3f3f", "#007f00", "#00007f", "#7f00ff", "#007f7f", "#ff3f3f",
    "#007fff", "#ff7f00", "#ff007f", "#7fff00", "#ff7f7f", "#7f7fff",
]


class Plotting:
    """
    Provides static methods to display coordinates, clusters and
    """
    @staticmethod
    def display_clusters(coordinates: ndarray,
                         cluster_assignments: ndarray,
                         num_days: int,
                         centroids: ndarray | None = None,
                         centre: ndarray | None = None,
                         title: str | None = None,
                         save_plot: bool = False) -> None:
        """
        Uses plotly to display provided clusters on a world map centred at a
        given coordinate or the first item in coordinates.
        :param coordinates: Coordinates to display.
        :param cluster_assignments: Cluster assigned to each coordinate.
        :param num_days: Number of days/clusters.
        :param centroids: Central points of clusters, useful for step by step
        plotting of centroid best clustering (such as kmeans).
        :param centre: Point to centre the map on.
        :param title: Title of the plot.
        :param save_plot: Whether to save the plot as an image.
        """
        if centre is None:
            lon = (coordinates[:, 0].min() + coordinates[:, 0].max()) / 2
            lat = (coordinates[:, 1].min() + coordinates[:, 1].max()) / 2
            centre = np.array((lon,lat))

        clusters = [np.where(cluster_assignments == i) for i in range(num_days)]

        figure = go.Figure(go.Scattermap(lat=coordinates[clusters[0], 1][0],
                                         lon=coordinates[clusters[0], 0][0],
                                         mode='markers',
                                         name="Day 1",
                                         marker=dict(
                                             size=10,
                                             color=colour_set[0]
                                         )))

        for i in range(1, num_days):
            figure.add_scattermap(lat=coordinates[clusters[i], 1][0],
                                  lon=coordinates[clusters[i], 0][0],
                                  mode='markers',
                                  name=f"Day {i + 1}",
                                  marker=dict(
                                      size=10,
                                      color=colour_set[i]
                                  ))

        if centroids is not None:
            for i in range(num_days):
                figure.add_scattermap(lat=[centroids[i, 1]],
                                      lon=[centroids[i, 0]],
                                      mode='markers',
                                      name=f"Centroid {i+1}",
                                      marker=dict(
                                          size=20,
                                          color=colour_set[i],
                                          opacity=0.5
                                      ))

        Plotting._update_layout_and_show_figure(title, centre, figure,
                                                save_plot)

    @staticmethod
    def display_coordinates(coordinates: ndarray,
                            centre: ndarray | None = None,
                            title: str | None = None,
                            save_plot: bool = False) -> None:
        """
        Uses plotly to display provided coordinates on a world map centred at a
        given coordinate or the first item in coordinates.
        :param coordinates: Coordinates to display.
        :param centre: Point to centre the map on.
        :param title: Title of the plot.
        :param save_plot: Whether to save the plot as an image.
        """
        if centre is None:
            lon = (coordinates[:, 0].min() + coordinates[:, 0].max()) / 2
            lat = (coordinates[:, 1].min() + coordinates[:, 1].max()) / 2
            centre = np.array((lon,lat))

        figure = go.Figure(go.Scattermap(lat=coordinates[:, 1],
                                         lon=coordinates[:, 0],
                                         mode='markers',
                                         name="Locations",
                                         marker=dict(
                                             size=10,
                                             color=colour_set[0]
                                         )))

        Plotting._update_layout_and_show_figure(title, centre, figure,
                                                save_plot)

    @staticmethod
    def display_route(route: ndarray,
                      coordinates: ndarray,
                      centre: ndarray | None = None,
                      title: str | None = None,
                      evaluation_per_day: ndarray | None = None,
                      durations: ndarray | None = None,
                      save_plot: bool = False) -> None:
        """
        Uses plotly to display provided route on a world map centred at a
        given coordinate or the first item in coordinates.
        :param route: Route to display.
        :param coordinates: Coordinates to display.
        :param centre: Point to centre the map on.
        :param title: Title of the plot.
        :param evaluation_per_day: Evaluation per day. 1D array.
        :param durations: Evaluation per day. 1D array.
        :param save_plot: Whether to save the plot as an image.
        """
        if centre is None:
            lon = (coordinates[:, 0].min() + coordinates[:, 0].max()) / 2
            lat = (coordinates[:, 1].min() + coordinates[:, 1].max()) / 2
            centre = np.array((lon,lat))

        route_per_day = np.split(route, np.where(route == 0)[0])[:-1]

        if len(route_per_day) == 0 or len(route_per_day[0]) == 0:
            route_per_day = [route.tolist()]
        # Add zeroes where necessary
        route_per_day[0] = np.concatenate(([0], route_per_day[0]))
        route_per_day = [np.concatenate((arr, [0])) for arr in route_per_day]

        coordinates_per_day = [coordinates[day] for day in route_per_day]
        durations_per_day = [durations[day] for day in route_per_day]
        if evaluation_per_day is None:
            day_name = f"Day1"
        else:
            day_name = f"Day1: {evaluation_per_day[0]}"
        figure = go.Figure(go.Scattermap(
            mode="markers+lines",
            lat=coordinates_per_day[0][:, 1],
            lon=coordinates_per_day[0][:, 0],
            name=day_name,
            marker=dict(
                size=10,
                color=colour_set[0]
            ),
            text=durations_per_day[0]))

        for i in range(1, len(coordinates_per_day)):
            if evaluation_per_day is None:
                day_name = f"Day{i+1}"
            else:
                day_name = f"Day{i+1}: {evaluation_per_day[i]}"
            figure.add_trace(go.Scattermap(
                mode="markers+lines",
                lat=coordinates_per_day[i][:, 1],
                lon=coordinates_per_day[i][:, 0],
                name=day_name,
                marker=dict(
                    size=10,
                    color=colour_set[i]
                ),
                text=durations_per_day[i]))
        Plotting._update_layout_and_show_figure(title, centre, figure, save_plot)

    @staticmethod
    def plot_line_graph(x_data: ndarray,
                        y_data: ndarray,
                        x_label: str,
                        y_label: str,
                        title: str) -> None:
        """
        Plots a line graph using matplotlib with the given x and y data.

        :param x_data: 1D array representing the data for the x-axis.
        :param y_data: 1D array representing the data for the y-axis.
        :param x_label: Label for the x-axis.
        :param y_label: Label for the y-axis.
        :param title: Title of the graph.
        """
        fig, ax = plt.subplots(dpi=600)

        ax.plot(x_data, y_data)
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid()

        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()

        if x_data[0] not in x_ticks:
            x_ticks = np.append(x_ticks, x_data[0])
        if x_data[-1] not in x_ticks:
            x_ticks = np.append(x_ticks, x_data[-1])
        if y_data[0] not in y_ticks:
            y_ticks = np.append(y_ticks, y_data[0])
        if y_data[-1] not in y_ticks:
            y_ticks = np.append(y_ticks, y_data[-1])

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        
        plt.margins(x=0, y=0)

        plt.show()

    @staticmethod
    def _update_layout_and_show_figure(title: str | None,
                                       centre: ndarray,
                                       figure: go.Figure,
                                       save_plot: bool = False) -> None:
        """
        Updates figure layout and displays it.
        :param title: Title of the plot.
        :param centre: Point to centre the map on.
        :param figure: The figure to update and display.
        :param save_plot: Whether to save the plot as an image.
        """
        if title is not None:
            figure.update_layout(title=title, title_x=0.5, title_y=0.91)

        figure.update_layout(width=1920,
                             height=1080,
                             map=dict(
                                 bearing=0,
                                 center=dict(
                                     lat=centre[1],
                                     lon=centre[0]
                                 ),
                                 pitch=0,
                                 zoom=13,
                                 style='carto-voyager'
                             ))
        figure.show()

        if save_plot:
            figure.update_layout(showlegend=False,
                                 margin=dict(l=0, r=0, t=0, b=0),
                                 title="")
            figure.write_image(f"temp.png", scale=2)

