"""
Provides Plotting class to display coordinates, clusters and routes. Uses plotly
to get a world map which plots are drawn onto
"""

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
    def display_coordinates(coordinates: ndarray,
                            centre: ndarray | None = None,
                            title: str | None = None) -> None:
        """
        Uses plotly to display provided coordinates on a world map centred at a
        given coordinate or the first item in coordinates.
        :param coordinates: Coordinates to display.
        :param centre: Point to centre the map on.
        :param title: Title of the plot.
        """
        if centre is None:
            centre = coordinates[0]

        figure = go.Figure(go.Scattermap(lat=coordinates[:, 1],
                                         lon=coordinates[:, 0],
                                         mode='markers',
                                         name="Locations",
                                         marker=dict(
                                             size=10,
                                             color=colour_set[0]
                                         )))

        if title is not None:
            figure.update_layout(title=title)

        figure.update_layout(autosize=True,
                             map=dict(
                                 bearing=0,
                                 center=dict(
                                     lat=centre[1],
                                     lon=centre[0]
                                 ),
                                 pitch=0,
                                 zoom=11,
                                 style='carto-voyager'
                             ))

        figure.show()

    @staticmethod
    def display_route(route: ndarray,
                      coordinates: ndarray,
                      centre: ndarray | None = None,
                      title: str | None = None) -> None:
        """
        Uses plotly to display provided route on a world map centred at a
        given coordinate or the first item in coordinates.
        :param route: Route to display.
        :param coordinates: Coordinates to display.
        :param centre: Point to centre the map on.
        :param title: Title of the plot.
        """
        if centre is None:
            centre = coordinates[0]

        route_per_day = np.split(route, np.where(route == 0)[0])[:-1]

        # Add zeroes where necessary
        route_per_day[0] = np.concatenate(([0], route_per_day[0]))
        route_per_day = [np.concatenate((arr, [0])) for arr in route_per_day]

        coordinates_per_day = [coordinates[day] for day in route_per_day]

        figure = go.Figure(go.Scattermap(
            mode="markers+lines",
            lat=coordinates_per_day[0][:, 1],
            lon=coordinates_per_day[0][:, 0],
            name="Day 1",
            marker=dict(
                size=10,
                color=colour_set[0]
            )))

        for i in range(1, len(coordinates_per_day)):
            figure.add_trace(go.Scattermap(
                mode="markers+lines",
                lat=coordinates_per_day[i][:, 1],
                lon=coordinates_per_day[i][:, 0],
                name=f"Day {i+1}",
                marker=dict(
                    size=10,
                    color=colour_set[i]
                )))

        if title is not None:
            figure.update_layout(title=title)

        figure.update_layout(
            autosize=True,
            map=dict(
                bearing=0,
                center=dict(
                    lat=centre[1],
                    lon=centre[0]
                ),
                pitch=0,
                zoom=11,
                style='carto-voyager'
             ))

        figure.show()

    @staticmethod
    def display_clusters(coordinates: ndarray,
                         cluster_assignments: ndarray,
                         num_days: int,
                         centroids: ndarray | None = None,
                         centre: ndarray | None = None,
                         title: str | None = None) -> None:
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
        """
        if centre is None:
            centre = coordinates[0]

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
                                  name=f"Day {i+1}",
                                  marker=dict(
                                      size=10,
                                      color=colour_set[i]
                                  ))

        if title is not None:
            figure.update_layout(title=title)

        figure.update_layout(autosize=True,
                             map=dict(
                                 bearing=0,
                                 center=dict(
                                     lat=centre[1],
                                     lon=centre[0]
                                 ),
                                 pitch=0,
                                 zoom=11,
                                 style='carto-voyager'
                             ))

        if centroids is None:
            figure.show()
            return

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
        figure.show()
