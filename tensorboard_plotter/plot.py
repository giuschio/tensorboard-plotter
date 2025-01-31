#!/usr/bin/env python

import os
import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator

# -------------------------------------------------------------------
# Matplotlib font settings (vector fonts for PS/PDF)
# -------------------------------------------------------------------
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams['text.usetex'] = True


class TensorBoardPlotter:
    """
    A class for reading, smoothing, and plotting TensorBoard scalar metrics
    from multiple runs (organized by 'versions') with an interactive legend.
    """

    def __init__(self,
                 log_dir: str,
                 x_axis: str,
                 metric: str,
                 alpha: float,
                 x_axis_label: str = None,
                 y_axis_label: str = None,
                 plot_title: str = None,
                 plot_mean_only: bool = False):
        """
        Parameters
        ----------
        log_dir : str
            Directory containing subfolders (each subfolder is considered a 'version').
        x_axis : str
            Name of the column to use for the x-axis (e.g. "step", "wall_time").
        metric : str
            Name of the TensorBoard scalar metric to plot (e.g. "train_loss").
        alpha : float
            User-specified smoothing factor in [0, 1].
            * alpha=0.0 => no smoothing
            * alpha=1.0 => maximum smoothing

        x_axis_label : str, optional
            Custom label for the x-axis. If None or empty, defaults to the value of `x_axis`.
        y_axis_label : str, optional
            Custom label for the y-axis. If None or empty, defaults to the value of `y_axis`.
        plot_title : str, optional
            Title to display at the top of the plot. If None or empty, no title is shown.
        plot_mean_only : bool, optional
            If True, only plots the mean line of each version (no individual runs).
            If False (default), plots both the runs and the mean line.
        """
        self.log_dir = log_dir
        self.x_axis = x_axis
        self.metric = metric
        self.alpha = alpha
        # If user doesn't provide a y-axis label, default to y_axis
        self.y_axis_label = y_axis_label if y_axis_label else metric
        self.x_axis_label = x_axis_label if x_axis_label else x_axis
        self.plot_title = plot_title
        self.plot_mean_only = plot_mean_only

        # -------------------------------------------------------------------
        # Matplotlib font settings (vector fonts for PS/PDF)
        # Repeat here in case I am only importing the class
        # -------------------------------------------------------------------
        mpl.rcParams["ps.fonttype"] = 42
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams['text.usetex'] = True

    def _extract_and_smooth_metrics(self,
                                    run_path,
                                    alpha_i):
        """
        Extracts and smooths TensorBoard scalar data from a single run folder.

        Parameters
        ----------
        run_path : str
            Directory containing the event file for one run.
        alpha_i : float
            Inverted smoothing factor actually used by pandas ewm().

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ["wall_time", "step", "value"] after applying
            exponential smoothing. Returns an empty DataFrame if the requested
            scalar (y_axis) does not exist in this run.
        """
        ea = event_accumulator.EventAccumulator(run_path)
        ea.Reload()
        if self.metric not in ea.scalars.Keys():
            return pd.DataFrame()

        scalar_data = ea.Scalars(self.metric)
        df = pd.DataFrame(scalar_data)  # columns = [wall_time, step, value]
        df["value"] = df["value"].ewm(alpha=alpha_i).mean()
        return df

    def _collect_version_data(self,
                              alpha_i):
        """
        Iterates over each subdirectory (version) within self.exp_dir,
        checks for run subdirectories, extracts the scalar data from each run,
        and returns a structured list.

        Parameters
        ----------
        alpha_i : float
            Inverted smoothing factor passed to `_extract_and_smooth_metrics`.

        Returns
        -------
        version_data : list of dict
            A list where each dict corresponds to a version and has:
            {
                "alias": <str>            # The folder name or alias from folder_alias.txt
                "run_dataframes": <list>  # List of DataFrames (one per run) 
            }
        """
        version_data = []
        # Sort subdirectories for consistent ordering
        for version_name in sorted(os.listdir(self.log_dir)):
            version_path = os.path.join(self.log_dir, version_name)
            if not os.path.isdir(version_path):
                continue

            # Attempt to read alias from folder_alias.txt
            alias_path = os.path.join(version_path, "folder_alias.txt")
            if os.path.isfile(alias_path):
                with open(alias_path, "r") as f:
                    alias = f.read().strip()
            else:
                # Fallback: use the folder name
                alias = version_name

            run_dataframes = []
            for run_dir in sorted(os.listdir(version_path)):
                run_path = os.path.join(version_path, run_dir)
                if not os.path.isdir(run_path):
                    continue

                df = self._extract_and_smooth_metrics(run_path, alpha_i)
                if not df.empty:
                    run_dataframes.append(df)

            version_data.append({
                "alias": alias,
                "run_dataframes": run_dataframes
            })

        return version_data

    def _plot_versions(self,
                       ax,
                       version_data):
        """
        Draws the lines for each version (and its runs) onto the provided Axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes object on which to draw plots.
        version_data : list of dict
            The structure returned by `_collect_version_data`.

        Returns
        -------
        line_collections_main : list
            Each element is a list of mean line objects for a version (usually just one line).
        line_collections_runs : list
            Each element is a list of line objects for the individual runs of a version.
        legend_lines : list
            References to the mean lines (one per version) that appear in the legend.
        """
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        line_collections_main = []
        line_collections_runs = []
        legend_lines = []

        for idx, vdata in enumerate(version_data):
            alias = vdata["alias"]
            run_dataframes = vdata["run_dataframes"]

            runs_lines = []
            all_runs_values = []
            all_runs_x = []

            for df in run_dataframes:
                # Grab x and y
                min_len = min(len(df), len(df["value"]))
                x = df[self.x_axis][:min_len]
                y = df["value"][:min_len]

                # If not plotting only the mean, plot each run
                if not self.plot_mean_only:
                    (line,) = ax.plot(
                        x, y,
                        label="",  # Not in legend (only mean is in legend)
                        color=colors[idx % len(colors)],
                        alpha=0.3,
                        linewidth=1
                    )
                    runs_lines.append(line)

                all_runs_values.append(y)
                all_runs_x.append(x)

            # Keep track of run lines
            line_collections_runs.append(runs_lines if not self.plot_mean_only else [])

            # Plot the mean if there is at least one run
            if len(run_dataframes) > 0:
                version_df = pd.concat(all_runs_values, axis=1)
                mean_values = version_df.mean(axis=1)

                # Align x and y lengths
                min_len = min(len(all_runs_x[-1]), len(mean_values))
                x_mean = all_runs_x[-1][:min_len]
                y_mean = mean_values[:min_len]

                (mean_line,) = ax.plot(
                    x_mean, y_mean,
                    label=alias,
                    color=colors[idx % len(colors)],
                    alpha=1.0,
                    linewidth=2
                )
                legend_lines.append(mean_line)
                line_collections_main.append([mean_line])
            else:
                # If no runs, store empty placeholders
                line_collections_main.append([])

        return line_collections_main, line_collections_runs, legend_lines

    def get_on_legend_click(self,
                            fig,
                            legend,
                            legend_lines_labels,
                            line_collections_main,
                            line_collections_runs):
        """
        Returns a callback function to handle legend click events, toggling
        the visibility of lines belonging to the selected version.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object for redrawing after toggling.
        legend : matplotlib.legend.Legend
            The legend object containing the clickable entries.
        legend_lines_labels : list of str
            The labels (aliases) used in the legend.
        line_collections_main : list
            List of lists, each containing the main (mean) line(s) for a version.
        line_collections_runs : list
            List of lists, each containing the run lines for a version.

        Returns
        -------
        on_legend_click : function
            A nested function (closure) that can be used as a pick_event callback.
        """
        def on_legend_click(event):
            artist_label = event.artist.get_label()
            # Toggle the alpha values of the lines
            if artist_label in legend_lines_labels:
                line_index = legend_lines_labels.index(artist_label)

                main_lines = line_collections_main[line_index]
                run_lines = line_collections_runs[line_index] if line_collections_runs else []

                # Toggle alpha for the mean line
                for line in main_lines:
                    line.set_alpha(0 if line.get_alpha() > 0 else 1)

                # Toggle alpha for each run in that version
                for line in run_lines:
                    line.set_alpha(0 if line.get_alpha() > 0 else 0.3)

                fig.canvas.draw()

        return on_legend_click

    def plot(self):
        """
        Reads TensorBoard logs from each version folder, applies smoothing,
        and plots the results. Each version's runs (if not hidden) and mean
        line are plotted, with a legend entry for the mean line only.

        The legend is clickable: clicking a version's legend entry toggles
        the visibility of that version's lines.

        Steps:
        1. Invert the user alpha into alpha_i = 1 - alpha for smoothing.
        2. Collect data from all versions.
        3. Create a Matplotlib figure and axes.
        4. Plot each version's runs (optional) and mean line.
        5. Configure x/y labels, grid, and title.
        6. Create and configure the interactive legend.
        7. Display the final plot.
        """
        # Invert user alpha -> alpha_i
        alpha_i = 1 - self.alpha
        if alpha_i == 0:
            alpha_i = 1e-4

        # --- Step 1: Collect data for all versions ---
        version_data = self._collect_version_data(alpha_i)

        # --- Step 2: Create the figure/axes ---
        fig, ax = plt.subplots(figsize=(12, 6))

        # --- Step 3: Plot each version (runs + mean line) ---
        (line_collections_main,
         line_collections_runs,
         legend_lines) = self._plot_versions(ax, version_data)

        # --- Step 4: Axis labels, grid, title ---
        ax.set_xlabel(self.x_axis_label)
        ax.set_ylabel(f"{self.y_axis_label} (smoothing={self.alpha})")
        ax.grid(visible=True)
        ax.set_title(self.plot_title)

        # --- Step 5: Legend setup ---
        legend_lines_labels = [l.get_label() for l in legend_lines]
        legend = ax.legend(handles=legend_lines, fancybox=True, shadow=True)

        # Make legend entries clickable
        for legend_entry in legend.get_lines():
            legend_entry.set_picker(5)

        # --- Step 6: Connect the clickable legend handler ---
        on_legend_click = self.get_on_legend_click(
            fig, legend, legend_lines_labels,
            line_collections_main, line_collections_runs
        )
        fig.canvas.mpl_connect("pick_event", on_legend_click)

        # --- Step 7: Show the final plot ---
        plt.tight_layout()
        plt.show()


def main():
    """
    Command-line entry point that parses arguments and invokes TensorBoardPlotter.

    Usage example:
        python your_script.py \\
            --log_dir /path/to/experiments \\
            --x_axis step \\
            --metric train_loss \\
            --alpha 0.3 \\
            --x_axis_label "Training Steps" \\
            --y_axis_label "Loss Value" \\
            --plot_title "Training Loss Over Time" \\
            --plot_mean_only
    """
    parser = argparse.ArgumentParser(description="Plot metrics from TensorBoard logs.")
    parser.add_argument("--log_dir",
                        type=str,
                        help="Directory containing version subfolders.",
                        required=True)
    parser.add_argument("--metric",
                        type=str,
                        help="Name of the scalar metric to plot.",
                        required=True)
    parser.add_argument("--x_axis",
                        type=str,
                        help="Which scalar field to use as X axis. Defaults to 'step'",
                        default="step")
    parser.add_argument("--alpha",
                        type=float,
                        default=0.0,
                        help="Smoothing parameter (0=no smoothing, 1=max smoothing). Defaults to 0.")
    parser.add_argument("--x_axis_label",
                        type=str,
                        default="",
                        help="Label for the x axis. Defaults to x_axis.")
    parser.add_argument("--y_axis_label",
                        type=str,
                        default="",
                        help="Label for the y axis. Defaults to y_axis.")
    parser.add_argument("--plot_title",
                        type=str,
                        default="",
                        help="Optional title for the plot.")
    parser.add_argument("--plot_mean_only",
                        action="store_true",
                        help="If set, plot only the mean line for each version.")

    args = parser.parse_args()

    plotter = TensorBoardPlotter(
        log_dir=args.log_dir,
        x_axis=args.x_axis,
        metric=args.metric,
        alpha=args.alpha,
        x_axis_label=args.x_axis_label,
        y_axis_label=args.y_axis_label,
        plot_title=args.plot_title,
        plot_mean_only=args.plot_mean_only,
    )
    plotter.plot()


if __name__ == "__main__":
    main()
