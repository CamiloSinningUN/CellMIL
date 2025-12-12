"""
pytest configuration and fixtures for the cellmil project.
"""

import pytest
from _pytest.fixtures import FixtureRequest
from typing import Any
from pathlib import Path


def pytest_html_report_title(report: Any):
    """Customize the HTML report title"""
    report.title = "CellMIL Test Report with Graph Visualizations"


def pytest_html_results_table_header(cells: list[str]):
    """Add custom columns to the results table"""
    cells.insert(2, '<th class="sortable" col="visualization">Plots</th>')


def pytest_html_results_table_row(report: Any, cells: list[str]):
    """Add visualization content to each test result row"""
    # Look for any plot files related to this test
    plot_files = []

    # Find plot files in test_reports directory
    test_reports_dir = Path("test_reports")
    if test_reports_dir.exists():
        # Look for plot files that might be related to this test
        all_plots = list(test_reports_dir.glob("plot_*.png"))

        # If this is a graph creator test, include the relevant plots
        if "graph_creator" in report.nodeid:
            # Include all graph visualization plots for graph creator tests
            plot_files = all_plots

    # Create HTML content for plots
    if plot_files:
        plot_html = '<div style="display: flex; flex-wrap: wrap;">'
        for plot_file in plot_files:
            # Get relative path for the HTML report
            rel_path = plot_file.name
            plot_html += f'''
            <div style="margin: 5px;">
                <img src="{rel_path}" style="max-width: 200px; max-height: 150px; cursor: pointer;" 
                     onclick="window.open('{rel_path}', '_blank')" 
                     title="Click to view full size"/>
                <br><small>{rel_path}</small>
            </div>
            '''
        plot_html += "</div>"
        cells.insert(2, f"<td>{plot_html}</td>")
    else:
        cells.insert(2, "<td>No plots</td>")


@pytest.fixture(scope="session", autouse=True)
def setup_test_reports_dir():
    """Ensure test_reports directory exists and is clean"""
    test_reports_dir = Path("test_reports")
    test_reports_dir.mkdir(exist_ok=True)

    # Clean up old plot files at the start of the session
    for plot_file in test_reports_dir.glob("plot_*.png"):
        try:
            plot_file.unlink()
        except Exception:
            pass  # Ignore errors if file is in use

    yield test_reports_dir


@pytest.fixture(autouse=True)
def add_plots_to_html_report(request: FixtureRequest):
    """Automatically add any generated plots to the HTML report"""
    # This fixture runs for every test
    yield

    # After test completes, check if any new plots were created
    if hasattr(request.config, "_html") and hasattr(request.node, "rep_call"):  # type: ignore
        test_reports_dir = Path("test_reports")
        if test_reports_dir.exists():
            # Look for plot files created during this test
            plot_files = list(test_reports_dir.glob("plot_*.png"))

            # Store plot info in the test node for later use
            if not hasattr(request.node, "plot_files"):  # type: ignore
                request.node.plot_files = []  # type: ignore
            request.node.plot_files.extend([p.name for p in plot_files])  # type: ignore
