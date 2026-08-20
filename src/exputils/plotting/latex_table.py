from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass
class LatexTable:
    # tuples of (column name, number of value columns to span), per column level
    column_headers: list[str] | list[list[tuple[str, int]]] | None = None
    # display names for each column level, shown right-aligned in the leftmost cell of each header row
    column_level_names: list[str] | None = None
    # only allow one row header for now
    row_header: str | None = None
    df_center: pd.DataFrame | None = None
    df_lower: pd.DataFrame | None = None
    df_upper: pd.DataFrame | None = None
    df_std: pd.DataFrame | None = None
    df_p_values: pd.DataFrame | None = None
    df_is_bold: pd.DataFrame | None = None  # boolean df, same shape as df_center

    def __post_init__(self) -> None:
        has_uncertainty = self.df_lower is not None or self.df_upper is not None or self.df_std is not None
        if has_uncertainty and self.df_p_values is not None:
            raise ValueError("Cannot set both uncertainty (df_lower/df_upper/df_std) and df_p_values on the same table.")

    # which column level to insert vertical lines at group boundaries
    split_column_level: int | None = None

    decimal_precision: int = 0

    # optional LaTeX font-size commands (e.g. r"\large") for labels
    # column_label_fontsize: str | None = None  # applied to all column header cells
    # row_label_fontsize: str | None = None  # applied to row-label column and row heading

    # how to display confidence intervals alongside the center value
    # ci_display: Literal["parenthetical", "stacked"] = "parenthetical"


def _derive_column_headers(columns: pd.Index) -> list[tuple[str, int]] | list[list[tuple[str, int]]]:
    """Derive column header structure from a DataFrame's column index at render time."""
    if columns.nlevels == 1:
        return [(str(col), 1) for col in columns]
    elif columns.nlevels == 2:
        top_level_seen: set = set()
        last_top_level = columns[0][0]
        headers_across_levels: list[list[tuple[str, int]]] = [[], []]
        last_switch = 0
        for i, col in enumerate(columns):
            if col[0] != last_top_level:
                if col[0] in top_level_seen:
                    raise ValueError(
                        "Column levels are not properly grouped, found a switch back to a previously seen top level column name"
                    )
                top_level_seen.add(col[0])
                headers_across_levels[0].append((str(last_top_level), i - last_switch))
                last_top_level = col[0]
                last_switch = i
            elif i == len(columns) - 1:
                headers_across_levels[0].append((str(col[0]), i - last_switch + 1))
            headers_across_levels[1].append((str(col[1]), 1))
        return headers_across_levels
    else:
        raise NotImplementedError("DataFrames with more than 2 column levels are not supported")


def _identify_best_values(df: pd.DataFrame, split_column_level: int | None = None, maximize: bool = True) -> pd.DataFrame:
    """
    Identifies the best values in the df_center dataframe, either by maximizing or minimizing, and returns a boolean dataframe of the same shape as df_center, where True indicates that the value is one of the best values in its row.
    If split_column_level is given, the best values are identified separately for each group of columns that share the same value in the given column level.
    """
    if split_column_level == 0:
        best_values = pd.DataFrame(False, index=df.index, columns=df.columns)
        for column in df.columns.get_level_values(0).unique():
            sub_df = df[column]
            if maximize:
                best_values[column] = sub_df.eq(sub_df.max(axis=1), axis=0)
            else:
                best_values[column] = sub_df.eq(sub_df.min(axis=1), axis=0)
    else:
        if maximize:
            best_values = df.eq(df.max(axis=1), axis=0)
        else:
            best_values = df.eq(df.min(axis=1), axis=0)
    return best_values


def _populate_latex_table_from_df(
    df_center: pd.DataFrame,
    df_lower: pd.DataFrame | None = None,
    df_upper: pd.DataFrame | None = None,
    df_std: pd.DataFrame | None = None,
    df_p_values: pd.DataFrame | None = None,
    highlight_best: Literal["min", "max"] | None = None,
    split_column_level: int | None = None,
    decimal_precision: int = 0,
) -> LatexTable:
    # make sure all dfs have the same index and columns
    index = df_center.index
    columns = df_center.columns
    for df in [df_lower, df_upper, df_std, df_p_values]:
        if df is not None:
            assert df.index.equals(index), "All dataframes must have the same index"
            assert df.columns.equals(columns), "All dataframes must have the same columns"

    table = LatexTable(
        df_center=df_center,
        df_lower=df_lower,
        df_upper=df_upper,
        df_std=df_std,
        df_p_values=df_p_values,
        split_column_level=split_column_level,
        decimal_precision=decimal_precision,
    )

    if highlight_best is not None:
        table.df_is_bold = _identify_best_values(df_center, split_column_level=table.split_column_level, maximize=highlight_best == "max")

    # column_headers, column_level_names, and row_header are all derived from
    # df_center at render time so that mutations after table creation are reflected.

    return table


def latex_table_to_string(
    table: LatexTable,
    ci_display: Literal["parenthetical", "stacked"] = "parenthetical",
    vertical_lines: bool = True,
    column_label_fontsize: str | None = None,  # applied to all column header cells
    row_label_fontsize: str | None = None,  # applied to row-label column and row heading
    uncertainty_decimal_precision: int | None = None,  # falls back to table.decimal_precision when None
    p_value_precision: int = 3,  # decimal precision for p-values (independent of table.decimal_precision)
    p_value_threshold: float | None = None,  # if set, values below this are shown as "p<threshold" instead
) -> str:
    """Render a LatexTable to a LaTeX tabular string.

    All column header rows are followed by a single \\midrule (no rule between
    individual header levels). Top-level column titles are centered and bold;
    descriptor (sub-column) headers are right-aligned and bold. Column level
    names appear right-aligned in the leftmost header cell. The row heading is
    bold and occupies its own row above the data rows. Row labels are
    left-aligned.

    Uncertainty display (requires xcolor package):
      - If df_lower/df_upper are set, confidence interval bounds are shown
        stacked (upper above lower) in \\footnotesize gray after the center value.
      - If only df_std is set, ±std is shown in \\footnotesize gray.
      - CI takes precedence over std when both are provided.
      - If df_p_values is set, p-values are appended in \\footnotesize gray as
        "p=X.XXX" (or "p<threshold" when p_value_threshold is given and the
        value is below it). P-values are shown independently of CI/std.
      - Bold (df_is_bold) applies only to the center value, not the uncertainty.

    Vertical line placement (when vertical_lines=True):
      - split_column_level is None  → | between every column (no closing border)
      - split_column_level given    → | only where the top-level group changes

    Font sizes (column_label_fontsize / row_label_fontsize):
      LaTeX size commands, e.g. r"\\large". Applied to header cells and row
      labels respectively.
    """
    df = table.df_center
    assert df is not None, "df_center must be set"

    n_val_cols = len(df.columns)
    prec = table.decimal_precision
    uprec = uncertainty_decimal_precision if uncertainty_decimal_precision is not None else prec

    # ------------------------------------------------------------------ #
    # Determine vertical-line positions                                    #
    # vline_positions: set of int i where | appears before value column i  #
    #   i=0  → between row-label column and first value column            #
    #   i=k  → between value column k-1 and k                             #
    # The rightmost column never gets a closing border.                   #
    # ------------------------------------------------------------------ #
    vline_positions: set[int] = set()

    if vertical_lines:
        vline_positions.add(0)  # always separate row labels from values
        if table.split_column_level is None:
            for i in range(1, n_val_cols):
                vline_positions.add(i)
        elif isinstance(df.columns, pd.MultiIndex):
            top_level = df.columns.get_level_values(table.split_column_level)
            for i in range(1, n_val_cols):
                if top_level[i] != top_level[i - 1]:
                    vline_positions.add(i)

    # ------------------------------------------------------------------ #
    # Column spec                                                          #
    # ------------------------------------------------------------------ #
    col_spec = "l"
    if 0 in vline_positions:
        col_spec += "|"
    for i in range(n_val_cols):
        col_spec += "r"
        if (i + 1) in vline_positions:
            col_spec += "|"

    lines: list[str] = []
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # ------------------------------------------------------------------ #
    # Column header rows — all levels, then a single \midrule             #
    # ------------------------------------------------------------------ #
    # Derive row_header and column_level_names from the live df so that
    # mutations to index/column names after table creation are reflected.
    if table.row_header:
        row_header: str | None = table.row_header
    elif isinstance(df.index, pd.MultiIndex):
        row_header = str(df.index.names[-1]) or None
    else:
        row_header = str(df.index.name) or None

    live_col_level_names = table.column_level_names or [str(n) for n in df.columns.names]

    header_rows = _build_header_rows(
        table,
        vline_positions,
        column_label_fontsize=column_label_fontsize,
        row_header=row_header,
        column_level_names=live_col_level_names,
    )
    for header_row in header_rows:
        lines.append(header_row + " \\\\")
    if header_rows:
        lines.append("\\midrule")

    # ------------------------------------------------------------------ #
    # Data rows                                                            #
    # ------------------------------------------------------------------ #
    has_ci = table.df_lower is not None and table.df_upper is not None
    has_std = table.df_std is not None
    has_pval = table.df_p_values is not None

    # Iterate in the exact order of df.index / df.columns; use label-based
    # .loc for all auxiliary DataFrames so their internal ordering doesn't matter.
    multi_row_index = isinstance(df.index, pd.MultiIndex)
    current_top_row_level: object = object()  # sentinel — never equals a real value
    top_index_name: str | None = (str(df.index.names[0]) or None) if multi_row_index else None
    first_group = True

    for row_label in df.index:
        # Multi-level rows: emit a centered group-header row on top-level change,
        # surrounded by \hline above (except before the very first group) and below.
        if multi_row_index:
            top_val, sub_val = row_label[0], row_label[1]  # type: ignore[index]
            if top_val != current_top_row_level:
                current_top_row_level = top_val
                if not first_group:
                    lines.append("\\hline")
                first_group = False
                label = f"{top_index_name}: {top_val}" if top_index_name else str(top_val)
                group_heading = f"\\multicolumn{{{n_val_cols + 1}}}{{c}}{{\\textbf{{{label}}}}}"
                if row_label_fontsize:
                    group_heading = f"{{{row_label_fontsize} {group_heading}}}"
                lines.append(group_heading + " \\\\")
                lines.append("\\hline")
            row_label_display: object = sub_val
        else:
            row_label_display = row_label

        row_label_str = str(row_label_display)
        if row_label_fontsize:
            row_label_str = f"{{{row_label_fontsize} {row_label_str}}}"
        cells = [row_label_str]

        for col_label in df.columns:
            center = df.loc[row_label, col_label]
            center_str = f"{center:.{prec}f}"

            # Bold wraps only the center value
            is_bold = table.df_is_bold is not None and bool(table.df_is_bold.loc[row_label, col_label])
            if is_bold:
                center_str = f"\\textbf{{{center_str}}}"

            # Uncertainty annotation — greyed out, smaller font, after center.
            uncertainty = ""
            if has_ci:
                lower = table.df_lower.loc[row_label, col_label]  # type: ignore[union-attr]
                upper = table.df_upper.loc[row_label, col_label]  # type: ignore[union-attr]
                if ci_display == "stacked":
                    # Two \scriptsize lines compressed to roughly normal line
                    # height via reduced \arraystretch.  The nested tabular[c]
                    # vertically centers the stack on the surrounding baseline.
                    inner = (
                        f"{{\\renewcommand{{\\arraystretch}}{{0.75}}"
                        f"\\begin{{tabular}}[c]{{@{{}}r@{{}}}}"
                        f"{upper:.{uprec}f} \\\\ {lower:.{uprec}f}"
                        f"\\end{{tabular}}}}"
                    )
                    uncertainty = f" {{\\scriptsize\\textcolor{{gray}}{{{inner}}}}}"
                else:
                    uncertainty = f" {{\\footnotesize\\textcolor{{gray}}{{({lower:.{uprec}f}, {upper:.{uprec}f})}}}}"
            elif has_std:
                std = table.df_std.loc[row_label, col_label]  # type: ignore[union-attr]
                uncertainty = f" {{\\footnotesize\\textcolor{{gray}}{{$\\pm${std:.{uprec}f}}}}}"

            p_value_str = ""
            if has_pval:
                p_val = table.df_p_values.loc[row_label, col_label]  # type: ignore[union-attr]
                pprec = p_value_precision
                if p_value_threshold is not None and float(p_val) < p_value_threshold:  # type: ignore[union-attr]
                    p_annotation = f"p<{p_value_threshold:.{pprec}f}"
                else:
                    p_annotation = f"p={p_val:.{pprec}f}"
                p_value_str = f" {{\\footnotesize\\textcolor{{gray}}{{{p_annotation}}}}}"

            cells.append(center_str + uncertainty + p_value_str)

        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _build_header_rows(
    table: LatexTable,
    vline_positions: set[int],
    column_label_fontsize: str | None = None,
    row_header: str | None = None,
    column_level_names: list[str] | None = None,
) -> list[str]:
    """Return a list of header row strings (without trailing \\\\).

    Layout rules:
    - Multi-level columns (2 levels): the first header row uses
      column_level_names[0] as the leftmost descriptor; the last header row
      uses row_header as the leftmost descriptor (column_level_names[-1] is
      dropped — it is superseded by the row-label column header).
    - Single-level columns: if row_header is present, a synthetic top row is
      prepended that spans all data columns with column_level_names[0] as a
      centered bold label, and the column-names row uses row_header as its
      leftmost descriptor.  If row_header is absent, column_level_names[0] is
      used as the descriptor of the sole header row.
    """
    headers = table.column_headers or _derive_column_headers(table.df_center.columns)  # type: ignore[union-attr]
    fontsize = column_label_fontsize
    if not headers:
        return []

    def _sized(text: str) -> str:
        return f"{{{fontsize} {text}}}" if fontsize else text

    if isinstance(headers[0], list):
        # Multi-level columns.
        # For every level except the last: descriptor (leftmost) = column_level_names[level_idx],
        #   column values spanning their columns.
        # For the last level: first emit a synthetic spanning row that shows
        #   column_level_names[-1] across all data columns (mirroring the
        #   single-level pattern), then emit the leaf-names row with row_header
        #   as its leftmost descriptor.
        multi: list[list[tuple[str, int]]] = headers  # type: ignore[assignment]
        rows = []
        last_idx = len(multi) - 1
        n_data_cols = sum(span for _, span in multi[-1])

        for level_idx, level_headers in enumerate(multi):
            is_last = level_idx == last_idx
            if is_last:
                col_level_label = column_level_names[last_idx] if column_level_names and last_idx < len(column_level_names) else ""
                if col_level_label:
                    span_cell = _sized(f"\\multicolumn{{{n_data_cols}}}{{c}}{{\\textbf{{{col_level_label}}}}}")
                    rows.append(f" & {span_cell}")
                rows.append(
                    _build_header_row(
                        level_headers, vline_positions, centered=True, descriptor=row_header, descriptor_is_header=True, fontsize=fontsize
                    )
                )
            else:
                descriptor = column_level_names[level_idx] if column_level_names and level_idx < len(column_level_names) else None
                rows.append(_build_header_row(level_headers, vline_positions, centered=True, descriptor=descriptor, fontsize=fontsize))
        return rows

    # Single-level columns
    single: list[str | tuple[str, int]] = headers  # type: ignore[assignment]
    items: list[tuple[str, int]] = [
        (str(h), 1) if isinstance(h, str) else h  # type: ignore[misc]
        for h in single
    ]
    n_data_cols = sum(span for _, span in items)

    if row_header:
        # Prepend a row that spans the data columns with the column level name,
        # leaving the leftmost (row-label) cell empty.
        col_level_label = column_level_names[0] if column_level_names else ""
        span_cell = _sized(f"\\multicolumn{{{n_data_cols}}}{{c}}{{\\textbf{{{col_level_label}}}}}")
        top_row = f" & {span_cell}"
        col_names_row = _build_header_row(
            items, vline_positions, centered=True, descriptor=row_header, descriptor_is_header=True, fontsize=fontsize
        )
        return [top_row, col_names_row]

    # No row_header: use column level name as the sole descriptor
    descriptor = column_level_names[0] if column_level_names else None
    return [_build_header_row(items, vline_positions, centered=True, descriptor=descriptor, fontsize=fontsize)]


def _build_header_row(
    headers: list[tuple[str, int]],
    vline_positions: set[int],
    centered: bool,
    descriptor: str | None = None,
    descriptor_is_header: bool = False,
    fontsize: str | None = None,
) -> str:
    """Build one header row string from a list of (name, span) tuples.

    centered=True  → top-level: \\multicolumn with 'c' alignment, bold.
    centered=False → descriptor row: 'r' alignment (\\multicolumn only when
                     span > 1), bold.
    descriptor     → if given, placed bold in the leftmost cell (the
                     row-label column); otherwise that cell is empty.
    descriptor_is_header → when True the descriptor cell is centred with no
                     trailing colon (used for row_header, which is a proper
                     column header).  When False (default) it is right-aligned
                     with a trailing colon (used for column level names).
    fontsize       → optional LaTeX size command applied to all cells in the row.
    """

    def _sized(text: str) -> str:
        return f"{{{fontsize} {text}}}" if fontsize else text

    if descriptor:
        if descriptor_is_header:
            first_cell = _sized(f"\\multicolumn{{1}}{{c|}}{{\\textbf{{{descriptor}}}}}")
        else:
            first_cell = _sized(f"\\multicolumn{{1}}{{r|}}{{\\textbf{{{descriptor}}}:}}")
    else:
        first_cell = ""

    cells = [first_cell]
    col_pos = 0

    for name, span in headers:
        has_right_vline = (col_pos + span) in vline_positions
        border = "|" if has_right_vline else ""
        bold_name = _sized(f"\\textbf{{{name}}}")

        if span > 1 or centered:
            align_char = "c" if centered else "r"
            cells.append(f"\\multicolumn{{{span}}}{{{align_char}{border}}}{{{bold_name}}}")
        else:
            # span == 1, right-aligned: column spec handles alignment/vlines
            cells.append(bold_name)

        col_pos += span

    return " & ".join(cells)
