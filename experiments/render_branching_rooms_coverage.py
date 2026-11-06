#!/usr/bin/env python3
"""Render exact Branching Rooms coverage diagnostics.

The input is ``diagnostics/environment_coverage_cells.csv`` written by the
ensemble MuZero training loop. Rendering does not require Gymnasium or
MiniGrid; the logical layout is read from the environment source.
"""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from PIL import Image, ImageDraw

try:
    from experiments.visualize_branching_rooms import (
        ENV_SOURCE,
        MUTED_TEXT_COLOR,
        PANEL_COLOR,
        TEXT_COLOR,
        Layout,
        _font,
        _load_class_literals,
        _map_base,
        _outlined_text,
    )
except ModuleNotFoundError:
    from visualize_branching_rooms import (  # type: ignore
        ENV_SOURCE,
        MUTED_TEXT_COLOR,
        PANEL_COLOR,
        TEXT_COLOR,
        Layout,
        _font,
        _load_class_literals,
        _map_base,
        _outlined_text,
    )


Position = Tuple[int, int]

BACKGROUND_COLOR = (18, 21, 25)
HEAT_LOW_COLOR = (18, 124, 119)
HEAT_HIGH_COLOR = (250, 204, 21)
CUMULATIVE_COLOR = (20, 184, 166)
NEW_COLOR = (250, 204, 21)
COMMON_COLOR = (190, 195, 201)
BASELINE_ONLY_COLOR = (230, 126, 52)
AUXILIARY_ONLY_COLOR = (20, 184, 166)


def _resolve_coverage_csv(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / 'diagnostics' / 'environment_coverage_cells.csv'
    if candidate.is_file():
        return candidate
    candidate = path / 'environment_coverage_cells.csv'
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        'Could not find environment_coverage_cells.csv under {}'.format(path)
    )


class CoverageRun:
    def __init__(self, path: Path, label: Optional[str] = None) -> None:
        self.csv_path = _resolve_coverage_csv(path)
        if self.csv_path.parent.name == 'diagnostics':
            default_label = self.csv_path.parent.parent.name
        else:
            default_label = self.csv_path.stem
        self.label = label or default_label
        self.rows_by_phase: Dict[int, List[Dict[str, int]]] = defaultdict(list)
        with self.csv_path.open(newline='') as file:
            reader = csv.DictReader(file)
            required = {
                'collection_phase',
                'x',
                'y',
                'phase_visit_count',
                'new_global_position',
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    '{} is missing columns: {}'.format(
                        self.csv_path, ', '.join(sorted(missing))
                    )
                )
            for source_row in reader:
                row = {
                    key: int(source_row[key])
                    for key in source_row
                    if source_row[key] not in (None, '')
                }
                self.rows_by_phase[row['collection_phase']].append(row)
        if not self.rows_by_phase:
            raise ValueError('{} contains no coverage rows'.format(self.csv_path))

    @property
    def phases(self) -> List[int]:
        return sorted(self.rows_by_phase)

    def phase_counts(self, phase: int) -> Counter:
        return Counter({
            (row['x'], row['y']): row['phase_visit_count']
            for row in self.rows_by_phase.get(phase, [])
        })

    def cumulative_counts(self, phase: int) -> Counter:
        counts = Counter()
        for current_phase in self.phases:
            if current_phase > phase:
                break
            counts.update(self.phase_counts(current_phase))
        return counts

    def new_positions(self, phase: int) -> Set[Position]:
        return {
            (row['x'], row['y'])
            for row in self.rows_by_phase.get(phase, [])
            if row['new_global_position']
        }


def _interpolate_color(
        start: Tuple[int, int, int],
        end: Tuple[int, int, int],
        fraction: float,
) -> Tuple[int, int, int]:
    fraction = min(max(float(fraction), 0.0), 1.0)
    return tuple(
        int(round(left + fraction * (right - left)))
        for left, right in zip(start, end)
    )


def _draw_cell_overlay(
        image: Image.Image,
        cell_colors: Mapping[Position, Tuple[int, int, int]],
        cell_size: int,
        alpha: int = 210,
) -> None:
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    inset = max(1, cell_size // 12)
    for (x, y), color in cell_colors.items():
        draw.rectangle(
            [
                x * cell_size + inset,
                y * cell_size + inset,
                (x + 1) * cell_size - inset - 1,
                (y + 1) * cell_size - inset - 1,
            ],
            fill=(*color, alpha),
        )
    image.paste(Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB'))


def _draw_room_labels(layout: Layout, image: Image.Image, cell_size: int) -> None:
    draw = ImageDraw.Draw(image)
    font = _font(max(9, cell_size - 4), bold=True)
    for room in sorted(layout.graph):
        center_x, center_y = layout.room_center(room)
        label = 'R{},{}'.format(*room)
        width, height = draw.textsize(label, font=font)
        _outlined_text(
            draw,
            (
                (center_x + 0.5) * cell_size - width / 2,
                (center_y + 0.5) * cell_size - height / 2,
            ),
            label,
            font,
            (250, 250, 250),
            (20, 22, 25),
            1,
        )


def _map_with_heat(
        layout: Layout,
        counts: Mapping[Position, int],
        cell_size: int,
) -> Image.Image:
    image = _map_base(layout, cell_size, labels=False)
    maximum = max(counts.values(), default=1)
    scale = math.log1p(maximum)
    colors = {
        position: _interpolate_color(
            HEAT_LOW_COLOR,
            HEAT_HIGH_COLOR,
            math.log1p(count) / scale if scale > 0 else 0.0,
        )
        for position, count in counts.items()
    }
    _draw_cell_overlay(image, colors, cell_size)
    _draw_room_labels(layout, image, cell_size)
    return image


def _map_with_cumulative_coverage(
        layout: Layout,
        cumulative_positions: Set[Position],
        new_positions: Set[Position],
        cell_size: int,
) -> Image.Image:
    image = _map_base(layout, cell_size, labels=False)
    colors = {
        position: NEW_COLOR if position in new_positions else CUMULATIVE_COLOR
        for position in cumulative_positions
    }
    _draw_cell_overlay(image, colors, cell_size)
    _draw_room_labels(layout, image, cell_size)
    return image


def _map_with_comparison(
        layout: Layout,
        baseline_positions: Set[Position],
        auxiliary_positions: Set[Position],
        cell_size: int,
) -> Image.Image:
    image = _map_base(layout, cell_size, labels=False)
    common = baseline_positions & auxiliary_positions
    baseline_only = baseline_positions - auxiliary_positions
    auxiliary_only = auxiliary_positions - baseline_positions
    colors = {position: COMMON_COLOR for position in common}
    colors.update({position: BASELINE_ONLY_COLOR for position in baseline_only})
    colors.update({position: AUXILIARY_ONLY_COLOR for position in auxiliary_only})
    _draw_cell_overlay(image, colors, cell_size)
    _draw_room_labels(layout, image, cell_size)
    return image


def _panel(
        map_image: Image.Image,
        title: str,
        subtitle: str,
) -> Image.Image:
    margin = 14
    header_height = 62
    panel = Image.new(
        'RGB',
        (map_image.width + 2 * margin, map_image.height + header_height + margin),
        PANEL_COLOR,
    )
    panel.paste(map_image, (margin, header_height))
    draw = ImageDraw.Draw(panel)
    draw.text((margin, 10), title, font=_font(18, True), fill=TEXT_COLOR)
    draw.text((margin, 36), subtitle, font=_font(12), fill=MUTED_TEXT_COLOR)
    return panel


def _save_panel_grid(
        panels: Sequence[Image.Image],
        title: str,
        legend: Sequence[Tuple[Tuple[int, int, int], str]],
        output_path: Path,
) -> None:
    columns = min(3, len(panels))
    rows = int(math.ceil(len(panels) / columns))
    gap = 12
    top = 74 if legend else 50
    panel_width = max(panel.width for panel in panels)
    panel_height = max(panel.height for panel in panels)
    canvas = Image.new(
        'RGB',
        (
            columns * panel_width + (columns + 1) * gap,
            top + rows * panel_height + (rows + 1) * gap,
        ),
        BACKGROUND_COLOR,
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), title, font=_font(23, True), fill=TEXT_COLOR)
    legend_x = gap
    for color, label in legend:
        draw.rectangle([legend_x, 44, legend_x + 15, 59], fill=color)
        draw.text((legend_x + 22, 42), label, font=_font(12), fill=MUTED_TEXT_COLOR)
        legend_x += 22 + len(label) * 8 + 22
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        x = gap + column * (panel_width + gap)
        y = top + gap + row * (panel_height + gap)
        canvas.paste(panel, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _safe_label(label: str) -> str:
    return ''.join(
        character if character.isalnum() or character in '-_' else '_'
        for character in label
    ).strip('_')


def render_run(
        layout: Layout,
        run: CoverageRun,
        output_dir: Path,
        cell_size: int,
) -> List[Path]:
    heat_panels = []
    cumulative_panels = []
    for phase in run.phases:
        phase_counts = run.phase_counts(phase)
        cumulative_counts = run.cumulative_counts(phase)
        new_positions = run.new_positions(phase)
        heat_panels.append(_panel(
            _map_with_heat(layout, phase_counts, cell_size),
            'Phase {} visits'.format(phase),
            '{} cells, {} recorded state samples'.format(
                len(phase_counts), sum(phase_counts.values())
            ),
        ))
        cumulative_panels.append(_panel(
            _map_with_cumulative_coverage(
                layout, set(cumulative_counts), new_positions, cell_size
            ),
            'Through phase {}'.format(phase),
            '{} cumulative cells, {} newly added'.format(
                len(cumulative_counts), len(new_positions)
            ),
        ))

    prefix = _safe_label(run.label)
    heat_path = output_dir / '{}_phase_visit_heatmaps.png'.format(prefix)
    cumulative_path = output_dir / '{}_cumulative_coverage.png'.format(prefix)
    _save_panel_grid(
        heat_panels,
        '{}: per-phase visitation'.format(run.label),
        ((HEAT_LOW_COLOR, 'fewer visits'), (HEAT_HIGH_COLOR, 'more visits')),
        heat_path,
    )
    _save_panel_grid(
        cumulative_panels,
        '{}: cumulative coverage'.format(run.label),
        ((CUMULATIVE_COLOR, 'previously covered'), (NEW_COLOR, 'new this phase')),
        cumulative_path,
    )
    return [heat_path, cumulative_path]


def render_comparison(
        layout: Layout,
        baseline: CoverageRun,
        auxiliary: CoverageRun,
        output_dir: Path,
        cell_size: int,
) -> List[Path]:
    phases = sorted(set(baseline.phases) | set(auxiliary.phases))
    phase_panels = []
    cumulative_panels = []
    for phase in phases:
        baseline_phase = set(baseline.phase_counts(phase))
        auxiliary_phase = set(auxiliary.phase_counts(phase))
        baseline_cumulative = set(baseline.cumulative_counts(phase))
        auxiliary_cumulative = set(auxiliary.cumulative_counts(phase))

        def make_panel(
                title: str,
                baseline_positions: Set[Position],
                auxiliary_positions: Set[Position],
        ) -> Image.Image:
            common = baseline_positions & auxiliary_positions
            baseline_only = baseline_positions - auxiliary_positions
            auxiliary_only = auxiliary_positions - baseline_positions
            return _panel(
                _map_with_comparison(
                    layout, baseline_positions, auxiliary_positions, cell_size
                ),
                title,
                'shared {} | baseline-only {} | auxiliary-only {}'.format(
                    len(common), len(baseline_only), len(auxiliary_only)
                ),
            )

        phase_panels.append(make_panel(
            'Phase {} visits'.format(phase), baseline_phase, auxiliary_phase
        ))
        cumulative_panels.append(make_panel(
            'Through phase {}'.format(phase),
            baseline_cumulative,
            auxiliary_cumulative,
        ))

    legend = (
        (COMMON_COLOR, 'both'),
        (BASELINE_ONLY_COLOR, 'baseline only'),
        (AUXILIARY_ONLY_COLOR, 'auxiliary only'),
    )
    phase_path = output_dir / 'baseline_vs_auxiliary_phase_visits.png'
    cumulative_path = output_dir / 'baseline_vs_auxiliary_cumulative_coverage.png'
    _save_panel_grid(
        phase_panels,
        'Baseline versus auxiliary: per-phase visitation',
        legend,
        phase_path,
    )
    _save_panel_grid(
        cumulative_panels,
        'Baseline versus auxiliary: cumulative coverage',
        legend,
        cumulative_path,
    )
    return [phase_path, cumulative_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Render exact Branching Rooms coverage from saved cell diagnostics.'
    )
    parser.add_argument('--run', type=Path)
    parser.add_argument('--baseline-run', type=Path)
    parser.add_argument('--auxiliary-run', type=Path)
    parser.add_argument('--label')
    parser.add_argument('--baseline-label', default='baseline')
    parser.add_argument('--auxiliary-label', default='auxiliary')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('video/branching_rooms_coverage'),
    )
    parser.add_argument('--cell-size', type=int, default=14)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    single_mode = args.run is not None
    comparison_mode = args.baseline_run is not None or args.auxiliary_run is not None
    if single_mode == comparison_mode:
        raise SystemExit(
            'Specify either --run, or both --baseline-run and --auxiliary-run.'
        )
    if comparison_mode and (
            args.baseline_run is None or args.auxiliary_run is None
    ):
        raise SystemExit(
            'Comparison mode requires both --baseline-run and --auxiliary-run.'
        )
    if args.cell_size < 8:
        raise SystemExit('--cell-size must be at least 8')

    layout = Layout(_load_class_literals(ENV_SOURCE))
    output_paths = []
    if single_mode:
        run = CoverageRun(args.run, args.label)
        output_paths.extend(render_run(layout, run, args.output_dir, args.cell_size))
    else:
        baseline = CoverageRun(args.baseline_run, args.baseline_label)
        auxiliary = CoverageRun(args.auxiliary_run, args.auxiliary_label)
        output_paths.extend(render_run(
            layout, baseline, args.output_dir / 'baseline', args.cell_size
        ))
        output_paths.extend(render_run(
            layout, auxiliary, args.output_dir / 'auxiliary', args.cell_size
        ))
        output_paths.extend(render_comparison(
            layout, baseline, auxiliary, args.output_dir, args.cell_size
        ))

    for output_path in output_paths:
        print(output_path)


if __name__ == '__main__':
    main()
