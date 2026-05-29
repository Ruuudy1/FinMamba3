"""Build validation-loss history and plots for local-search trials."""
# region imports
import argparse
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
# endregion


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRIAL_ROOT = REPO_ROOT / "runs" / "local_search"
METRIC_NAME = "Val/yes_log_loss"
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
NEW_BEST_RE = re.compile(rf"new best val_loss=({FLOAT_PATTERN}) at step (\d+)")
FINAL_BEST_RE = re.compile(rf"best val_loss=({FLOAT_PATTERN}) at step (\d+)")
STEP_RE = re.compile(r"\|\s*(\d+)/\d+\s*\[")
VAL_LOSS_RE = re.compile(rf"val_loss=({FLOAT_PATTERN})")
BEST_RE = re.compile(rf"best=({FLOAT_PATTERN})")
WIDTH = 1200
HEIGHT = 720
MARGIN_LEFT = 86
MARGIN_RIGHT = 250
MARGIN_TOP = 72
MARGIN_BOTTOM = 92
COLORS = [
    (34, 105, 209),
    (219, 87, 54),
    (33, 150, 83),
    (151, 83, 184),
    (226, 147, 37),
    (0, 137, 145),
    (196, 65, 118),
    (93, 101, 117),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-root", type=Path, default=TRIAL_ROOT)
    parser.add_argument("--history", type=Path, default=TRIAL_ROOT / "val_loss_history.jsonl")
    parser.add_argument("--png", type=Path, default=TRIAL_ROOT / "val_loss.png")
    parser.add_argument("--html", type=Path, default=TRIAL_ROOT / "val_loss.html")
    return parser.parse_args()


def trial_sort_key(trial_dir):
    number_match = re.match(r"t(\d+)_", trial_dir.name)
    if number_match:
        return int(number_match.group(1)), trial_dir.name
    return 999999, trial_dir.name


def finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return parsed


def add_record(records_by_key, trial_id, step, val_loss, best_so_far, source):
    if val_loss is None or step is None:
        return
    if best_so_far is None or not math.isfinite(float(best_so_far)):
        best_so_far = val_loss
    key = (trial_id, int(step))
    existing = records_by_key.get(key)
    record = {
        "trial_id": trial_id,
        "step": int(step),
        "metric": METRIC_NAME,
        "val_loss": float(val_loss),
        "best_so_far": float(best_so_far),
        "source": source,
    }
    if existing is None:
        records_by_key[key] = record
        return
    source_rank = {"stdout_new_best": 0, "stdout_progress": 1, "stdout_final_best": 2, "wandb_history": 3, "wandb_binary": 4, "metrics_final": 5}
    if source_rank.get(source, 9) < source_rank.get(existing["source"], 9):
        records_by_key[key] = record


def parse_stdout_records(trial_dir):
    records_by_key = {}
    stdout_path = trial_dir / "stdout.log"
    best_so_far = math.inf
    last_step_seen = None
    pending_validation_step = None
    if not stdout_path.exists():
        return []
    for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines():
        step_match = STEP_RE.search(line)
        if step_match:
            last_step_seen = int(step_match.group(1))
        if "validation top feature MSE:" in line:
            pending_validation_step = last_step_seen
        new_best_match = NEW_BEST_RE.search(line)
        if new_best_match:
            val_loss = finite_float(new_best_match.group(1))
            if val_loss is None:
                continue
            step = int(new_best_match.group(2))
            best_so_far = min(best_so_far, val_loss)
            add_record(records_by_key, trial_dir.name, step, val_loss, best_so_far, "stdout_new_best")
            pending_validation_step = step
        final_best_match = FINAL_BEST_RE.search(line)
        if final_best_match and new_best_match is None:
            val_loss = finite_float(final_best_match.group(1))
            if val_loss is None:
                continue
            step = int(final_best_match.group(2))
            best_so_far = min(best_so_far, val_loss)
            add_record(records_by_key, trial_dir.name, step, val_loss, best_so_far, "stdout_final_best")
        if "pretrain:" not in line or "val_loss=" not in line:
            continue
        val_loss_match = VAL_LOSS_RE.search(line)
        best_match = BEST_RE.search(line)
        if step_match is None or val_loss_match is None:
            continue
        step = int(step_match.group(1))
        val_loss = finite_float(val_loss_match.group(1))
        if val_loss is None:
            continue
        if pending_validation_step is None:
            continue
        if step not in (pending_validation_step, pending_validation_step + 1):
            continue
        if best_match:
            best_candidate = finite_float(best_match.group(1))
            if best_candidate is not None:
                best_so_far = min(best_so_far, best_candidate)
        best_value = min(best_so_far, val_loss)
        add_record(records_by_key, trial_dir.name, pending_validation_step, val_loss, best_value, "stdout_progress")
        pending_validation_step = None
    return sorted(records_by_key.values(), key=lambda row: row["step"])


def parse_wandb_records(trial_dir):
    records_by_key = {}
    wandb_dir = trial_dir / "wandb"
    history_paths = sorted(wandb_dir.glob("offline-run-*/files/wandb-history.jsonl"))
    history_paths += sorted((wandb_dir / "wandb").glob("offline-run-*/files/wandb-history.jsonl"))
    best_so_far = math.inf
    for history_path in history_paths:
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if METRIC_NAME not in row or row[METRIC_NAME] is None:
                continue
            step = row.get("_step", row.get("trainer/global_step", 0))
            val_loss = finite_float(row[METRIC_NAME])
            if val_loss is None:
                continue
            best_so_far = min(best_so_far, val_loss)
            add_record(records_by_key, trial_dir.name, step, val_loss, best_so_far, "wandb_history")
    return sorted(records_by_key.values(), key=lambda row: row["step"])


def offline_run_wandb_candidates(wandb_dir):
    direct_candidates = sorted(wandb_dir.glob("offline-run-*/run-*.wandb"))
    nested_candidates = sorted((wandb_dir / "wandb").glob("offline-run-*/run-*.wandb"))
    return sorted(direct_candidates + nested_candidates)


def wandb_record_key(record_item):
    if record_item.nested_key:
        return "/".join(record_item.nested_key)
    return record_item.key


def parse_wandb_value(value_json):
    value = json.loads(value_json)
    if value is None:
        return None
    return value


def parse_wandb_binary_records(trial_dir):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
    records_by_key = {}
    best_so_far = math.inf
    for wandb_path in offline_run_wandb_candidates(trial_dir / "wandb"):
        store = DataStore()
        store.open_for_scan(str(wandb_path))
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "history":
                continue
            step = int(record.history.step.num)
            val_loss = None
            for item in record.history.item:
                key = wandb_record_key(item)
                parsed_value = parse_wandb_value(item.value_json)
                if key == "_step" and parsed_value is not None:
                    step = int(parsed_value)
                if key == METRIC_NAME:
                    val_loss = parsed_value
            if val_loss is not None:
                best_so_far = min(best_so_far, float(val_loss))
                add_record(records_by_key, trial_dir.name, step, val_loss, best_so_far, "wandb_binary")
        store.close()
    return sorted(records_by_key.values(), key=lambda row: row["step"])


def parse_metrics_final_record(trial_dir):
    metrics_path = trial_dir / "metrics_final.json"
    if not metrics_path.exists():
        return []
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    step = metrics.get("best_step_log_loss")
    val_loss = metrics.get("best_val_yes_log_loss")
    if step is None or val_loss is None:
        return []
    return [{
        "trial_id": trial_dir.name,
        "step": int(step),
        "metric": METRIC_NAME,
        "val_loss": float(val_loss),
        "best_so_far": float(val_loss),
        "source": "metrics_final",
    }]


def collect_records(trial_root):
    records_by_key = {}
    trial_dirs = [path for path in trial_root.iterdir() if path.is_dir() and path.name.startswith("t")]
    for trial_dir in sorted(trial_dirs, key=trial_sort_key):
        for record in parse_stdout_records(trial_dir):
            add_record(records_by_key, record["trial_id"], record["step"], record["val_loss"], record["best_so_far"], record["source"])
        for record in parse_wandb_records(trial_dir):
            add_record(records_by_key, record["trial_id"], record["step"], record["val_loss"], record["best_so_far"], record["source"])
        for record in parse_wandb_binary_records(trial_dir):
            add_record(records_by_key, record["trial_id"], record["step"], record["val_loss"], record["best_so_far"], record["source"])
        for record in parse_metrics_final_record(trial_dir):
            add_record(records_by_key, record["trial_id"], record["step"], record["val_loss"], record["best_so_far"], record["source"])
    return sorted(records_by_key.values(), key=lambda row: (trial_sort_key(Path(row["trial_id"])), row["step"]))


def write_history(records, history_path):
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as history_file:
        for record in records:
            history_file.write(json.dumps(record, sort_keys=True) + "\n")


def group_records(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["trial_id"]].append(record)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["step"])
    return dict(sorted(grouped.items(), key=lambda item: trial_sort_key(Path(item[0]))))


def draw_text(draw, xy, text, fill, font, anchor=None):
    if anchor:
        draw.text(xy, text, fill=fill, font=font, anchor=anchor)
        return
    draw.text(xy, text, fill=fill, font=font)


def scale_value(value, low, high, pixel_low, pixel_high):
    if high == low:
        return (pixel_low + pixel_high) / 2
    return pixel_low + (value - low) * (pixel_high - pixel_low) / (high - low)


def padded_range(values):
    low = min(values)
    high = max(values)
    if low == high:
        pad = max(abs(low) * 0.05, 0.05)
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def write_empty_png(path, message):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw_text(draw, (WIDTH // 2, HEIGHT // 2), message, (68, 75, 86), font, "mm")
    image.save(path)


def write_png(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        write_empty_png(path, "No validation loss records found.")
        return
    grouped = group_records(records)
    all_steps = [record["step"] for record in records]
    all_losses = [record["val_loss"] for record in records]
    x_min, x_max = padded_range(all_steps)
    y_min, y_max = padded_range(all_losses)
    plot_left = MARGIN_LEFT
    plot_right = WIDTH - MARGIN_RIGHT
    plot_top = MARGIN_TOP
    plot_bottom = HEIGHT - MARGIN_BOTTOM
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default(size=24)
    draw_text(draw, (MARGIN_LEFT, 28), f"{METRIC_NAME} by local-search trial", (23, 32, 42), title_font)
    for index in range(6):
        y_value = y_min + (y_max - y_min) * index / 5
        y = scale_value(y_value, y_min, y_max, plot_bottom, plot_top)
        draw.line((plot_left, y, plot_right, y), fill=(225, 230, 238), width=1)
        draw_text(draw, (plot_left - 10, y), f"{y_value:.4f}", (82, 91, 105), font, "rm")
    for index in range(6):
        x_value = x_min + (x_max - x_min) * index / 5
        x = scale_value(x_value, x_min, x_max, plot_left, plot_right)
        draw.line((x, plot_top, x, plot_bottom), fill=(236, 239, 244), width=1)
        draw_text(draw, (x, plot_bottom + 16), f"{int(round(x_value))}", (82, 91, 105), font, "mt")
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(33, 43, 54), width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(33, 43, 54), width=2)
    draw_text(draw, ((plot_left + plot_right) // 2, HEIGHT - 38), "step", (33, 43, 54), font, "mm")
    draw_text(draw, (30, (plot_top + plot_bottom) // 2), "val_loss", (33, 43, 54), font, "mm")
    for trial_index, (trial_id, rows) in enumerate(grouped.items()):
        color = COLORS[trial_index % len(COLORS)]
        points = []
        for row in rows:
            x = scale_value(row["step"], x_min, x_max, plot_left, plot_right)
            y = scale_value(row["val_loss"], y_min, y_max, plot_bottom, plot_top)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for point in points:
            x, y = point
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white", width=1)
        legend_y = MARGIN_TOP + trial_index * 24
        draw.line((WIDTH - MARGIN_RIGHT + 28, legend_y + 7, WIDTH - MARGIN_RIGHT + 56, legend_y + 7), fill=color, width=3)
        draw.ellipse((WIDTH - MARGIN_RIGHT + 38, legend_y + 3, WIDTH - MARGIN_RIGHT + 46, legend_y + 11), fill=color)
        best = min(row["val_loss"] for row in rows)
        draw_text(draw, (WIDTH - MARGIN_RIGHT + 66, legend_y), f"{trial_id}  best {best:.4f}", (33, 43, 54), font)
    image.save(path)


def svg_point(record, x_min, x_max, y_min, y_max):
    x = scale_value(record["step"], x_min, x_max, MARGIN_LEFT, WIDTH - MARGIN_RIGHT)
    y = scale_value(record["val_loss"], y_min, y_max, HEIGHT - MARGIN_BOTTOM, MARGIN_TOP)
    return x, y


def write_html(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = group_records(records)
    all_steps = [record["step"] for record in records] or [0, 1]
    all_losses = [record["val_loss"] for record in records] or [0, 1]
    x_min, x_max = padded_range(all_steps)
    y_min, y_max = padded_range(all_losses)
    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<title>Validation Loss</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#1f2933}svg{max-width:100%;height:auto}table{border-collapse:collapse;margin-top:20px}td,th{border:1px solid #d8dee8;padding:6px 9px;text-align:right}td:first-child,th:first-child{text-align:left}</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(METRIC_NAME)} by local-search trial</h1>",
        f"<svg viewBox=\"0 0 {WIDTH} {HEIGHT}\" role=\"img\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
        f"<line x1=\"{MARGIN_LEFT}\" y1=\"{HEIGHT - MARGIN_BOTTOM}\" x2=\"{WIDTH - MARGIN_RIGHT}\" y2=\"{HEIGHT - MARGIN_BOTTOM}\" stroke=\"#212b36\" stroke-width=\"2\"/>",
        f"<line x1=\"{MARGIN_LEFT}\" y1=\"{MARGIN_TOP}\" x2=\"{MARGIN_LEFT}\" y2=\"{HEIGHT - MARGIN_BOTTOM}\" stroke=\"#212b36\" stroke-width=\"2\"/>",
    ]
    for trial_index, (trial_id, rows) in enumerate(grouped.items()):
        color = COLORS[trial_index % len(COLORS)]
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        point_text = " ".join(f"{svg_point(row, x_min, x_max, y_min, y_max)[0]:.1f},{svg_point(row, x_min, x_max, y_min, y_max)[1]:.1f}" for row in rows)
        lines.append(f"<polyline points=\"{point_text}\" fill=\"none\" stroke=\"{hex_color}\" stroke-width=\"4\"/>")
        for row in rows:
            x, y = svg_point(row, x_min, x_max, y_min, y_max)
            title = html.escape(f"{trial_id} step {row['step']} {METRIC_NAME}={row['val_loss']:.6f}")
            lines.append(f"<circle cx=\"{x:.1f}\" cy=\"{y:.1f}\" r=\"6\" fill=\"{hex_color}\"><title>{title}</title></circle>")
    lines.extend([
        "</svg>",
        "<table>",
        "<thead><tr><th>trial_id</th><th>step</th><th>val_loss</th><th>best_so_far</th><th>source</th></tr></thead>",
        "<tbody>",
    ])
    for row in records:
        lines.append(f"<tr><td>{html.escape(row['trial_id'])}</td><td>{row['step']}</td><td>{row['val_loss']:.6f}</td><td>{row['best_so_far']:.6f}</td><td>{html.escape(row['source'])}</td></tr>")
    lines.extend(["</tbody>", "</table>", "</body>", "</html>"])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    records = collect_records(args.trial_root)
    write_history(records, args.history)
    write_png(records, args.png)
    write_html(records, args.html)
    print(f"wrote {len(records)} validation records")
    print(f"  history: {args.history}")
    print(f"  png:     {args.png}")
    print(f"  html:    {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
