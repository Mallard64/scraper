from __future__ import annotations

import argparse
import time
from collections import defaultdict

import cv2
import numpy as np
import pyautogui


def screenshot_bgr() -> np.ndarray:
    shot = pyautogui.screenshot()
    return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)


def find_vertical_ellipsis(
    screen: np.ndarray,
    left_fraction: float,
) -> list[tuple[int, int]]:
    """
    Detect tiny vertical three-dot icons in screenshot-pixel coordinates.
    """
    height, width = screen.shape[:2]
    crop_width = int(width * left_fraction)
    crop = screen[40:height - 40, :crop_width]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)

    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    blobs: list[tuple[float, float]] = []

    for index in range(1, count):
        _, _, w, h, area = stats[index]
        cx, cy = centroids[index]

        if 1 <= w <= 9 and 1 <= h <= 9 and 1 <= area <= 55:
            blobs.append((cx, cy))

    candidates: list[tuple[int, int]] = []

    for ax, ay in blobs:
        for bx, by in blobs:
            first_gap = by - ay

            if not (2 <= first_gap <= 18):
                continue
            if abs(bx - ax) > 5:
                continue

            for cx, cy in blobs:
                second_gap = cy - by

                if (
                    2 <= second_gap <= 18
                    and abs(cx - ax) <= 5
                    and abs(first_gap - second_gap) <= 5
                ):
                    px = int(round((ax + bx + cx) / 3))
                    py = int(round((ay + by + cy) / 3)) + 40
                    candidates.append((px, py))

    candidates.sort(key=lambda point: point[1])

    deduped: list[tuple[int, int]] = []
    for x, y in candidates:
        if all(abs(x - ox) > 12 or abs(y - oy) > 30 for ox, oy in deduped):
            deduped.append((x, y))

    return deduped


def keep_dominant_x_column(
    points: list[tuple[int, int]],
    screenshot_width: int,
    min_x_fraction: float,
    max_x_fraction: float,
    tolerance_px: int,
) -> list[tuple[int, int]]:
    """
    Replay menu buttons form a strong vertical column. Keep the largest x cluster.
    """
    filtered = [
        (x, y)
        for x, y in points
        if int(screenshot_width * min_x_fraction)
        <= x
        <= int(screenshot_width * max_x_fraction)
    ]

    if not filtered:
        return []

    clusters: list[list[tuple[int, int]]] = []

    for point in sorted(filtered, key=lambda p: p[0]):
        x, _ = point
        placed = False

        for cluster in clusters:
            cluster_mean = sum(p[0] for p in cluster) / len(cluster)

            if abs(x - cluster_mean) <= tolerance_px:
                cluster.append(point)
                placed = True
                break

        if not placed:
            clusters.append([point])

    best = max(clusters, key=len)
    best.sort(key=lambda p: p[1])
    return best


def run(args: argparse.Namespace) -> None:
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.08

    print(
        "Bring Kaggle Game History to the front.\n"
        f"Starting in {args.start_delay:.0f} seconds.\n"
        "Emergency stop: move the mouse to the upper-left corner."
    )
    time.sleep(args.start_delay)

    logical_w, logical_h = pyautogui.size()
    total = 0

    for round_index in range(1, args.max_rounds + 1):
        screen = screenshot_bgr()
        image_h, image_w = screen.shape[:2]

        # macOS Retina screenshots may be 2x or more larger than PyAutoGUI's
        # logical mouse coordinate space.
        scale_x = logical_w / image_w
        scale_y = logical_h / image_h

        raw = find_vertical_ellipsis(
            screen,
            left_fraction=args.search_left_fraction,
        )

        column = keep_dominant_x_column(
            raw,
            screenshot_width=image_w,
            min_x_fraction=args.menu_min_x_fraction,
            max_x_fraction=args.menu_max_x_fraction,
            tolerance_px=max(10, int(18 / scale_x)),
        )

        # Convert screenshot pixels to logical mouse coordinates.
        buttons = [
            (
                int(round(x * scale_x)),
                int(round(y * scale_y)),
            )
            for x, y in column
        ]

        buttons = [
            (x, y)
            for x, y in buttons
            if 90 <= y <= logical_h - 80
        ]

        print(
            f"Round {round_index}: screenshot={image_w}x{image_h}, "
            f"mouse-space={logical_w}x{logical_h}, "
            f"scale=({scale_x:.3f}, {scale_y:.3f}), "
            f"found {len(buttons)} replay-menu candidates"
        )

        if not buttons:
            print("No valid menu column found; stopping.")
            break

        successful = 0

        for index, (menu_x, menu_y) in enumerate(buttons, start=1):
            try:
                pyautogui.click(menu_x, menu_y)
                time.sleep(args.menu_delay)

                target_x = menu_x + args.download_offset_x
                target_y = menu_y + args.download_offset_y

                # Never click outside the visible display.
                if not (
                    0 <= target_x < logical_w
                    and 0 <= target_y < logical_h
                ):
                    pyautogui.press("esc")
                    print(
                        f"  skipped candidate {index}: "
                        f"target ({target_x}, {target_y}) is off-screen"
                    )
                    continue

                pyautogui.click(target_x, target_y)
                time.sleep(args.download_delay)

                total += 1
                successful += 1
                print(
                    f"  attempt {total}: menu=({menu_x},{menu_y}) "
                    f"download=({target_x},{target_y})"
                )

            except pyautogui.FailSafeException:
                print("Stopped by failsafe.")
                return

        if successful == 0:
            print("No valid download clicks this round; stopping.")
            break

        pyautogui.moveTo(
            int(logical_w * args.scroll_x_fraction),
            int(logical_h * 0.72),
        )
        pyautogui.scroll(-abs(args.scroll_clicks))
        time.sleep(args.scroll_delay)

    print(f"Done. Total download attempts: {total}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retina-aware screen-vision Kaggle replay downloader."
        )
    )

    parser.add_argument("--max-rounds", type=int, default=300)
    parser.add_argument("--start-delay", type=float, default=5.0)
    parser.add_argument("--menu-delay", type=float, default=0.45)
    parser.add_argument("--download-delay", type=float, default=0.9)
    parser.add_argument("--scroll-delay", type=float, default=1.0)
    parser.add_argument("--scroll-clicks", type=int, default=7)

    parser.add_argument(
        "--search-left-fraction",
        type=float,
        default=0.45,
        help="How much of the screenshot to inspect.",
    )
    parser.add_argument(
        "--menu-min-x-fraction",
        type=float,
        default=0.18,
        help="Minimum x-position for the vertical replay-menu column.",
    )
    parser.add_argument(
        "--menu-max-x-fraction",
        type=float,
        default=0.28,
        help="Maximum x-position for the vertical replay-menu column.",
    )
    parser.add_argument(
        "--scroll-x-fraction",
        type=float,
        default=0.18,
    )
    parser.add_argument(
        "--download-offset-x",
        type=int,
        default=82,
        help="Logical screen pixels right of the detected menu button.",
    )
    parser.add_argument(
        "--download-offset-y",
        type=int,
        default=26,
        help="Logical screen pixels below the detected menu button.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
