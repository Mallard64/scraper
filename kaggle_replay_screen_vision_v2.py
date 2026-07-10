from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui


def screenshot_bgr() -> np.ndarray:
    shot = pyautogui.screenshot()
    return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)


def find_vertical_ellipsis(
    screen: np.ndarray,
    left_fraction: float = 0.36,
) -> list[tuple[int, int]]:
    """
    Detect vertical three-dot menu icons in the left pane without templates.
    Looks for three tiny bright blobs aligned vertically.
    """
    h, w = screen.shape[:2]
    crop = screen[40:h - 40, 0:int(w * left_fraction)]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Bright pixels on Kaggle's dark UI.
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Remove text/lines by keeping only tiny connected components.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    dots: list[tuple[float, float, int, int]] = []
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        cx, cy = centroids[i]

        if (
            1 <= cw <= 7
            and 1 <= ch <= 7
            and 1 <= area <= 35
            and y > 40
        ):
            dots.append((cx, cy, cw, ch))

    # Group 3 dots with nearly identical x and regular vertical spacing.
    candidates: list[tuple[int, int]] = []

    for i, a in enumerate(dots):
        ax, ay, _, _ = a
        nearby = [
            b for b in dots
            if abs(b[0] - ax) <= 3 and 2 <= b[1] - ay <= 18
        ]

        for b in nearby:
            bx, by, _, _ = b
            for c in dots:
                cx, cy, _, _ = c
                if (
                    abs(cx - ax) <= 3
                    and 2 <= cy - by <= 18
                    and abs((by - ay) - (cy - by)) <= 4
                ):
                    center_x = int(round((ax + bx + cx) / 3))
                    center_y = int(round((ay + by + cy) / 3)) + 40
                    candidates.append((center_x, center_y))

    # Deduplicate nearby detections.
    candidates.sort(key=lambda p: p[1])
    deduped: list[tuple[int, int]] = []

    for x, y in candidates:
        if all(abs(x - ox) > 8 or abs(y - oy) > 20 for ox, oy in deduped):
            deduped.append((x, y))

    # Kaggle row menus are usually near the right edge of the left pane.
    min_x = int(w * 0.18)
    max_x = int(w * left_fraction)
    return [(x, y) for x, y in deduped if min_x <= x <= max_x]


def find_popup_menu(
    before: np.ndarray,
    after: np.ndarray,
    click_x: int,
    click_y: int,
) -> tuple[int, int] | None:
    """
    Detect the newly opened popup menu using frame differencing.
    Returns the center of the changed rectangular region near the clicked icon.
    """
    diff = cv2.absdiff(after, before)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions: list[tuple[float, int, int, int, int]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if not (2500 <= area <= 80000):
            continue

        if not (60 <= w <= 350 and 25 <= h <= 180):
            continue

        # Prefer a popup near and to the right of the clicked menu icon.
        cx = x + w / 2
        cy = y + h / 2
        distance = (cx - click_x) ** 2 + (cy - click_y) ** 2

        if x > click_x - 80 and x < click_x + 420:
            regions.append((distance, x, y, w, h))

    if not regions:
        return None

    _, x, y, w, h = min(regions, key=lambda item: item[0])

    # Click the vertical center of the popup. The menu has one item.
    return int(x + w / 2), int(y + h / 2)


def run(args: argparse.Namespace) -> None:
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.08

    print(
        "Bring Kaggle Game History to the front.\n"
        f"Starting in {args.start_delay:.0f} seconds.\n"
        "Emergency stop: move the mouse to the upper-left corner."
    )
    time.sleep(args.start_delay)

    screen_w, screen_h = pyautogui.size()
    total = 0
    stagnant = 0

    for round_index in range(args.max_rounds):
        base = screenshot_bgr()
        buttons = find_vertical_ellipsis(base, args.left_fraction)

        # Avoid header controls and menus too near the bottom edge.
        buttons = [
            (x, y)
            for x, y in buttons
            if 90 < y < screen_h - 60
        ]

        print(
            f"Round {round_index + 1}: found {len(buttons)} possible replay menus"
        )

        successful = 0

        for x, y in buttons:
            try:
                before = screenshot_bgr()
                pyautogui.click(x, y)
                time.sleep(args.menu_delay)
                after = screenshot_bgr()

                popup = find_popup_menu(before, after, x, y)

                if popup is None:
                    pyautogui.press("esc")
                    continue

                pyautogui.click(*popup)
                time.sleep(args.download_delay)

                successful += 1
                total += 1
                print(f"  download attempt {total}")

            except pyautogui.FailSafeException:
                print("Stopped by failsafe.")
                return

        if successful == 0:
            stagnant += 1
        else:
            stagnant = 0

        if stagnant >= args.stop_after_stagnant:
            print("No replay menus were successfully handled; stopping.")
            break

        # Scroll only the left history pane.
        pyautogui.moveTo(
            int(screen_w * args.scroll_x_fraction),
            int(screen_h * 0.72),
        )
        pyautogui.scroll(-abs(args.scroll_clicks))
        time.sleep(args.scroll_delay)

    print(f"Done. Total download attempts: {total}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Vision-based Kaggle replay downloader using geometric detection, "
            "not templates or browser debugging."
        )
    )
    parser.add_argument("--max-rounds", type=int, default=300)
    parser.add_argument("--start-delay", type=float, default=5.0)
    parser.add_argument("--menu-delay", type=float, default=0.45)
    parser.add_argument("--download-delay", type=float, default=0.9)
    parser.add_argument("--scroll-delay", type=float, default=1.0)
    parser.add_argument("--scroll-clicks", type=int, default=7)
    parser.add_argument("--left-fraction", type=float, default=0.36)
    parser.add_argument("--scroll-x-fraction", type=float, default=0.18)
    parser.add_argument("--stop-after-stagnant", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
