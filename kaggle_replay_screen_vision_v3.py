from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui


DEBUG_DIR = Path("macro_debug")


def screenshot_bgr() -> np.ndarray:
    shot = pyautogui.screenshot()
    return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)


def find_vertical_ellipsis(
    screen: np.ndarray,
    left_fraction: float = 0.45,
) -> list[tuple[int, int]]:
    """
    Find vertical three-dot buttons in Kaggle's dark left game-history pane.
    """
    height, width = screen.shape[:2]
    crop_width = int(width * left_fraction)
    crop = screen[40:height - 40, :crop_width]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # The dots are light gray/white against a dark background.
    _, mask = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)

    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    blobs: list[tuple[float, float]] = []

    for index in range(1, count):
        x, y, w, h, area = stats[index]
        cx, cy = centroids[index]

        if (
            1 <= w <= 8
            and 1 <= h <= 8
            and 1 <= area <= 45
        ):
            blobs.append((cx, cy))

    candidates: list[tuple[int, int]] = []

    for ax, ay in blobs:
        for bx, by in blobs:
            if by <= ay:
                continue

            first_gap = by - ay
            if not (2 <= first_gap <= 15):
                continue

            if abs(bx - ax) > 4:
                continue

            for cx, cy in blobs:
                if cy <= by:
                    continue

                second_gap = cy - by

                if (
                    2 <= second_gap <= 15
                    and abs(cx - ax) <= 4
                    and abs(first_gap - second_gap) <= 4
                ):
                    px = int(round((ax + bx + cx) / 3))
                    py = int(round((ay + by + cy) / 3)) + 40
                    candidates.append((px, py))

    # Deduplicate and keep buttons near the right side of the left panel.
    candidates.sort(key=lambda point: point[1])
    deduped: list[tuple[int, int]] = []

    for x, y in candidates:
        if all(abs(x - ox) > 10 or abs(y - oy) > 22 for ox, oy in deduped):
            deduped.append((x, y))

    min_x = int(width * 0.16)
    max_x = int(width * left_fraction)

    return [
        (x, y)
        for x, y in deduped
        if min_x <= x <= max_x and 95 <= y <= height - 75
    ]


def popup_visible_near_click(
    before: np.ndarray,
    after: np.ndarray,
    click_x: int,
    click_y: int,
) -> bool:
    """
    Loosely confirm that a popup appeared near the clicked three-dot button.
    """
    height, width = before.shape[:2]

    x1 = max(0, click_x - 25)
    x2 = min(width, click_x + 260)
    y1 = max(0, click_y - 40)
    y2 = min(height, click_y + 110)

    before_crop = before[y1:y2, x1:x2]
    after_crop = after[y1:y2, x1:x2]

    diff = cv2.absdiff(before_crop, after_crop)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    changed_pixels = int(np.count_nonzero(gray > 15))
    return changed_pixels >= 250


def save_debug(
    before: np.ndarray,
    after: np.ndarray,
    round_index: int,
    button_index: int,
) -> None:
    DEBUG_DIR.mkdir(exist_ok=True)
    cv2.imwrite(
        str(DEBUG_DIR / f"round_{round_index}_button_{button_index}_before.png"),
        before,
    )
    cv2.imwrite(
        str(DEBUG_DIR / f"round_{round_index}_button_{button_index}_after.png"),
        after,
    )


def run(args: argparse.Namespace) -> None:
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.08

    print(
        "Bring Kaggle Game History to the front.\n"
        f"Starting in {args.start_delay:.0f} seconds.\n"
        "Emergency stop: move the mouse to the upper-left corner."
    )
    time.sleep(args.start_delay)

    screen_width, screen_height = pyautogui.size()

    total = 0
    stagnant_rounds = 0

    for round_index in range(1, args.max_rounds + 1):
        base = screenshot_bgr()
        buttons = find_vertical_ellipsis(base, args.left_fraction)

        print(
            f"Round {round_index}: found "
            f"{len(buttons)} possible replay menus"
        )

        successful = 0

        for button_index, (x, y) in enumerate(buttons, start=1):
            try:
                before = screenshot_bgr()

                pyautogui.click(x, y)
                time.sleep(args.menu_delay)

                after = screenshot_bgr()
                popup_found = popup_visible_near_click(
                    before,
                    after,
                    x,
                    y,
                )

                if args.debug:
                    save_debug(
                        before,
                        after,
                        round_index,
                        button_index,
                    )

                if not popup_found:
                    print(
                        f"  button {button_index}: "
                        "popup not confidently detected; trying fallback"
                    )

                # Kaggle's one-item popup opens to the right and slightly below
                # the vertical ellipsis. These offsets come from the current UI,
                # not from user calibration.
                target_x = x + args.download_offset_x
                target_y = y + args.download_offset_y

                pyautogui.click(target_x, target_y)
                time.sleep(args.download_delay)

                successful += 1
                total += 1
                print(
                    f"  download attempt {total} "
                    f"at ({target_x}, {target_y})"
                )

            except pyautogui.FailSafeException:
                print("Stopped by failsafe.")
                return

        if successful == 0:
            stagnant_rounds += 1
        else:
            stagnant_rounds = 0

        if stagnant_rounds >= args.stop_after_stagnant:
            print("No downloads attempted for several rounds; stopping.")
            break

        # Keep pointer inside the left game-history list before scrolling.
        pyautogui.moveTo(
            int(screen_width * args.scroll_x_fraction),
            int(screen_height * 0.72),
        )
        pyautogui.scroll(-abs(args.scroll_clicks))
        time.sleep(args.scroll_delay)

    print(f"Done. Total download attempts: {total}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Vision-based Kaggle replay downloader using screen detection "
            "and fixed UI-relative popup offsets."
        )
    )

    parser.add_argument("--max-rounds", type=int, default=300)
    parser.add_argument("--start-delay", type=float, default=5.0)
    parser.add_argument("--menu-delay", type=float, default=0.45)
    parser.add_argument("--download-delay", type=float, default=0.9)
    parser.add_argument("--scroll-delay", type=float, default=1.0)
    parser.add_argument("--scroll-clicks", type=int, default=7)
    parser.add_argument("--left-fraction", type=float, default=0.45)
    parser.add_argument("--scroll-x-fraction", type=float, default=0.18)
    parser.add_argument("--stop-after-stagnant", type=int, default=4)

    parser.add_argument(
        "--download-offset-x",
        type=int,
        default=82,
        help="Horizontal offset from three-dot button to popup item center.",
    )
    parser.add_argument(
        "--download-offset-y",
        type=int,
        default=26,
        help="Vertical offset from three-dot button to popup item center.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save before/after screenshots in macro_debug/.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
