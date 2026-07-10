from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui


DOTS_TEMPLATE = Path(__file__).with_name("kaggle_menu_dots_template.png")
DOWNLOAD_TEMPLATE = Path(__file__).with_name("kaggle_download_replay_template.png")


def screenshot_bgr() -> np.ndarray:
    shot = pyautogui.screenshot()
    return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)


def find_matches(
    screen: np.ndarray,
    template_path: Path,
    threshold: float,
    region: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, int, int, int, float]]:
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(template_path)

    ox = oy = 0
    target = screen

    if region is not None:
        x, y, w, h = region
        target = screen[y:y+h, x:x+w]
        ox, oy = x, y

    result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(result >= threshold)

    h, w = template.shape[:2]
    candidates = [
        (int(x + ox), int(y + oy), w, h, float(result[y, x]))
        for y, x in zip(ys, xs)
    ]

    # Non-maximum suppression.
    candidates.sort(key=lambda item: item[4], reverse=True)
    kept: list[tuple[int, int, int, int, float]] = []

    for candidate in candidates:
        x, y, cw, ch, score = candidate
        cx, cy = x + cw / 2, y + ch / 2

        if all(
            abs(cx - (kx + kw / 2)) > max(cw, kw) * 0.6
            or abs(cy - (ky + kh / 2)) > max(ch, kh) * 0.6
            for kx, ky, kw, kh, _ in kept
        ):
            kept.append(candidate)

    return kept


def click_center(match: tuple[int, int, int, int, float]) -> None:
    x, y, w, h, _ = match
    pyautogui.click(x + w // 2, y + h // 2)


def run(args: argparse.Namespace) -> None:
    if not DOTS_TEMPLATE.exists() or not DOWNLOAD_TEMPLATE.exists():
        raise SystemExit(
            "Template images must be in the same folder as this script."
        )

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.08

    print(
        "Bring the Kaggle Game History window to the front.\n"
        "Keep browser zoom and macOS display scaling the same as in the screenshot.\n"
        f"Starting in {args.start_delay:.0f} seconds.\n"
        "Emergency stop: move the mouse to the upper-left corner."
    )
    time.sleep(args.start_delay)

    screen_width, screen_height = pyautogui.size()
    left_region = (0, 40, int(screen_width * 0.34), screen_height - 80)

    total = 0
    stagnant_rounds = 0

    for round_index in range(args.max_rounds):
        screen = screenshot_bgr()
        dots = find_matches(
            screen,
            DOTS_TEMPLATE,
            args.dots_threshold,
            region=left_region,
        )

        # Ignore header/filter buttons; sort top-to-bottom.
        dots = [m for m in dots if m[1] > 100]
        dots.sort(key=lambda m: m[1])

        successful = 0

        for dot in dots:
            try:
                click_center(dot)
                time.sleep(args.menu_delay)

                menu_screen = screenshot_bgr()
                downloads = find_matches(
                    menu_screen,
                    DOWNLOAD_TEMPLATE,
                    args.download_threshold,
                )

                visible = [
                    m for m in downloads
                    if 0 <= m[0] < screen_width
                    and 0 <= m[1] < screen_height
                ]

                if not visible:
                    pyautogui.press("esc")
                    continue

                # The newly opened menu should be nearest the clicked dots.
                dx = dot[0] + dot[2] / 2
                dy = dot[1] + dot[3] / 2
                visible.sort(
                    key=lambda m: (
                        (m[0] + m[2] / 2 - dx) ** 2
                        + (m[1] + m[3] / 2 - dy) ** 2
                    )
                )

                click_center(visible[0])
                time.sleep(args.download_delay)

                successful += 1
                total += 1
                print(f"Downloaded/attempted: {total}")

            except pyautogui.FailSafeException:
                print("Stopped by failsafe.")
                return

        print(
            f"Round {round_index + 1}: found {len(dots)} menu icons, "
            f"clicked {successful} downloads"
        )

        if successful == 0:
            stagnant_rounds += 1
        else:
            stagnant_rounds = 0

        if stagnant_rounds >= args.stop_after_stagnant:
            print("No usable replay menus found for several rounds; stopping.")
            break

        # Scroll only the left game-history panel.
        pyautogui.moveTo(
            int(screen_width * 0.18),
            int(screen_height * 0.75),
        )
        pyautogui.scroll(-abs(args.scroll_clicks))
        time.sleep(args.scroll_delay)

    print(f"Done. Total download attempts: {total}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen-vision Kaggle replay downloader; no CDP or DOM access."
    )
    parser.add_argument("--max-rounds", type=int, default=300)
    parser.add_argument("--scroll-clicks", type=int, default=7)
    parser.add_argument("--start-delay", type=float, default=5.0)
    parser.add_argument("--menu-delay", type=float, default=0.35)
    parser.add_argument("--download-delay", type=float, default=0.8)
    parser.add_argument("--scroll-delay", type=float, default=1.0)
    parser.add_argument("--dots-threshold", type=float, default=0.82)
    parser.add_argument("--download-threshold", type=float, default=0.80)
    parser.add_argument("--stop-after-stagnant", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
