from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pyautogui


CONFIG_PATH = Path("kaggle_replay_macro_config.json")


@dataclass
class MacroConfig:
    menu_x: int
    first_menu_y: int
    download_dx: int
    download_dy: int
    scroll_x: int
    scroll_y: int
    row_spacing: int = 101


def wait_for_position(message: str) -> tuple[int, int]:
    print()
    print(message)
    input("Hover the mouse there, then press Enter in Terminal...")
    pos = pyautogui.position()
    print(f"Recorded: ({pos.x}, {pos.y})")
    return pos.x, pos.y


def calibrate(row_spacing: int) -> MacroConfig:
    print(
        "\nCALIBRATION\n"
        "1. Open Kaggle Game History in Chrome.\n"
        "2. Set browser zoom to 100%.\n"
        "3. Scroll the game list to the top.\n"
        "4. Keep Chrome and Terminal side by side so you can press Enter.\n"
    )

    menu_x, first_menu_y = wait_for_position(
        "Hover over the three-dot button for the FIRST fully visible replay."
    )

    print(
        "\nNow click that three-dot button manually so the menu opens."
    )
    download_x, download_y = wait_for_position(
        'Hover over the center of the "Download replay" menu item.'
    )

    scroll_x, scroll_y = wait_for_position(
        "Hover over the middle of the LEFT game-history list."
    )

    config = MacroConfig(
        menu_x=menu_x,
        first_menu_y=first_menu_y,
        download_dx=download_x - menu_x,
        download_dy=download_y - first_menu_y,
        scroll_x=scroll_x,
        scroll_y=scroll_y,
        row_spacing=row_spacing,
    )

    CONFIG_PATH.write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved calibration to {CONFIG_PATH.resolve()}")
    return config


def load_config() -> MacroConfig:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return MacroConfig(**data)


def click_replay_download(
    config: MacroConfig,
    row_index: int,
    menu_delay: float,
    download_delay: float,
) -> None:
    menu_y = config.first_menu_y + row_index * config.row_spacing

    pyautogui.click(config.menu_x, menu_y)
    time.sleep(menu_delay)

    pyautogui.click(
        config.menu_x + config.download_dx,
        menu_y + config.download_dy,
    )
    time.sleep(download_delay)


def run_macro(
    config: MacroConfig,
    pages: int,
    rows_per_page: int,
    scroll_clicks: int,
    menu_delay: float,
    download_delay: float,
    page_delay: float,
    start_delay: float,
) -> None:
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05

    print(
        "\nREADY\n"
        "Bring Chrome to the front and place the game list at the top.\n"
        f"Starting in {start_delay:.0f} seconds.\n"
        "Emergency stop: move the mouse sharply to the UPPER-LEFT corner.\n"
    )
    time.sleep(start_delay)

    attempted = 0

    for page_index in range(pages):
        print(f"Page {page_index + 1}/{pages}")

        for row_index in range(rows_per_page):
            try:
                click_replay_download(
                    config=config,
                    row_index=row_index,
                    menu_delay=menu_delay,
                    download_delay=download_delay,
                )
                attempted += 1
                print(f"  attempted replay {attempted}")
            except pyautogui.FailSafeException:
                print("\nStopped by PyAutoGUI failsafe.")
                return

        # Keep the mouse over the game list so only that pane scrolls.
        pyautogui.moveTo(config.scroll_x, config.scroll_y)
        pyautogui.scroll(-abs(scroll_clicks))
        time.sleep(page_delay)

    print(f"\nDone. Attempted {attempted} replay downloads.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen-coordinate macro for downloading Kaggle replay JSON files."
        )
    )

    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Create or replace the coordinate calibration file.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=220,
        help="Number of visible batches to process.",
    )
    parser.add_argument(
        "--rows-per-page",
        type=int,
        default=6,
        help=(
            "Rows clicked before scrolling. Six avoids menus near the "
            "bottom edge opening upward."
        ),
    )
    parser.add_argument(
        "--scroll-clicks",
        type=int,
        default=6,
        help="Mouse-wheel scroll amount after each batch.",
    )
    parser.add_argument(
        "--row-spacing",
        type=int,
        default=101,
        help="Vertical pixel distance between replay three-dot buttons.",
    )
    parser.add_argument(
        "--menu-delay",
        type=float,
        default=0.35,
        help="Wait after opening the three-dot menu.",
    )
    parser.add_argument(
        "--download-delay",
        type=float,
        default=0.8,
        help="Wait after clicking Download replay.",
    )
    parser.add_argument(
        "--page-delay",
        type=float,
        default=1.2,
        help="Wait after scrolling the game list.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=5.0,
        help="Seconds to switch focus back to Chrome.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.calibrate or not CONFIG_PATH.exists():
        config = calibrate(args.row_spacing)
        print(
            "\nCalibration finished. Close any open replay menu, then run "
            "the script again without --calibrate."
        )
        return

    config = load_config()

    run_macro(
        config=config,
        pages=max(1, args.pages),
        rows_per_page=max(1, args.rows_per_page),
        scroll_clicks=max(1, args.scroll_clicks),
        menu_delay=max(0.1, args.menu_delay),
        download_delay=max(0.2, args.download_delay),
        page_delay=max(0.2, args.page_delay),
        start_delay=max(1.0, args.start_delay),
    )


if __name__ == "__main__":
    main()
