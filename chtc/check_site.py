"""Check local report assets and browser interactions at two viewport sizes."""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    parser.add_argument("--screenshots", type=Path, required=True)
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for name, width, height in [("desktop", 1440, 1000), ("mobile", 390, 844)]:
            page = browser.new_page(viewport={"width": width, "height": height})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(args.site.resolve().as_uri())
            page.wait_for_function("document.querySelectorAll('.frame img').length === 3 && [...document.querySelectorAll('.frame img')].every(x => x.complete && x.naturalWidth > 0)")
            assert page.locator(".agent").count() == 3
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.locator("#next").click()
            assert page.locator("#position").inner_text().startswith("Step 2 / ")
            page.locator("#play").click()
            page.wait_for_timeout(1300)
            page.locator("#play").click()
            assert int(page.locator("#step").input_value()) >= 2
            page.locator("#seed").select_option(index=1)
            page.wait_for_function("document.querySelector('#play').disabled === false")
            assert page.locator("#step").input_value() == "0"
            page.locator(".agent details").first.locator("summary").click()
            page.screenshot(path=str(args.screenshots / f"{name}.png"), full_page=True)
            assert not errors, errors
            page.close()
        browser.close()
    print("Desktop/mobile images, replay, selectors and layout passed")


if __name__ == "__main__":
    main()
