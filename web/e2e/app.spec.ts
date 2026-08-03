import { expect, test } from "@playwright/test";

test("starts a local game and moves a pawn", async ({ page, isMobile }) => {
  await page.goto("./");
  await expect(page.getByRole("heading", { name: "AlphaSho" })).toBeVisible();
  await page.locator('[data-action="start-local"]').click();
  await expect(page.locator(".board .square")).toHaveCount(81);
  const from = page.locator('[data-square="24"]');
  const to = page.locator('[data-square="23"]');
  if (isMobile) { await from.tap(); await expect(from).toHaveClass(/selected/); await expect(to).toHaveClass(/target/); await to.tap(); }
  else { await from.click(); await to.click(); }
  await expect(page.locator(".move-list b")).toContainText("7g7f");
});

test("switches languages and lays out on a phone", async ({ page, isMobile }) => {
  await page.goto("./");
  await page.locator('select[data-action="language"]').selectOption("zh");
  await expect(page.getByText("人机对战", { exact: true })).toBeVisible();
  if (isMobile) await expect(page.locator(".mode-grid")).toHaveCSS("grid-template-columns", /.+/);
});

test("opens an invitation from its room query", async ({ page }) => {
  await page.goto("./?room=ABCDEFGHJK");
  await expect(page.locator("#room-code")).toHaveValue("ABCDEFGHJK");
});

test("runs the easy ABP in a worker without blocking the board", async ({ page }) => {
  await page.goto("./");
  await page.locator('[data-action="setup-cpu"]').click();
  await page.locator('input[name="difficulty"][value="easy"] + span').click();
  await page.locator('[data-action="start-cpu"]').click();
  await page.locator('[data-square="24"]').click();
  await page.locator('[data-square="23"]').click();
  await expect(page.locator(".move-list b")).toHaveCount(2, { timeout: 10_000 });
  await expect(page.locator(".board .square")).toHaveCount(81);
});

test("installs an offline-capable service worker", async ({ page, context, browserName }) => {
  test.skip(browserName !== "chromium", "PWA cache smoke is deterministic in Chromium");
  await page.goto("./");
  await page.evaluate(() => navigator.serviceWorker.ready);
  await context.setOffline(true);
  await page.reload();
  await expect(page.getByRole("heading", { name: "AlphaSho" })).toBeVisible();
  await context.setOffline(false);
});
