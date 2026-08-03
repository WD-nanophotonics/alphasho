import { expect, test } from "@playwright/test";

test("two browsers synchronize a real PeerJS room", async ({ browser, browserName }) => {
  test.skip(!process.env.PEERJS_SMOKE || browserName !== "chromium", "Set PEERJS_SMOKE=1 for the public PeerServer smoke test");
  const hostContext = await browser.newContext();
  const guestContext = await browser.newContext();
  const host = await hostContext.newPage();
  const guest = await guestContext.newPage();
  try {
    await host.goto("./");
    await host.locator('[data-action="setup-online"]').click();
    await host.locator('input[name="host-side"][value="0"] + span').click();
    await host.locator('[data-action="create-room"]').click();
    const room = (await host.locator(".room-chip b").textContent())?.trim();
    expect(room).toMatch(/^[A-Z2-9]{10}$/);

    await guest.goto(`./?room=${room}`);
    await guest.locator('[data-action="join-room"]').click();
    await expect(host.getByText(/Your turn|あなたの手番|轮到你/)).toBeVisible({ timeout: 20_000 });
    for (let game = 0; game < 3; game += 1) {
      await host.locator('[data-square="24"]').click();
      await host.locator('[data-square="23"]').click();
      await expect(guest.locator(".move-list b")).toContainText("7g7f", { timeout: 10_000 });
      await guest.locator('.desktop-actions [data-action="resign"]').click();
      await expect(host.locator(".game-status strong")).toContainText(/wins|勝ち|胜/, { timeout: 10_000 });
      await expect(guest.locator(".game-status strong")).toContainText(/wins|勝ち|胜/, { timeout: 10_000 });
      if (game < 2) {
        await host.locator('.desktop-actions [data-action="restart"]').click();
        await expect(host.locator(".move-list b")).toHaveCount(0);
        await expect(guest.locator(".move-list b")).toHaveCount(0);
      }
    }
  } finally {
    await hostContext.close(); await guestContext.close();
  }
});
