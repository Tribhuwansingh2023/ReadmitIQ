import { chromium } from "../web/node_modules/playwright/index.mjs";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, "..");
const DEMO_DIR = path.resolve(ROOT_DIR, "docs/demo");
const TMP_VIDEO_DIR = path.resolve(DEMO_DIR, "tmp_recordings");
const TARGET_VIDEO = path.resolve(DEMO_DIR, "ReadmitIQ_tour.webm");
const BASE_URL = "http://localhost:5173";

if (!fs.existsSync(DEMO_DIR)) {
  fs.mkdirSync(DEMO_DIR, { recursive: true });
}
if (!fs.existsSync(TMP_VIDEO_DIR)) {
  fs.mkdirSync(TMP_VIDEO_DIR, { recursive: true });
}

async function recordTour() {
  console.log("==========================================================");
  console.log("🎬 Starting ReadmitIQ UI Tour Video Recording (60-90s)");
  console.log("==========================================================");

  const browser = await chromium.launch({
    channel: "msedge",
    headless: true,
  });

  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    recordVideo: {
      dir: TMP_VIDEO_DIR,
      size: { width: 1280, height: 720 },
    },
  });

  const page = await context.newPage();

  // Helper smooth scroll
  async function smoothScroll(distance, duration = 1500) {
    await page.evaluate(
      async ({ d, dur }) => {
        const start = window.scrollY;
        const startTime = performance.now();
        return new Promise((resolve) => {
          function step(time) {
            const elapsed = time - startTime;
            const progress = Math.min(elapsed / dur, 1);
            window.scrollTo(0, start + d * progress);
            if (progress < 1) {
              requestAnimationFrame(step);
            } else {
              resolve();
            }
          }
          requestAnimationFrame(step);
        });
      },
      { d: distance, dur: duration }
    );
  }

  // 1. Initial Load: Desk
  console.log("1. Opening Discharge Desk (Presentation Mode)...");
  await page.goto(BASE_URL, { waitUntil: "networkidle" });
  await page.waitForTimeout(2000);

  // Activate Presentation Mode (Key P)
  await page.keyboard.press("p");
  await page.waitForTimeout(2000);

  // 2. Start Live Simulated Feed
  console.log("2. Starting Live Simulated Feed...");
  const startFeedBtn = await page.locator("button:has-text('Start Feed')");
  if (await startFeedBtn.isVisible()) {
    await startFeedBtn.click();
  }
  await page.waitForTimeout(5000); // Watch arrivals and odometers

  // 3. Filter / Capacity Exploration
  console.log("3. Toggling Capacity Mode (Top 10)...");
  const capInput = await page.locator("input[aria-label='Follow-up patient capacity limit']");
  if (await capInput.isVisible()) {
    await capInput.fill("10");
    const limitBtn = await page.locator("button:has-text('Limit')");
    if (await limitBtn.isVisible()) await limitBtn.click();
    await page.waitForTimeout(3000);
  }

  // 4. Open Critical Patient Discharge Brief
  console.log("4. Opening Critical Patient Discharge Brief...");
  const criticalRow = await page.locator("tr[data-testid='patient-row']").first();
  await criticalRow.click();
  await page.waitForTimeout(3000);

  // Inspect Brief: scroll modal content
  console.log("   - Inspecting Precedent Orbit & Driver Divergence...");
  await page.evaluate(() => {
    const modal = document.querySelector(".max-h-\\[90vh\\]");
    if (modal) modal.scrollTo({ top: 400, behavior: "smooth" });
  });
  await page.waitForTimeout(3500);

  console.log("   - Inspecting Suggested Actions & Toggling Action...");
  await page.evaluate(() => {
    const modal = document.querySelector(".max-h-\\[90vh\\]");
    if (modal) modal.scrollTo({ top: 900, behavior: "smooth" });
  });
  await page.waitForTimeout(2500);

  // Click an action checkbox to persist
  const actionCheckbox = await page.locator("input[type='checkbox']").first();
  if (await actionCheckbox.isVisible()) {
    await actionCheckbox.click();
    await page.waitForTimeout(2000);
  }

  // Close Brief
  await page.keyboard.press("Escape");
  await page.waitForTimeout(2000);

  // 5. Navigate to Live Monitoring
  console.log("5. Navigating to Live Monitoring...");
  await page.keyboard.press("4"); // shortcut key 4 in presentation bar
  await page.waitForTimeout(4000);

  // Advance Time Machine by 30 days
  console.log("   - Triggering Time Machine (+30 days outcomes)...");
  const tmBtn = await page.locator("button:has-text('+30d')");
  if (await tmBtn.isVisible()) {
    await tmBtn.click();
    await page.waitForTimeout(5000); // Watch calibration, PSI, ROC-AUC recalculate
  }

  // 6. Navigate to Evidence Dossier
  console.log("6. Navigating to Evidence Dossier...");
  await page.keyboard.press("5"); // shortcut key 5
  await page.waitForTimeout(3500);

  // Switch Evidence tabs
  console.log("   - Inspecting Fairness Small Multiples...");
  const fairnessTab = await page.locator("button:has-text('Fairness & Subgroups')");
  if (await fairnessTab.isVisible()) {
    await fairnessTab.click();
    await page.waitForTimeout(4000);
  }

  console.log("   - Inspecting Leakage Safeguards Architecture...");
  const safeguardsTab = await page.locator("button:has-text('How It Works & Safeguards')");
  if (await safeguardsTab.isVisible()) {
    await safeguardsTab.click();
    await page.waitForTimeout(4000);
  }

  // 7. Navigate to Integration Architecture
  console.log("7. Navigating to Integration Architecture...");
  await page.click("button:has-text('Integration')");
  await page.waitForTimeout(4000);

  // 8. Return to Desk & Final Review
  console.log("8. Returning to Discharge Desk...");
  await page.keyboard.press("1"); // key 1 for Desk
  await page.waitForTimeout(3500);

  // Theme switch to Night mode
  const themeBtn = await page.locator("button[aria-label*='Switch to']");
  if (await themeBtn.isVisible()) {
    await themeBtn.click();
    await page.waitForTimeout(3000);
    await themeBtn.click();
    await page.waitForTimeout(2000);
  }

  console.log("Finishing tour and closing browser...");
  const video = page.video();
  await page.close();
  await context.close();
  await browser.close();

  // Obtain video file
  const videoPath = await video.path();
  console.log(`Raw video saved to: ${videoPath}`);

  if (fs.existsSync(videoPath)) {
    fs.copyFileSync(videoPath, TARGET_VIDEO);
    console.log(`✅ Final Tour Video successfully copied to: ${TARGET_VIDEO}`);
    const stats = fs.statSync(TARGET_VIDEO);
    console.log(`Video size: ${(stats.size / (1024 * 1024)).toFixed(2)} MB`);
  }

  // Clean up tmp directory
  try {
    fs.rmSync(TMP_VIDEO_DIR, { recursive: true, force: true });
  } catch (e) {}

  console.log("🎉 UI TOUR RECORDING FINISHED!");
}

recordTour().catch((err) => {
  console.error("UI Tour recording failed:", err);
  process.exit(1);
});
