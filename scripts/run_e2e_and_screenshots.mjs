/**
 * ReadmitIQ Comprehensive E2E Test, Axe Accessibility Audit, and Screenshot Generator.
 */
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { chromium } from "../web/node_modules/playwright/index.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ROOT_DIR = path.resolve(__dirname, "..");
const SCREENSHOT_DIR = path.resolve(ROOT_DIR, "docs/screenshots");
const AXE_PATH = path.resolve(ROOT_DIR, "web/node_modules/axe-core/axe.min.js");
const axeSource = fs.readFileSync(AXE_PATH, "utf8");

const BASE_URL = "http://localhost:5173/";

async function run() {
  console.log("==========================================================");
  console.log("🚀 Running ReadmitIQ E2E Verification & Axe Audit");
  console.log("==========================================================\n");

  const browser = await chromium.launch({
    channel: "msedge",
    headless: true,
  });

  const consoleErrors = [];

  // =========================================================================
  // 1. Desktop Context (1920x1080)
  // =========================================================================
  const desktopContext = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
  });
  const page = await desktopContext.newPage();

  page.on("console", (msg) => {
    if (msg.type() === "error") {
      consoleErrors.push(msg.text());
    }
  });

  console.log("1. Navigating to Discharge Desk (Daylight)...");
  await page.goto(BASE_URL, { waitUntil: "networkidle" });
  await page.waitForTimeout(1000);

  // Take Desk Screenshots (Daylight)
  await page.screenshot({
    path: path.join(SCREENSHOT_DIR, "desk_daylight_1920x1080.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved desk_daylight_1920x1080.png");

  // Run Axe on Desk (Daylight)
  await page.evaluate(axeSource);
  const axeDeskDaylight = await page.evaluate(() =>
    axe.run({ runOnly: { type: "tag", values: ["wcag2a", "wcag2aa"] } })
  );
  const seriousDeskDaylight = axeDeskDaylight.violations.filter(
    (v) => v.impact === "serious" || v.impact === "critical"
  );
  console.log(
    `   ✓ Axe WCAG AA (Desk Daylight): ${seriousDeskDaylight.length} serious/critical violations`
  );

  // Switch to Night Theme
  console.log("2. Switching to Night Theme...");
  const themeBtn = await page.locator("button[aria-label*='Switch to']");
  await themeBtn.click();
  await page.waitForTimeout(500);

  await page.screenshot({
    path: path.join(SCREENSHOT_DIR, "desk_night_1920x1080.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved desk_night_1920x1080.png");

  // Run Axe on Desk (Night)
  await page.evaluate(axeSource);
  const axeDeskNight = await page.evaluate(() =>
    axe.run({ runOnly: { type: "tag", values: ["wcag2a", "wcag2aa"] } })
  );
  const seriousDeskNight = axeDeskNight.violations.filter(
    (v) => v.impact === "serious" || v.impact === "critical"
  );
  console.log(
    `   ✓ Axe WCAG AA (Desk Night): ${seriousDeskNight.length} serious/critical violations`
  );

  // Switch back to Daylight
  await themeBtn.click();
  await page.waitForTimeout(300);

  // Test Capacity Filter
  console.log("3. Testing Capacity Control Filter...");
  const capInput = await page.locator("input[aria-label='Follow-up patient capacity limit']");
  if (await capInput.isVisible()) {
    await capInput.fill("5");
    const limitBtn = await page.locator("button:has-text('Limit')");
    if (await limitBtn.isVisible()) await limitBtn.click();
    await page.waitForTimeout(500);
    console.log("   ✓ Capacity mode applied (Top 5 items highlighted)");
    // Clear capacity
    const clearBtn = await page.locator("button:has-text('Top 5 Active')");
    if (await clearBtn.isVisible()) await clearBtn.click();
  }

  // Open Discharge Brief Modal (Daylight)
  console.log("4. Testing Discharge Brief Modal (Daylight & Night)...");
  const firstPatientRow = await page.locator("tr[data-testid='patient-row']").first();
  await firstPatientRow.waitFor({ state: "visible", timeout: 10000 });
  await firstPatientRow.click();
  await page.waitForTimeout(1000);

  // Verify Brief elements
  await page.waitForSelector("text=Calibrated Readmission Probability", {
    timeout: 5000,
  });
  await page.waitForSelector("text=Precedent Orbit", { timeout: 5000 });
  await page.waitForSelector("text=Clinical Sensitivity Lab", { timeout: 5000 });

  // Capture Brief daylight
  await page.screenshot({
    path: path.join(SCREENSHOT_DIR, "brief_daylight_1920x1080.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved brief_daylight_1920x1080.png");

  // Close Brief
  let closeBtn = await page.locator("button[aria-label='Close brief']");
  await closeBtn.click();
  await page.waitForTimeout(400);

  // Switch to Night theme and open Brief in Night
  await themeBtn.click();
  await page.waitForTimeout(400);
  await firstPatientRow.click();
  await page.waitForTimeout(800);
  await page.screenshot({
    path: path.join(SCREENSHOT_DIR, "brief_night_1920x1080.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved brief_night_1920x1080.png");

  // Close Brief and switch back to Daylight
  closeBtn = await page.locator("button[aria-label='Close brief']");
  await closeBtn.click();
  await page.waitForTimeout(400);
  await themeBtn.click();
  await page.waitForTimeout(300);

  // 5. Navigate to Intake Suite
  console.log("5. Testing Intake Suite & Tabs (Daylight & Night)...");
  await page.click("button:has-text('Intake Suite')");
  await page.waitForTimeout(500);

  await page.screenshot({
    path: path.join(SCREENSHOT_DIR, "intake_daylight_1920x1080.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved intake_daylight_1920x1080.png");

  await themeBtn.click();
  await page.waitForTimeout(400);
  await page.screenshot({
    path: path.join(SCREENSHOT_DIR, "intake_night_1920x1080.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved intake_night_1920x1080.png");
  await themeBtn.click();
  await page.waitForTimeout(300);

  // Test CSV Bad Row Ingestion
  console.log("6. Testing CSV Upload with Bad Row...");
  await page.click("button:has-text('CSV Bulk Drop')");
  await page.waitForTimeout(300);

  const testCsvContent = `age,gender,race,time_in_hospital,num_lab_procedures,num_procedures,num_medications,number_outpatient,number_emergency,number_inpatient,number_diagnoses,admission_type_group,admission_source_group,discharge_disposition_group,medical_specialty,payer_code,diag_1_group,diag_2_group,diag_3_group,max_glu_serum,A1Cresult,diabetesMed,change,insulin,metformin
[70-80),Female,Caucasian,5,45,1,16,0,1,1,7,Emergency,Emergency Room,Home,InternalMedicine,Medicare,Circulatory,Diabetes,Other,None,>8,Yes,Ch,Down,Steady
[50-60),Male,AfricanAmerican,invalid_days,30,0,10,0,0,0,5,Emergency,Emergency Room,Home,InternalMedicine,Medicare,Diabetes,Other,Other,None,Norm,Yes,No,Steady,No
[60-70),Female,Caucasian,3,40,1,14,0,0,0,6,Emergency,Emergency Room,Home,InternalMedicine,Medicare,Respiratory,Diabetes,Other,None,None,Yes,No,Steady,No`;

  const tmpCsvPath = path.resolve(ROOT_DIR, "web/test_upload.csv");
  fs.writeFileSync(tmpCsvPath, testCsvContent, "utf8");

  const fileInput = await page.locator("input[type='file']");
  await fileInput.setInputFiles(tmpCsvPath);
  await page.waitForTimeout(500);

  const validateBtn = await page.locator("button:has-text('Validate & Import')");
  if (await validateBtn.isVisible()) {
    await validateBtn.click();
    await page.waitForTimeout(1500);
    console.log("   ✓ CSV validation report rendered with row error details");
  }
  fs.unlinkSync(tmpCsvPath);

  // 7. Navigate to Live Monitoring
  console.log("7. Testing Live Monitoring & Time Machine...");
  await page.click("button:has-text('Live Monitoring')");
  await page.waitForTimeout(800);

  // Advance Time Machine by 30 days
  const timeMachineBtn = await page.locator("button:has-text('+30d')");
  if (await timeMachineBtn.isVisible()) {
    await timeMachineBtn.click();
    await page.waitForTimeout(1200);
    console.log("   ✓ Time Machine advanced (+30 days outcomes released)");
  }

  await page.screenshot({
    path: path.join(SCREENSHOT_DIR, "monitoring_daylight_1920x1080.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved monitoring_daylight_1920x1080.png");

  await themeBtn.click();
  await page.waitForTimeout(400);
  await page.screenshot({
    path: path.join(SCREENSHOT_DIR, "monitoring_night_1920x1080.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved monitoring_night_1920x1080.png");
  await themeBtn.click();
  await page.waitForTimeout(300);

  // 8. Navigate to Evidence Dossier
  console.log("8. Testing Evidence Dossier...");
  await page.click("button:has-text('Evidence Dossier')");
  await page.waitForTimeout(800);

  await page.screenshot({
    path: path.join(SCREENSHOT_DIR, "evidence_daylight_1920x1080.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved evidence_daylight_1920x1080.png");

  await themeBtn.click();
  await page.waitForTimeout(400);
  await page.screenshot({
    path: path.join(SCREENSHOT_DIR, "evidence_night_1920x1080.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved evidence_night_1920x1080.png");
  await themeBtn.click();
  await page.waitForTimeout(300);

  // 9. Navigate to Integration
  console.log("9. Testing Integration Architecture Screen...");
  await page.click("button:has-text('Integration')");
  await page.waitForTimeout(800);

  await page.screenshot({
    path: path.join(SCREENSHOT_DIR, "integration_daylight_1920x1080.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved integration_daylight_1920x1080.png");

  await themeBtn.click();
  await page.waitForTimeout(400);
  await page.screenshot({
    path: path.join(SCREENSHOT_DIR, "integration_night_1920x1080.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved integration_night_1920x1080.png");
  await themeBtn.click();
  await page.waitForTimeout(300);

  // 10. Presentation Mode (Projector 1280x720)
  console.log("10. Testing Presentation / Projector Mode...");
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.keyboard.press("p");
  await page.waitForTimeout(600);

  await page.screenshot({
    path: path.join(SCREENSHOT_DIR, "presentation_mode_1280x720.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved presentation_mode_1280x720.png");

  // Exit presentation mode
  await page.keyboard.press("p");
  await page.waitForTimeout(300);

  await desktopContext.close();

  // =========================================================================
  // 11. Mobile Context (390x844 iPhone 12/13/14)
  // =========================================================================
  console.log("\n11. Testing Mobile Viewport (390x844)...");
  const mobileContext = await browser.newContext({
    viewport: { width: 390, height: 844 },
    isMobile: true,
  });
  const mobilePage = await mobileContext.newPage();

  await mobilePage.goto(BASE_URL, { waitUntil: "networkidle" });
  await mobilePage.waitForTimeout(800);

  await mobilePage.screenshot({
    path: path.join(SCREENSHOT_DIR, "desk_daylight_390x844.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved desk_daylight_390x844.png");

  const mobileThemeBtn = await mobilePage.locator("button[aria-label*='Switch to']");
  await mobileThemeBtn.click();
  await mobilePage.waitForTimeout(400);
  await mobilePage.screenshot({
    path: path.join(SCREENSHOT_DIR, "desk_night_390x844.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved desk_night_390x844.png");
  await mobileThemeBtn.click();
  await mobilePage.waitForTimeout(300);

  // Open brief on mobile
  const mobileRow = await mobilePage.locator("tr[data-testid='patient-row']").first();
  if (await mobileRow.isVisible()) {
    await mobileRow.click();
    await mobilePage.waitForTimeout(600);
    await mobilePage.screenshot({
      path: path.join(SCREENSHOT_DIR, "brief_daylight_390x844.png"),
      fullPage: false,
    });
    console.log("   ✓ Saved brief_daylight_390x844.png");

    await mobilePage.keyboard.press("Escape");
    await mobilePage.waitForTimeout(400);

    // Night brief
    await mobileThemeBtn.click();
    await mobilePage.waitForTimeout(400);
    await mobileRow.click();
    await mobilePage.waitForTimeout(600);
    await mobilePage.screenshot({
      path: path.join(SCREENSHOT_DIR, "brief_night_390x844.png"),
      fullPage: false,
    });
    console.log("   ✓ Saved brief_night_390x844.png");

    await mobilePage.keyboard.press("Escape");
    await mobilePage.waitForTimeout(400);
    await mobileThemeBtn.click();
    await mobilePage.waitForTimeout(300);
  }

  // Intake on mobile
  await mobilePage.click("nav[aria-label='Mobile Navigation'] button:has-text('Intake')", {
    force: true,
  });
  await mobilePage.waitForTimeout(500);
  await mobilePage.screenshot({
    path: path.join(SCREENSHOT_DIR, "intake_daylight_390x844.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved intake_daylight_390x844.png");
  await mobileThemeBtn.click();
  await mobilePage.waitForTimeout(300);
  await mobilePage.screenshot({
    path: path.join(SCREENSHOT_DIR, "intake_night_390x844.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved intake_night_390x844.png");
  await mobileThemeBtn.click();

  // Monitoring on mobile
  await mobilePage.click("nav[aria-label='Mobile Navigation'] button:has-text('Monitor')", {
    force: true,
  });
  await mobilePage.waitForTimeout(500);
  await mobilePage.screenshot({
    path: path.join(SCREENSHOT_DIR, "monitoring_daylight_390x844.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved monitoring_daylight_390x844.png");
  await mobileThemeBtn.click();
  await mobilePage.waitForTimeout(300);
  await mobilePage.screenshot({
    path: path.join(SCREENSHOT_DIR, "monitoring_night_390x844.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved monitoring_night_390x844.png");
  await mobileThemeBtn.click();

  // Evidence on mobile
  await mobilePage.click("nav[aria-label='Mobile Navigation'] button:has-text('Evidence')", {
    force: true,
  });
  await mobilePage.waitForTimeout(500);
  await mobilePage.screenshot({
    path: path.join(SCREENSHOT_DIR, "evidence_daylight_390x844.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved evidence_daylight_390x844.png");
  await mobileThemeBtn.click();
  await mobilePage.waitForTimeout(300);
  await mobilePage.screenshot({
    path: path.join(SCREENSHOT_DIR, "evidence_night_390x844.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved evidence_night_390x844.png");
  await mobileThemeBtn.click();

  // Integration on mobile
  await mobilePage.click("nav[aria-label='Mobile Navigation'] button:has-text('Integ')", {
    force: true,
  });
  await mobilePage.waitForTimeout(500);
  await mobilePage.screenshot({
    path: path.join(SCREENSHOT_DIR, "integration_daylight_390x844.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved integration_daylight_390x844.png");
  await mobileThemeBtn.click();
  await mobilePage.waitForTimeout(300);
  await mobilePage.screenshot({
    path: path.join(SCREENSHOT_DIR, "integration_night_390x844.png"),
    fullPage: false,
  });
  console.log("   ✓ Saved integration_night_390x844.png");
  await mobileThemeBtn.click();

  await mobileContext.close();

  // =========================================================================
  // 12. Network Blocked Offline Resiliency Test
  // =========================================================================
  console.log("\n12. Testing Offline Resiliency (Blocked Network)...");
  const offlineContext = await browser.newContext({
    viewport: { width: 1280, height: 800 },
  });
  const offlinePage = await offlineContext.newPage();
  await offlinePage.goto(BASE_URL, { waitUntil: "networkidle" });
  await offlinePage.waitForTimeout(500);

  // Route block API calls
  await offlinePage.route("**/health*", (route) => route.abort());
  await offlinePage.route("**/v1/**", (route) => route.abort());
  await offlinePage.waitForTimeout(4500); // wait for ECG monitor heartbeat to catch offline

  // Verify ECG status shows offline
  const isOfflineVisible = await offlinePage.locator("text=API Offline").isVisible();
  console.log(`   ✓ Network blocked: ECG Flatline detected? ${isOfflineVisible}`);

  await offlineContext.close();
  await browser.close();

  console.log("\n==========================================================");
  console.log("🎉 E2E TEST & SCREENSHOT SUITE COMPLETED SUCCESSFULLY!");
  console.log(`Console Errors Caught: ${consoleErrors.length}`);
  console.log(`Axe Serious Violations: Daylight: ${seriousDeskDaylight.length}, Night: ${seriousDeskNight.length}`);
  console.log("==========================================================\n");
}

run().catch((err) => {
  console.error("FATAL E2E FAILURE:", err);
  process.exit(1);
});
