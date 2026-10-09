// Playwright E2E smoke tests and screenshot generator for ReadmitIQ
const path = require('path');
const fs = require('fs');

const playwrightPath = 'C:/Users/tribh/AppData/Local/npm-cache/_npx/e41f203b7505f1fb/node_modules/playwright';
const { chromium } = require(playwrightPath);

const SCREENSHOT_DIR = path.resolve(__dirname, '../docs/screenshots');
if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

const BASE_URL = 'http://127.0.0.1:5173/';

async function runTests() {
  console.log('====================================================');
  console.log('🚀 Starting ReadmitIQ E2E Smoke & Screenshot Suite');
  console.log('====================================================\n');

  const browser = await chromium.launch({
    channel: 'msedge',
    headless: true,
  });

  const viewports = [
    { name: 'desktop', width: 1280, height: 800 },
    { name: 'mobile', width: 375, height: 812 },
  ];

  let passedTests = 0;
  let totalTests = 0;

  for (const vp of viewports) {
    console.log(`\n--- Testing Viewport: ${vp.name.toUpperCase()} (${vp.width}x${vp.height}) ---`);
    const context = await browser.newContext({
      viewport: { width: vp.width, height: vp.height },
    });
    const page = await context.newPage();

    // Catch page errors
    page.on('pageerror', (err) => console.error(`  [PAGE ERROR ${vp.name}]:`, err.message));

    // Load initial page (Cohort Triage)
    await page.goto(BASE_URL, { waitUntil: 'networkidle', timeout: 20000 });
    await page.waitForTimeout(1000);

    const prefix = vp.name === 'mobile' ? 'mobile-tab' : 'nav-tab';

    // 1. Cohort Triage Page
    totalTests++;
    try {
      console.log(`[${vp.name}] Verifying Cohort Triage page & Impact Simulation...`);
      await page.waitForSelector('text=Cohort Command Center', { timeout: 12000 });
      await page.waitForSelector('text=Illustrative What-If Scenario', { timeout: 12000 });
      await page.waitForSelector('text=Projected Clinical & Economic Impact Estimate', { timeout: 12000 });

      const shotPath = path.join(SCREENSHOT_DIR, `cohort_triage_${vp.name}.png`);
      await page.screenshot({ path: shotPath, fullPage: true });
      console.log(`  ✓ Cohort Triage passed. Saved: ${path.basename(shotPath)}`);
      passedTests++;
    } catch (err) {
      console.error(`  ✗ Cohort Triage failed on ${vp.name}:`, err.message);
    }

    // Light Theme Test (run immediately on Cohort desktop view)
    if (vp.name === 'desktop') {
      totalTests++;
      try {
        console.log(`[desktop] Testing Light Theme toggle...`);
        const themeBtn = page.locator('#btn-theme-toggle');
        await themeBtn.click();
        await page.waitForTimeout(600);

        const shotPath = path.join(SCREENSHOT_DIR, 'cohort_triage_light_desktop.png');
        await page.screenshot({ path: shotPath, fullPage: true });
        console.log(`  ✓ Light theme toggle passed. Saved: ${path.basename(shotPath)}`);
        passedTests++;

        // Switch back to dark theme
        await themeBtn.click();
        await page.waitForTimeout(400);
      } catch (err) {
        console.error(`  ✗ Light theme test failed:`, err.message);
      }
    }

    // 2. Patient Risk & Suggested Interventions
    totalTests++;
    try {
      console.log(`[${vp.name}] Navigating to Patient page & running decision support...`);
      const patientBtn = page.locator(`#${prefix}-patient`);
      await patientBtn.click();
      await page.waitForTimeout(600);

      await page.waitForSelector('text=Patient Risk Profile & Clinical Precedent RAG', { timeout: 12000 });

      // Click High-Risk Preset
      const highRiskPreset = page.locator('#btn-preset-0');
      await highRiskPreset.click();
      await page.waitForTimeout(600);

      // Click Run Prediction
      const runPredictBtn = page.locator('#btn-run-prediction');
      await runPredictBtn.click();

      // Wait for results
      await page.waitForSelector('#suggested-interventions-panel', { timeout: 20000 });
      await page.waitForSelector('text=Decision-support suggestions, not clinical advice', { timeout: 12000 });
      await page.waitForSelector('text=Suggested Care-Transition Follow-Up Actions', { timeout: 12000 });

      const shotPath = path.join(SCREENSHOT_DIR, `patient_risk_${vp.name}.png`);
      await page.screenshot({ path: shotPath, fullPage: true });
      console.log(`  ✓ Patient Risk & Interventions passed. Saved: ${path.basename(shotPath)}`);
      passedTests++;
    } catch (err) {
      console.error(`  ✗ Patient Risk failed on ${vp.name}:`, err.message);
    }

    // 3. Model Lab & Ablation Study
    totalTests++;
    try {
      console.log(`[${vp.name}] Navigating to Model Lab page...`);
      const labBtn = page.locator(`#${prefix}-lab`);
      await labBtn.click();
      await page.waitForTimeout(600);

      await page.waitForSelector('text=Model Lab & Honest RAG Ablation Analysis', { timeout: 12000 });
      await page.waitForSelector('text=LightGBM + Structured RAG', { timeout: 12000 });

      const shotPath = path.join(SCREENSHOT_DIR, `model_lab_${vp.name}.png`);
      await page.screenshot({ path: shotPath, fullPage: true });
      console.log(`  ✓ Model Lab passed. Saved: ${path.basename(shotPath)}`);
      passedTests++;
    } catch (err) {
      console.error(`  ✗ Model Lab failed on ${vp.name}:`, err.message);
    }

    // 4. Fairness, Bias Audit & Drift
    totalTests++;
    try {
      console.log(`[${vp.name}] Navigating to Fairness & Drift page...`);
      const fairnessBtn = page.locator(`#${prefix}-fairness`);
      await fairnessBtn.click();
      await page.waitForTimeout(600);

      await page.waitForSelector('text=Fairness Audit & Dataset Drift Telemetry', { timeout: 12000 });
      await page.waitForSelector('text=Population Stability Index (PSI) Drift Telemetry', { timeout: 12000 });

      const shotPath = path.join(SCREENSHOT_DIR, `fairness_drift_${vp.name}.png`);
      await page.screenshot({ path: shotPath, fullPage: true });
      console.log(`  ✓ Fairness & Drift passed. Saved: ${path.basename(shotPath)}`);
      passedTests++;
    } catch (err) {
      console.error(`  ✗ Fairness & Drift failed on ${vp.name}:`, err.message);
    }

    // 5. How It Works Page
    totalTests++;
    try {
      console.log(`[${vp.name}] Navigating to How It Works page...`);
      const howBtn = page.locator(`#${prefix}-how-it-works`);
      await howBtn.click();
      await page.waitForTimeout(600);

      await page.waitForSelector('text=How ReadmitIQ Works', { timeout: 12000 });
      await page.waitForSelector('text=Cryptographic & Algorithmic Leakage Safeguards', { timeout: 12000 });
      await page.waitForSelector('text=Honest Clinical & Methodological Limitations', { timeout: 12000 });

      const shotPath = path.join(SCREENSHOT_DIR, `how_it_works_${vp.name}.png`);
      await page.screenshot({ path: shotPath, fullPage: true });
      console.log(`  ✓ How It Works passed. Saved: ${path.basename(shotPath)}`);
      passedTests++;
    } catch (err) {
      console.error(`  ✗ How It Works failed on ${vp.name}:`, err.message);
    }

    await context.close();
  }

  await browser.close();

  console.log('\n====================================================');
  console.log(`🏁 Smoke Suite Completed: ${passedTests}/${totalTests} Tests Passed`);
  console.log('====================================================');

  if (passedTests !== totalTests) {
    process.exit(1);
  }
}

runTests().catch((err) => {
  console.error('Fatal test error:', err);
  process.exit(1);
});
