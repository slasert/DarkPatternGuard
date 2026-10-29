import { test, expect } from "@playwright/test";

test("Robustness evaluation shows actual model attribution and a perturbed comparison", async ({ page }) => {
  await page.route("**/explain", async (route) => {
    expect(route.request().postDataJSON()).toEqual({ text: "🛒 midnight sale", perturbed_text: "plain item" });
    await route.fulfill({ json: {
      text: "🛒 midnight sale", model: { name: "fixture_bert", seed: 1729, checkpoint_sha256: "fixture-hash", tokenizer_sha256: "fixture-tokenizer" },
      label: "Urgency", confidence: 0.8, method: "word_deletion_probability_difference",
      description: "Model sensitivity, not causal evidence.", truncation_warning: "Truncation limit: 256 subwords.",
      feature_attributions: { target_class: "Urgency", bias: -0.1, logit: 0.4,
        description: "Exact TF-IDF times coefficient; not causal evidence.",
        features: [{ feature: "midnight", score: 0.5 }] },
      tokens: [{ text: "midnight", start: 2, end: 10, score: 0.5 }, { text: "sale", start: 11, end: 15, score: -0.02 }],
      comparison: { text: "plain item", label: "Normal", confidence: 0.7, prediction_changed: true, original_class_probability_delta: -0.6, note: "A semantic edit may change the correct label." },
    } });
  });
  await page.goto("/");
  await page.getByLabel("E-commerce text").fill("🛒 midnight sale");
  await page.getByLabel("Optional perturbed text").fill("plain item");
  await page.getByRole("button", { name: "Explain loaded model" }).click();
  await expect(page.getByLabel("Attributed text")).toHaveText("🛒 midnight sale");
  await expect(page.getByText("80.0% model confidence", { exact: false })).toBeVisible();
  await expect(page.getByText("fixture_bert", { exact: false })).toBeVisible();
  await expect(page.getByText("Model sensitivity, not causal evidence.")).toBeVisible();
  await expect(page.getByText("Prediction changed", { exact: false })).toBeVisible();
  await page.getByText("Word contribution values").click();
  await expect(page.getByRole("cell", { name: "-2.00", exact: true })).toBeVisible();
  await page.getByText("Exact TF-IDF feature contributions", { exact: true }).click();
  await expect(page.getByRole("cell", { name: "0.5000", exact: true })).toBeVisible();
  await page.screenshot({ path: "test-results/model-explanation.png", fullPage: true });
});

test("Robustness evaluation reports missing artifacts", async ({ page }) => {
  await page.route("**/explain", (route) => route.fulfill({ status: 503, json: { detail: "Trained weights unavailable" } }));
  await page.goto("/");
  await page.getByLabel("E-commerce text").fill("An item description");
  await page.getByRole("button", { name: "Explain loaded model" }).click();
  await expect(page.getByRole("alert")).toHaveText("Trained weights unavailable");
});
