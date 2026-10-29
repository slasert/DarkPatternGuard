import { test, expect } from "@playwright/test";

test("analyze page renders a mocked result", async ({ page }) => {
  await page.route("**/predict", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({
      prediction_id: 1, label: "Urgency", confidence_percent: 87.5,
      per_model: { logistic_regression: { label: "Urgency", probabilities: { Urgency: 0.875, Scarcity: 0.05, "Social Proof": 0.05, Normal: 0.025 }, top_contributing_words: ["hurry"] } },
      calibrated_ensemble: { label: "Urgency", probabilities: { Urgency: 0.875, Scarcity: 0.05, "Social Proof": 0.05, Normal: 0.025 }, top_contributing_words: ["hurry"] },
      top_contributing_words: ["hurry"],
    }) });
  });
  await page.goto("/");
  await page.getByLabel("E-commerce text").fill("Hurry, offer expires tonight!");
  await page.getByRole("button", { name: "Analyze copy" }).click();
  await expect(page.getByText("Urgency", { exact: true }).last()).toBeVisible();
  await expect(page.locator("strong", { hasText: "87.5%" })).toBeVisible();
  await expect(page.getByText("hurry", { exact: true })).toBeVisible();
});
