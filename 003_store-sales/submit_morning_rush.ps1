# 003_store-sales: Morning Rush Submissions (JST 9:00 / UTC 0:00 Reset)
# Run with: pwsh 003_store-sales/submit_morning_rush.ps1

Write-Host "=== Store Sales: Checking Submission Limits ===" -ForegroundColor Cyan
uv run kaggle competitions submission-limits -c store-sales-time-series-forecasting

Write-Host "`nReady to submit:" -ForegroundColor Yellow
Write-Host "  [1] Exp #040: 8-Model 33 Family-Adaptive SLSQP Blend (Primary Candidate)"
Write-Host "  [2] Exp #039: Regional School Disentangled Golden Blend (Secondary Candidate)"
Write-Host "  [3] Exp #038: Colab L4 GPU Regional School 5-Seeds GBDT (Single Record Val 0.39116)"

$confirm = Read-Host "`nSubmit Exp #040 now? (y/n)"
if ($confirm -eq 'y') {
    Write-Host "`nSubmitting Exp #040..." -ForegroundColor Green
    uv run kaggle competitions submit -c store-sales-time-series-forecasting -f 003_store-sales/submission_exp040_8models_slsqp_blend.csv -m "Exp040: 8-Model 33 Family-Adaptive SLSQP Blend (integrating Exp038 Regional School Val 0.39116)"
    Start-Sleep -Seconds 10
    uv run kaggle competitions submissions -c store-sales-time-series-forecasting
}
