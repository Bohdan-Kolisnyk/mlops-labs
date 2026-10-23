# Trigger the ETL DAG once per month, one after another (PowerShell).
#   .\scripts\trigger_months.ps1 -From 2023-12 -To 2026-10
param(
    [string]$From = "2023-12",
    [string]$To = "2026-10",
    [string]$RawDir = ""
)
$d = [datetime]::ParseExact("$From-01", "yyyy-MM-dd", $null)
$end = [datetime]::ParseExact("$To-01", "yyyy-MM-dd", $null)
while ($d -le $end) {
    $month = $d.ToString("yyyy-MM")
    $conf = "{`"month`": `"$month`"}"
    if ($RawDir) { $conf = "{`"month`": `"$month`", `"raw_dir`": `"$RawDir`"}" }
    Write-Host "trigger $month"
    docker compose exec -T airflow-scheduler airflow dags trigger truth_social_etl --conf $conf
    $d = $d.AddMonths(1)
}
